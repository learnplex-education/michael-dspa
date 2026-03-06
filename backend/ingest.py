#!/usr/bin/env python3
"""Standalone ingestion script — scrapes web pages and loads local files into Pinecone."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from glob import glob
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("USER_AGENT", "Learnplex-DataScience-Bot/1.0")

from langchain_community.document_loaders import (  # noqa: E402
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from langchain_openai import OpenAIEmbeddings  # noqa: E402
from pinecone import Pinecone  # noqa: E402

from config import KNOWLEDGE_BASE_DIR, MANIFEST_PATH, URLS  # noqa: E402

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader

    _DOCX_LOADER = UnstructuredWordDocumentLoader
except ImportError:
    from langchain_community.document_loaders import Docx2txtLoader

    _DOCX_LOADER = Docx2txtLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

FILE_LOADERS = {
    "**/*.pdf": PyPDFLoader,
    "**/*.docx": _DOCX_LOADER,
    "**/*.txt": TextLoader,
}


def _list_local_files() -> list[str]:
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        return []
    files = []
    for pattern in FILE_LOADERS:
        files.extend(glob(os.path.join(KNOWLEDGE_BASE_DIR, pattern), recursive=True))
    return sorted(files)


def count_sources() -> int:
    return len(URLS) + len(_list_local_files())


def _save_manifest() -> None:
    # Persist a simple manifest so we can detect when sources change and trigger re-ingestion.
    manifest_dir = os.path.dirname(MANIFEST_PATH)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump({"source_count": count_sources()}, f)


def _load_local_docs(splitter: RecursiveCharacterTextSplitter) -> list:
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        logger.warning("  ⓘ knowledge_base/ not found — skipping local files.")
        return []

    local_files = _list_local_files()
    if not local_files:
        logger.warning("  ⓘ knowledge_base/ is empty — skipping local files.")
        return []

    chunks = []
    for pattern, loader_cls in FILE_LOADERS.items():
        loader = DirectoryLoader(
            KNOWLEDGE_BASE_DIR,
            glob=pattern,
            loader_cls=loader_cls,
            silent_errors=True,
            recursive=True,
        )
        docs = loader.load()
        if not docs:
            continue

        for doc in docs:
            doc.metadata["source"] = doc.metadata.get("source", "Peer Advising Archive")
            doc.metadata["type"] = "Peer Advising Archive"

        split = splitter.split_documents(docs)
        chunks.extend(split)
        ext = pattern.split("*.")[-1]
        logger.info(f"  ✓ {len(split):>3} chunks from {len(docs)} local .{ext} file(s)")

    return chunks


def build() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    if not os.environ.get("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    if not os.environ.get("PINECONE_HOST"):
        logger.error("PINECONE_HOST not set. Copy the Host URL from your Pinecone dashboard into .env.")
        sys.exit(1)

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    logger.info("Loading web pages...")
    for url in URLS:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
                doc.metadata["type"] = "Official Website"
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            logger.info(f"  ✓ {len(chunks):>3} chunks from {url}")
        except Exception as exc:
            logger.error(f"  ✗ Failed: {url} — {exc}")

    logger.info("Loading local knowledge base...")
    local_chunks = _load_local_docs(splitter)
    all_chunks.extend(local_chunks)

    total_web = len(all_chunks) - len(local_chunks)
    total_local = len(local_chunks)
    logger.info(
        f"\nTotal: {len(all_chunks)} chunks "
        f"({total_web} web + {total_local} local)"
    )

    # Embed all chunks with OpenAI and push to Pinecone
    logger.info("Embedding chunks with OpenAI and upserting to Pinecone...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [doc.page_content for doc in all_chunks]
    vectors = embeddings.embed_documents(texts)

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_host = os.environ.get("PINECONE_HOST")
    index = pc.Index(host=index_host)

    pinecone_vectors = []
    for i, (doc, vec) in enumerate(zip(all_chunks, vectors)):
        metadata = {
            "source": doc.metadata.get("source", ""),
            "type": doc.metadata.get("type", ""),
            "text": doc.page_content,
        }
        pinecone_vectors.append((f"chunk-{i}", vec, metadata))

    # Upsert in a single batch; Pinecone will overwrite any existing ids.
    index.upsert(vectors=pinecone_vectors)

    _save_manifest()
    logger.info("✅ Vector store persisted to Pinecone index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest data sources for the DSPA Bot")
    parser.add_argument(
        "--force", action="store_true", help="Delete existing vector store and rebuild"
    )
    args = parser.parse_args()

    if os.path.exists(MANIFEST_PATH) and not args.force:
        logger.info(
            f"Existing Pinecone manifest found at {MANIFEST_PATH}.\n"
            "Use --force to re-embed and upsert all chunks."
        )
        return

    build()


if __name__ == "__main__":
    main()
