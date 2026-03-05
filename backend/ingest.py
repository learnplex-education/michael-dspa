#!/usr/bin/env python3
"""Standalone ingestion script — scrapes web pages and loads local files into ChromaDB."""

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
from langchain_chroma import Chroma  # noqa: E402

from config import CHROMA_DIR, KNOWLEDGE_BASE_DIR, MANIFEST_PATH, URLS  # noqa: E402

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
    os.makedirs(CHROMA_DIR, exist_ok=True)
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

    logger.info(
        f"\nTotal: {len(all_chunks)} chunks "
        f"({len(all_chunks) - len(local_chunks)} web + {len(local_chunks)} local)"
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(all_chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    _save_manifest()
    logger.info(f"✅ Vector store persisted to {CHROMA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest data sources for the DSPA Bot")
    parser.add_argument(
        "--force", action="store_true", help="Delete existing vector store and rebuild"
    )
    args = parser.parse_args()

    if os.path.exists(CHROMA_DIR) and not args.force:
        logger.info(
            f"Vector store already exists at {CHROMA_DIR}.\n"
            "Use --force to delete and rebuild."
        )
        return

    if os.path.exists(CHROMA_DIR):
        logger.info("Removing existing vector store...")
        shutil.rmtree(CHROMA_DIR)

    build()


if __name__ == "__main__":
    main()
