import json
import logging
import os
from glob import glob

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader

    _DOCX_LOADER = UnstructuredWordDocumentLoader
except ImportError:
    from langchain_community.document_loaders import Docx2txtLoader

    _DOCX_LOADER = Docx2txtLoader
    logging.warning(
        "unstructured is not installed — falling back to Docx2txtLoader for .docx files. "
        "Install with: pip install unstructured python-docx"
    )

logger = logging.getLogger(__name__)

CHROMA_DIR = "./chroma_db"
KNOWLEDGE_BASE_DIR = "./knowledge_base"
MANIFEST_PATH = os.path.join(CHROMA_DIR, "source_manifest.json")

URLS = [
    "https://cdss.berkeley.edu/dsus/academics/data-science-major",
    "https://cdss.berkeley.edu/dsus/academics/requirements-lower-division",
    "https://cdss.berkeley.edu/dsus/academics/majorrequirements",
    "https://cdss.berkeley.edu/dsus/academics/domain-emphasis",
    "https://cdss.berkeley.edu/dsus/advising/",
    "https://cdss.berkeley.edu/dsus/advising/data-science-faqs",
    "https://cdss.berkeley.edu/dsus/advising/data-science-peer-advising-dspa",
    "https://cdss.berkeley.edu/dsus/advising",
    "https://cdss.berkeley.edu/dsus/advising/data-science-advising",
]

FILE_LOADERS = {
    "**/*.pdf": PyPDFLoader,
    "**/*.docx": _DOCX_LOADER,
    "**/*.txt": TextLoader,
}

SYSTEM_PROMPT = """\
You are a strict but friendly UC Berkeley Data Science Peer Advisor. \
Use warm, student-friendly language, but NEVER validate or confirm a course, \
policy, or requirement unless it is explicitly listed in the provided context. \
If a user asks about a course that is NOT in the provided context, you MUST \
say you cannot find it in the official records. Accuracy is your top priority.

RULES YOU MUST FOLLOW:
1. Every response must start or end with exactly this disclaimer:
   "I am an AI assistant. For official decisions, please consult with a \
Data Science Major Advisor or email ds-advising@berkeley.edu."
2. If a student asks for a 2-year or 4-year plan, tell them to book an \
appointment with a Major Advisor.
3. Use Markdown formatting. Bold course names (e.g. **DATA C8**, **MATH 54**).
4. If a student asks about financial aid or mental health, redirect them:
   - Financial aid / basic needs → Basic Needs Office \
(https://basicneeds.berkeley.edu)
   - Mental health / personal safety → Path to Care Office \
(https://care.berkeley.edu)
5. If the question is outside the scope of what can be answered with the CDSS \
website, let the student know that it is outside the scope of this tool.
6. When citing sources, check the "type" and "source" metadata on each chunk:
   - If type is "Official Website", cite the specific URL \
(e.g. "According to the Domain Emphasis page \
(https://cdss.berkeley.edu/dsus/academics/domain-emphasis)…").
   - If type is "Peer Advising Archive", say: \
"Based on Peer Advising records…" and mention the filename from the source \
metadata if available.
7. When listing course requirements, carefully check the retrieved context for \
ALL alternative options. For example, the linear algebra requirement can be \
satisfied by **MATH 54**, **MATH 56**, **PHYSICS 89**, or **EECS 16A**. \
Do not omit alternatives that appear in the context.
8. Temporal awareness for course planning advice:
   - **DATA C8** is a foundational course typically taken in freshman year. \
Do not suggest it as a sophomore-year priority unless the student hasn't \
taken it yet.
   - Sophomore year should focus on completing Lower-Division Foundations: \
**CS 61B**, a linear algebra course (**MATH 54**, **MATH 56**, or \
**PHYSICS 89**), and ideally **DATA C100**.
   - If a student asks about "what to take sophomore year", emphasize these \
lower-division foundations rather than repeating DATA C8.
9. Fact-check rule — NEVER confirm a course counts toward the major unless \
that exact course name and number appears in the retrieved context. If a \
user asks about a specific course (e.g. STAT 188) that is NOT explicitly \
listed in the provided context, you MUST respond: \
"I don't see [COURSE] in the currently approved lists. I recommend checking \
the official Domain Emphasis page or consulting a Major Advisor to confirm."
10. Source verification — before answering "yes" or "no" about whether a \
course satisfies a requirement, cross-reference the course number against \
every retrieved chunk. Only confirm if you find an explicit match. If the \
course appears nowhere in the context, say so clearly instead of guessing.

Use only the following context to answer. Each chunk includes "source" and \
"type" metadata fields — always cite them appropriately. \
If the context does not contain enough information, say so honestly — \
never fabricate or assume information that is not present.

Context:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _list_local_files() -> list[str]:
    """Return paths of all supported files in the knowledge base folder."""
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        return []
    files = []
    for pattern in FILE_LOADERS:
        files.extend(glob(os.path.join(KNOWLEDGE_BASE_DIR, pattern), recursive=True))
    return sorted(files)


def count_sources() -> int:
    """Return the total number of source inputs (web URLs + local files)."""
    return len(URLS) + len(_list_local_files())


def _save_manifest() -> None:
    """Persist the current source count so we can detect changes on next startup."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump({"source_count": count_sources()}, f)


def _load_local_docs(splitter: RecursiveCharacterTextSplitter) -> list:
    """Load and chunk .pdf, .docx, and .txt files from the knowledge_base/ folder."""
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        logger.warning("knowledge_base/ folder not found — skipping local ingestion.")
        return []

    local_files = _list_local_files()
    if not local_files:
        logger.warning(
            "knowledge_base/ exists but contains no supported files — skipping."
        )
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
        print(f"  ✓ {len(split):>3} chunks from {len(docs)} local .{ext} file(s)")

    return chunks


def build_vector_store() -> Chroma:
    """Scrape all source pages, load local files, chunk, and persist to ChromaDB."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    print("Loading web pages...")
    for url in URLS:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
                doc.metadata["type"] = "Official Website"
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"  ✓ {len(chunks):>3} chunks from {url}")
        except Exception as exc:
            print(f"  ✗ Failed to load {url}: {exc}")

    print("Loading local knowledge base...")
    local_chunks = _load_local_docs(splitter)
    all_chunks.extend(local_chunks)

    print(
        f"\nTotal: {len(all_chunks)} chunks "
        f"({len(all_chunks) - len(local_chunks)} web + {len(local_chunks)} local)"
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        all_chunks, embedding=embeddings, persist_directory=CHROMA_DIR
    )
    _save_manifest()
    return vectorstore


def load_vector_store() -> Chroma:
    """Load an existing ChromaDB from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def get_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.25},
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )


def main():
    load_dotenv()
    os.environ.setdefault(
        "USER_AGENT", os.getenv("USER_AGENT", "Learnplex-DataScience-Bot/1.0")
    )

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: set the OPENAI_API_KEY environment variable first.")
        return

    if os.path.exists(CHROMA_DIR):
        print("Found existing vector store — loading from disk.")
        vectorstore = load_vector_store()
    else:
        print("No vector store found — scraping and building one...")
        vectorstore = build_vector_store()

    qa = get_qa_chain(vectorstore)

    print("\nAsk anything about the UC Berkeley Data Science major.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = qa.invoke(question)
        print(f"\nAnswer: {result['result']}\n")


if __name__ == "__main__":
    main()
