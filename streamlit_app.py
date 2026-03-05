import os
import json
import shutil
import time

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USER_AGENT", os.getenv("USER_AGENT", "Learnplex-DataScience-Bot/1.0"))

import streamlit as st  # noqa: E402

from app import (  # noqa: E402
    CHROMA_DIR,
    MANIFEST_PATH,
    build_vector_store,
    count_sources,
    load_vector_store,
    get_qa_chain,
)

MAX_QUERIES = 10

SAMPLE_QUERIES = [
    "Linear Algebra Options",
    "The C- Rule",
    "Data Science Clubs",
]


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _should_rebuild() -> bool:
    """Return True if the vector store is missing or the source count changed."""
    if not os.path.exists(CHROMA_DIR):
        return True
    if not os.path.exists(MANIFEST_PATH):
        return True
    try:
        with open(MANIFEST_PATH) as f:
            saved = json.load(f)
        return saved.get("source_count") != count_sources()
    except (json.JSONDecodeError, OSError):
        return True


def _init_qa_chain():
    if _should_rebuild():
        if os.path.exists(CHROMA_DIR):
            print("Source count changed — rebuilding vector store...")
            shutil.rmtree(CHROMA_DIR)
        else:
            print("No vector store found — building one...")
        vectorstore = build_vector_store()
    else:
        print("Vector store up to date — loading from disk.")
        vectorstore = load_vector_store()
    return get_qa_chain(vectorstore)


@st.cache_resource
def get_chain():
    return _init_qa_chain()


# ---------------------------------------------------------------------------
# Source extraction & rendering
# ---------------------------------------------------------------------------

def _extract_sources(source_docs: list) -> list[dict]:
    """Deduplicate retrieved chunks into unique source entries."""
    seen = set()
    sources = []
    for doc in source_docs:
        meta = doc.metadata
        src = meta.get("source", "Unknown")
        src_type = meta.get("type", "Official Website")
        if src in seen:
            continue
        seen.add(src)
        sources.append({"source": src, "type": src_type})
    return sources


def _render_sources(sources: list[dict]) -> None:
    """Render an inline 'Sources' section beneath an assistant message."""
    if not sources:
        return
    st.markdown("---")
    st.markdown("**Sources**")
    for entry in sources:
        src = entry["source"]
        if entry["type"] == "Peer Advising Archive":
            st.markdown(f"📍 Peer Advising Archive — `{src}`")
        elif src.startswith("http"):
            st.markdown(f"🔗 Official CDSS Website — [{src}]({src})")
        else:
            st.markdown(f"📄 `{src}`")


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def _render_feedback(msg_idx: int) -> None:
    """Render thumbs-up / thumbs-down buttons for a given message index."""
    fb_key = f"feedback_{msg_idx}"
    if fb_key not in st.session_state:
        st.session_state[fb_key] = None

    if st.session_state[fb_key] == "down":
        st.info(
            "Thanks for the feedback! This helps our Fellows improve the bot "
            "for their next Dev Diary entry!"
        )
        return
    if st.session_state[fb_key] == "up":
        st.success("Glad that was helpful!")
        return

    cols = st.columns([0.12, 0.12, 0.76])
    with cols[0]:
        if st.button("👍", key=f"up_{msg_idx}"):
            st.session_state[fb_key] = "up"
            st.rerun()
    with cols[1]:
        if st.button("👎", key=f"down_{msg_idx}"):
            st.session_state[fb_key] = "down"
            st.rerun()


# ---------------------------------------------------------------------------
# Query runner with st.status
# ---------------------------------------------------------------------------

def _run_query(qa, prompt: str) -> tuple[str, list[dict]]:
    """Execute the QA chain inside an st.status container for visual feedback."""
    with st.status("Researching your question...", expanded=True) as status:
        status.update(label="Searching official CDSS records...")
        time.sleep(0.3)

        result = qa.invoke(prompt)

        status.update(label="Consulting Peer Advising Archive...")
        time.sleep(0.3)

        answer = result["result"]
        sources = _extract_sources(result.get("source_documents", []))

        status.update(label="Done!", state="complete", expanded=False)

    return answer, sources


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="DSPA Bot", page_icon="🐻", layout="centered")
    st.title("UC Berkeley Data Science Peer Advisor")
    st.caption("Ask me anything about the Data Science major!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # --- Sidebar: Usage Tracker ---
    with st.sidebar:
        st.header("Usage Tracker")
        st.progress(st.session_state.query_count / MAX_QUERIES)
        st.markdown(
            f"**Queries used: {st.session_state.query_count}/{MAX_QUERIES}**"
        )

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Set the `OPENAI_API_KEY` environment variable in your .env file.")
        return

    qa = get_chain()

    # --- Render chat history ---
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_sources(msg.get("sources", []))
                _render_feedback(idx)

    limit_reached = st.session_state.query_count >= MAX_QUERIES

    if limit_reached:
        st.warning(
            "You've reached the limit for this session. As a nonprofit, "
            "Learnplex limits queries to keep this tool free for everyone. "
            "Please refresh to start over or contact us to learn more."
        )

    # --- Sample query bubbles ---
    if not limit_reached and not st.session_state.messages:
        st.markdown("**Try a sample question:**")
        cols = st.columns(len(SAMPLE_QUERIES))
        for col, query in zip(cols, SAMPLE_QUERIES):
            with col:
                if st.button(query, use_container_width=True):
                    st.session_state.pending_query = query
                    st.rerun()

    # --- Handle pending query from bubble click ---
    pending = st.session_state.pending_query
    if pending and not limit_reached:
        st.session_state.pending_query = None
        _handle_query(qa, pending)

    # --- Chat input ---
    if prompt := st.chat_input(
        "Ask a question about the DS major...",
        disabled=limit_reached,
    ):
        _handle_query(qa, prompt)


def _handle_query(qa, prompt: str) -> None:
    """Process a user query: display it, run RAG, render answer + sources."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer, sources = _run_query(qa, prompt)
        st.markdown(answer)
        _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.session_state.query_count += 1
    st.rerun()


if __name__ == "__main__":
    main()
