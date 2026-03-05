"""FastAPI backend — streaming /chat endpoint for the DSPA Bot."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("USER_AGENT", "Learnplex-DataScience-Bot/1.0")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_core.messages import SystemMessage, HumanMessage  # noqa: E402

from config import CHROMA_DIR, MAX_QUERIES_PER_SESSION, SYSTEM_PROMPT  # noqa: E402

app = FastAPI(title="DSPA Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STREAM_HEADERS = {
    "x-vercel-ai-ui-message-stream": "v1",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}
STREAM_MEDIA = "text/event-stream; charset=utf-8"

sessions: dict[str, int] = defaultdict(int)
vectorstore: Chroma | None = None
llm: ChatOpenAI | None = None


def _sse(payload: dict | str) -> str:
    """Format a single Server-Sent Event line."""
    return f"data: {json.dumps(payload) if isinstance(payload, dict) else payload}\n\n"


def _extract_question(messages: list[dict]) -> str:
    """Pull the user's question from the last message (v4 or v5 format)."""
    if not messages:
        return ""
    last = messages[-1]
    parts = last.get("parts", [])
    if parts:
        return " ".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
    return last.get("content", "")


@app.on_event("startup")
async def startup() -> None:
    global vectorstore, llm
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            f"Vector store not found at {CHROMA_DIR}. "
            "Run `python ingest.py` first."
        )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)


@app.get("/session")
async def get_session(request: Request):
    session_id = request.headers.get("X-Session-ID", "anonymous")
    return {
        "queries_used": sessions[session_id],
        "max_queries": MAX_QUERIES_PER_SESSION,
    }


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    session_id = request.headers.get("X-Session-ID", "anonymous")

    if sessions[session_id] >= MAX_QUERIES_PER_SESSION:
        async def limit_stream():
            mid = uuid.uuid4().hex
            tid = uuid.uuid4().hex
            msg = (
                "You've reached the query limit for this session. As a nonprofit, "
                "Learnplex limits queries to keep this tool free for everyone. "
                "Please refresh to start over or contact us to learn more."
            )
            yield _sse({"type": "start", "messageId": mid})
            yield _sse({"type": "start-step"})
            yield _sse({"type": "text-start", "id": tid})
            yield _sse({"type": "text-delta", "id": tid, "delta": msg})
            yield _sse({"type": "text-end", "id": tid})
            yield _sse({"type": "finish-step"})
            yield _sse({"type": "finish"})
            yield _sse("[DONE]")

        return StreamingResponse(limit_stream(), media_type=STREAM_MEDIA, headers=STREAM_HEADERS)

    messages = body.get("messages", [])
    question = _extract_question(messages)

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.25},
    )
    docs = await asyncio.to_thread(retriever.invoke, question)

    context_parts: list[str] = []
    seen_sources: set[str] = set()
    sources: list[dict] = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        src_type = doc.metadata.get("type", "Official Website")
        context_parts.append(
            f"[Source: {src} | Type: {src_type}]\n{doc.page_content}"
        )
        if src not in seen_sources:
            seen_sources.add(src)
            sources.append({"source": src, "type": src_type})

    context = "\n\n".join(context_parts)

    async def generate():
        mid = uuid.uuid4().hex
        tid = uuid.uuid4().hex

        yield _sse({"type": "start", "messageId": mid})
        yield _sse({"type": "start-step"})
        yield _sse({"type": "text-start", "id": tid})

        chat_messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=question),
        ]

        async for chunk in llm.astream(chat_messages):
            token = chunk.content
            if token:
                yield _sse({"type": "text-delta", "id": tid, "delta": token})

        yield _sse({"type": "text-end", "id": tid})

        for s in sources:
            if s["type"] == "Official Website" and s["source"].startswith("http"):
                yield _sse({
                    "type": "source-url",
                    "sourceId": s["source"],
                    "url": s["source"],
                })
            else:
                yield _sse({
                    "type": "source-document",
                    "sourceId": s["source"],
                    "mediaType": "text/plain",
                    "title": "Peer Advising Archive",
                })

        yield _sse({"type": "finish-step"})
        yield _sse({"type": "finish"})
        yield _sse("[DONE]")

        sessions[session_id] += 1

    return StreamingResponse(generate(), media_type=STREAM_MEDIA, headers=STREAM_HEADERS)
