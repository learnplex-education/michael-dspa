"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { motion, AnimatePresence } from "framer-motion";
import {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
  type FormEvent,
} from "react";
import ReactMarkdown from "react-markdown";
import { useSession, signIn, signOut, getSession } from "next-auth/react";

const MAX_QUERIES = 10;
const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://michael-dspa-backend.onrender.com";

const SAMPLE_QUERIES = [
  {
    label: "Linear Algebra Options",
    query:
      "What are the linear algebra course options for the Data Science major?",
  },
  {
    label: "The C- Rule",
    query: "What is the C- rule for Data Science major requirements?",
  },
  {
    label: "Data Science Clubs",
    query:
      "What Data Science student clubs and organizations are at Berkeley?",
  },
];

const QUICK_ACTIONS = [
  { label: "Major Requirements", query: "Major Requirements" },
  { label: "Research & Orgs", query: "Research & Orgs" },
  { label: "Course Planning", query: "Course Planning" },
];

function generateSessionId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

/* ─── Loading Indicator ─── */

function LoadingSteps() {
  const [step, setStep] = useState(0);
  const steps = [
    "Searching official CDSS records…",
    "Consulting Peer Advising Notes…",
    "Generating a helpful response…",
  ];

  useEffect(() => {
    const delays = [1200, 800, Infinity];
    let timeout: ReturnType<typeof setTimeout>;
    const advance = (current: number) => {
      if (current < steps.length - 1) {
        timeout = setTimeout(() => {
          setStep(current + 1);
          advance(current + 1);
        }, delays[current]);
      }
    };
    advance(0);
    return () => clearTimeout(timeout);
  }, [steps.length]);

  return (
    <div className="bg-white border border-gray-200 rounded-2xl px-5 py-4 shadow-sm max-w-[85%] md:max-w-[70%]">
      <div className="space-y-2">
        {steps.map((label, i) => (
          <motion.div
            key={label}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: i <= step ? 1 : 0.3, x: 0 }}
            transition={{ duration: 0.3, delay: i * 0.08 }}
            className="flex items-center gap-2 text-sm"
          >
            {i < step ? (
              <span className="text-green-500 text-xs">✓</span>
            ) : i === step ? (
              <motion.span
                animate={{ opacity: [0.4, 1, 0.4] }}
                transition={{ duration: 1.2, repeat: Infinity }}
                className="text-berkeley-blue text-xs"
              >
                ●
              </motion.span>
            ) : (
              <span className="text-gray-300 text-xs">○</span>
            )}
            <span className={i <= step ? "text-gray-500" : "text-gray-400"}>
              {label}
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

/* ─── Source Chip ─── */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function SourceChip({ part }: { part: any }) {
  if (part.type === "source-url") {
    return (
      <a
        href={part.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-blue-50 text-blue-700 text-xs font-medium border border-blue-200 hover:bg-blue-100 transition-colors"
      >
        🔗 CDSS Website
      </a>
    );
  }

  if (part.type === "source-document") {
    return (
      <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-amber-50 text-amber-800 text-xs font-medium border border-amber-200">
        📍 {part.title || "Peer Advising Archive"}
      </span>
    );
  }

  return null;
}

/* ─── Feedback ─── */

function FeedbackButtons({
  messageId,
  feedback,
  onFeedback,
}: {
  messageId: string;
  feedback: Record<string, "up" | "down">;
  onFeedback: (id: string, v: "up" | "down") => void;
}) {
  const current = feedback[messageId];

  if (current === "down") {
    return (
      <p className="mt-2 text-xs text-amber-600 italic">
        Thanks for the feedback! This helps our Fellows improve the bot for
        their next Dev Diary entry!
      </p>
    );
  }
  if (current === "up") {
    return (
      <p className="mt-2 text-xs text-green-600 italic">
        Glad that was helpful!
      </p>
    );
  }

  return (
    <div className="mt-3 flex items-center gap-1">
      <span className="text-xs text-gray-400 mr-1">Was this helpful?</span>
      <button
        onClick={() => onFeedback(messageId, "up")}
        className="p-1 rounded hover:bg-gray-100 transition-colors text-sm"
      >
        👍
      </button>
      <button
        onClick={() => onFeedback(messageId, "down")}
        className="p-1 rounded hover:bg-gray-100 transition-colors text-sm"
      >
        👎
      </button>
    </div>
  );
}

/* ─── Helpers ─── */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getTextContent(msg: any): string {
  if (msg.parts) {
    return msg.parts
      .filter((p: { type: string }) => p.type === "text")
      .map((p: { text: string }) => p.text)
      .join("");
  }
  return msg.content ?? "";
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getSourceParts(msg: any): any[] {
  if (!msg.parts) return [];
  return msg.parts.filter(
    (p: { type: string }) =>
      p.type === "source-url" || p.type === "source-document",
  );
}

/* ─── Main Page ─── */

export default function Home() {
  const { data: session, status: authStatus } = useSession();
  const [sessionId] = useState(generateSessionId);
  const [queryCount, setQueryCount] = useState(0);
  const [input, setInput] = useState("");
  const [feedback, setFeedback] = useState<Record<string, "up" | "down">>({});
  const scrollRef = useRef<HTMLDivElement>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  const limitReached = queryCount >= MAX_QUERIES;

  const isSessionReady =
    authStatus === "authenticated" && !!session?.idToken;

  const transport = useMemo(() => {
    const chatUrl = `${API_URL}/chat`;
    return new DefaultChatTransport({
      api: chatUrl,
      headers: {
        "Content-Type": "application/json",
        "X-Session-ID": sessionId,
      },
      fetch: async (input, init = {}) => {
        const freshSession = await getSession();
        init.headers = {
          ...(init.headers || {}),
          Authorization: freshSession?.idToken
            ? `Bearer ${freshSession.idToken}`
            : "",
        };
        const response = await fetch(input, init);
        if (response.status === 429) {
          throw new Error(
            "Whoa there! You're sending messages too fast. Please wait a minute.",
          );
        }
        return response;
      },
    });
  }, [sessionId, session?.idToken]);

  const {
    messages,
    sendMessage,
    status: chatStatus,
  } = useChat({
    transport,
    onFinish: () => setQueryCount((c) => c + 1),
    onError: (err) => {
      console.error("[DSPA Bot] Chat error:", err);
      if (
        err.message?.includes("Failed to fetch") ||
        err.message?.includes("NetworkError")
      ) {
        setBackendError(
          `Cannot reach the backend. Make sure the FastAPI server is running on ${API_URL}`,
        );
      } else if (
        err.message?.includes("429") ||
        err.message?.includes("too fast")
      ) {
        setBackendError(err.message);
      } else {
        setBackendError(`Something went wrong: ${err.message}`);
      }
    },
  });

  const isActive =
    chatStatus === "submitted" || chatStatus === "streaming";

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, chatStatus]);

  const handleSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      if (!input.trim() || limitReached || isActive) return;
      sendMessage({ text: input });
      setInput("");
    },
    [input, limitReached, isActive, sendMessage],
  );

  const handleSample = useCallback(
    (query: string) => {
      if (!limitReached && !isActive) sendMessage({ text: query });
    },
    [sendMessage, limitReached, isActive],
  );

  const handleFeedback = useCallback(
    (id: string, v: "up" | "down") =>
      setFeedback((prev) => ({ ...prev, [id]: v })),
    [],
  );

  const showLoadingSteps = chatStatus === "submitted";
  const [drawerOpen, setDrawerOpen] = useState(false);

  // ─── Auth Guards (after all hooks) ───
  if (authStatus === "loading") {
    return (
      <div className="flex h-screen items-center justify-center bg-berkeley-blue text-white">
        Loading Berkeley Session...
      </div>
    );
  }

  if (!isSessionReady) {
    return (
      <div className="min-h-screen bg-berkeley-blue text-white flex items-center justify-center px-4">
        <div className="max-w-md w-full text-center space-y-6">
          <div className="text-6xl mb-2">🐻</div>
          <h1 className="text-3xl font-extrabold tracking-tight">
            Michael-DSPA
          </h1>
          <p className="text-sm text-blue-100">
            AI peer advising for the UC Berkeley Data Science major. Sign in
            with your Berkeley Google account to get started.
          </p>
          <button
            type="button"
            onClick={() => signIn("google")}
            className="mt-4 inline-flex items-center justify-center w-full max-w-xs mx-auto rounded-full bg-california-gold text-berkeley-blue font-semibold text-base px-6 py-3 shadow-lg hover:bg-yellow-400 transition-colors min-h-[48px]"
          >
            Sign in with Berkeley Google Account
          </button>
          <p className="text-[11px] text-blue-200 mt-2">
            Access is restricted to @berkeley.edu addresses to protect student
            privacy and keep this tool focused on Berkeley&apos;s Data Science
            program. This tool is an independent project by Michael Florip for
            Learnplex. This is NOT an official UC Berkeley tool.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* ── Sidebar ── */}
      <aside className="hidden md:flex w-72 flex-col h-full bg-berkeley-blue text-white">
        <div className="p-6 border-b border-white/10">
          <h1 className="text-lg font-bold tracking-tight">🐻 Michael-DSPA</h1>
          <p className="text-xs text-blue-200 mt-1">
            Knowledge from a former DS Peer Advisor.
          </p>
        </div>

        <div className="p-6">
          <p className="text-xs font-semibold uppercase tracking-wider text-blue-300 mb-3">
            Usage Tracker
          </p>
          <div className="w-full bg-white/10 rounded-full h-2.5 overflow-hidden">
            <motion.div
              className="h-full rounded-full bg-california-gold"
              initial={{ width: 0 }}
              animate={{ width: `${(queryCount / MAX_QUERIES) * 100}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            />
          </div>
          <p className="text-sm text-blue-200 mt-2">
            Queries used:{" "}
            <span className="font-semibold text-white">
              {queryCount}/{MAX_QUERIES}
            </span>
          </p>
        </div>

        <div className="mt-auto p-6 border-t border-white/10">
          <p className="text-[10px] text-blue-300 leading-relaxed">
            This is an independent project by Michael Florip for Learnplex.
            <br />
            It is NOT an official UC Berkeley tool.
          </p>
        </div>

        <div className="mt-auto p-4 border-t border-california-gold/20">
          <button
            type="button"
            onClick={() => signOut({ callbackUrl: "/" })}
            className="flex items-center w-full gap-2 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-white/10 rounded-md"
          >
            <svg
              className="w-[18px] h-[18px] shrink-0"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
              />
            </svg>
            Sign Out
          </button>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Mobile header with hamburger + sign out */}
        <header className="md:hidden bg-berkeley-blue text-white px-4 py-3 flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            className="p-2 -ml-2 rounded-lg hover:bg-white/10 transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center"
            aria-label="Open menu"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <h1 className="text-sm font-bold flex-1 text-center">
            🐻 Michael-DSPA
          </h1>
          <div className="flex items-center gap-2">
            <span className="text-xs text-blue-200 min-w-[3.5rem] text-right">
              {queryCount}/{MAX_QUERIES}
            </span>
            <button
              type="button"
              onClick={() => signOut()}
              className="text-[10px] px-2 py-1 rounded-full border border-white/30 text-white/80 hover:bg-white/10"
            >
              Sign out
            </button>
          </div>
        </header>

        {/* Mobile drawer: Usage Tracker (collapsible) */}
        {drawerOpen && (
          <>
            <div
              className="md:hidden fixed inset-0 bg-black/40 z-40"
              aria-hidden
              onClick={() => setDrawerOpen(false)}
            />
            <aside className="md:hidden fixed top-0 left-0 bottom-0 w-72 max-w-[85vw] bg-berkeley-blue text-white z-50 flex flex-col shadow-xl">
              <div className="p-4 border-b border-white/10 flex items-center justify-between">
                <span className="font-bold">🐻 Michael-DSPA</span>
                <button
                  type="button"
                  onClick={() => setDrawerOpen(false)}
                  className="p-2 rounded-lg hover:bg-white/10 min-h-[44px] min-w-[44px] flex items-center justify-center"
                  aria-label="Close menu"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="p-6">
                <p className="text-xs font-semibold uppercase tracking-wider text-blue-300 mb-3">
                  Usage Tracker
                </p>
                <div className="w-full bg-white/10 rounded-full h-2.5 overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-california-gold"
                    initial={{ width: 0 }}
                    animate={{ width: `${(queryCount / MAX_QUERIES) * 100}%` }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                  />
                </div>
                <p className="text-sm text-blue-200 mt-2">
                  Queries used:{" "}
                  <span className="font-semibold text-white">
                    {queryCount}/{MAX_QUERIES}
                  </span>
                </p>
              </div>
              <div className="mt-auto p-6 border-t border-white/10">
                <p className="text-[10px] text-blue-300 leading-relaxed">
                  This is an independent project by Michael Florip for Learnplex.
                  <br />
                  It is NOT an official UC Berkeley tool.
                </p>
              </div>
            </aside>
          </>
        )}

        {/* Error banner */}
        {backendError && (
          <div className="mx-4 md:mx-8 mt-4 px-4 py-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700 flex items-center justify-between">
            <span>{backendError}</span>
            <button
              onClick={() => setBackendError(null)}
              className="ml-3 text-red-400 hover:text-red-600 font-bold"
            >
              ✕
            </button>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
          {messages.length === 0 && !isActive && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <div className="text-5xl mb-4">🐻</div>
                <h2 className="text-2xl font-bold text-berkeley-blue mb-2">
                  Michael-DSPA
                </h2>
                <p className="text-gray-500 mb-8 max-w-md">
                  Ask me about major requirements, course planning, domain
                  emphases, advising resources, and more!
                </p>
                <div className="flex flex-wrap gap-3 justify-center">
                  {SAMPLE_QUERIES.map(({ label, query }) => (
                    <motion.button
                      key={label}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => handleSample(query)}
                      disabled={limitReached}
                      className="px-4 py-2 bg-white border-2 border-berkeley-blue/20 text-berkeley-blue rounded-full text-sm font-medium hover:border-berkeley-blue hover:bg-berkeley-blue hover:text-white transition-all disabled:opacity-40"
                    >
                      {label}
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            </div>
          )}

          <AnimatePresence mode="popLayout">
            {messages.map((msg) => {
              const isUser = msg.role === "user";
              const textContent = getTextContent(msg);
              const sourceParts = !isUser ? getSourceParts(msg) : [];
              const isStreamingThis =
                status === "streaming" &&
                !isUser &&
                msg.id === messages[messages.length - 1]?.id;

              return (
                <motion.div
                  key={msg.id}
                  layout
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.25 }}
                  className={`mb-4 flex ${
                    isUser ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`relative w-full max-w-[85%] md:max-w-[70%] rounded-2xl px-5 py-3.5 shadow-sm ${
                      isUser
                        ? "bg-berkeley-blue text-white"
                        : "bg-white border border-gray-200 text-gray-800"
                    }`}
                  >
                    {isUser ? (
                      <p className="text-sm leading-relaxed">{textContent}</p>
                    ) : (
                      <div className="prose prose-sm max-w-none prose-headings:text-berkeley-blue prose-a:text-founders-rock">
                        <ReactMarkdown>{textContent}</ReactMarkdown>
                      </div>
                    )}

                    {/* Source chips */}
                    {!isUser && sourceParts.length > 0 && !isStreamingThis && (
                      <div className="mt-3 pt-3 border-t border-gray-100 flex flex-wrap gap-1.5">
                        {sourceParts.map((part: unknown, i: number) => (
                          <SourceChip key={i} part={part} />
                        ))}
                      </div>
                    )}

                    {/* Feedback */}
                    {!isUser && !isStreamingThis && (
                      <FeedbackButtons
                        messageId={msg.id}
                        feedback={feedback}
                        onFeedback={handleFeedback}
                      />
                    )}

                    {/* Quick Actions (touch-friendly: min 44x44px, 8px gap) */}
                    {!isUser && !isStreamingThis && (
                      <div className="mt-3 pt-3 border-t border-gray-100 flex flex-wrap gap-2">
                        {QUICK_ACTIONS.map(({ label, query }) => (
                          <button
                            key={label}
                            type="button"
                            onClick={() => {
                              if (!limitReached && !isActive)
                                sendMessage({ text: query });
                            }}
                            disabled={limitReached || isActive}
                            className="min-h-[44px] min-w-[44px] px-4 py-3 rounded-lg text-sm font-medium bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors disabled:opacity-40"
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>

          {/* Multi-step loading animation */}
          {showLoadingSteps && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-4 flex justify-start"
            >
              <LoadingSteps />
            </motion.div>
          )}

          <div ref={scrollRef} />
        </div>

        {/* Input area */}
        <div className="border-t bg-white px-4 py-4 md:px-8">
          {limitReached ? (
            <div className="text-center py-2">
              <p className="text-sm text-gray-500">
                You&apos;ve reached the limit for this session. As a nonprofit,
                Learnplex limits queries to keep this tool free for everyone.{" "}
                <button
                  onClick={() => window.location.reload()}
                  className="text-berkeley-blue font-medium hover:underline"
                >
                  Refresh to start over
                </button>
              </p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="flex gap-3 items-end">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about the Data Science major…"
                disabled={isActive}
                className="flex-1 rounded-xl border border-gray-300 px-4 py-3 text-base focus:outline-none focus:ring-2 focus:ring-berkeley-blue/40 focus:border-berkeley-blue disabled:bg-gray-50 transition-all min-w-0"
                style={{ fontSize: "16px" }}
              />
              <motion.button
                type="submit"
                disabled={isActive || !input.trim()}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="bg-berkeley-blue text-white px-6 py-3 rounded-xl text-sm font-medium hover:bg-blue-900 disabled:opacity-40 transition-colors"
              >
                {isActive ? "…" : "Send"}
              </motion.button>
            </form>
          )}
        </div>
      </main>
    </div>
  );
}
