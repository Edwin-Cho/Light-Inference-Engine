import { useState, useRef, useEffect, useCallback } from 'react';
import { api } from '@/lib/api';
import type { Citation, ChatMessage } from '@/lib/types';
import { useSession } from '@/contexts/SessionContext';
import { readSessions } from '@/lib/sessions';
import { Send, Loader2, BookOpen, AlertTriangle, ExternalLink, Copy, Check, Trash2, ChevronDown, ChevronUp, Sparkles, Search, ServerCrash } from 'lucide-react';
import MathRenderer from '@/components/MathRenderer';

type Message = ChatMessage;

const FALLBACK_QUERIES = [
  'What is the attention mechanism?',
  'How does BERT pre-training work?',
  'Compare encoder-only vs decoder-only models',
  'What is the computational complexity of self-attention?',
];

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.8 ? '#10b981' : score >= 0.65 ? '#059669' : '#4b5563';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1 rounded-full bg-white/5 overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-[10px] tabular-nums" style={{ color }}>{pct}%</span>
    </div>
  );
}

function CitationCard({ c }: { c: Citation }) {
  return (
    <div className="animate-enter-view rounded-xl p-3 text-xs space-y-2 bg-slate-900 border border-slate-800">
      <div className="flex items-start justify-between gap-2">
        <span className="font-medium text-slate-100 leading-snug">{c.paper_title || c.source_filename}</span>
        {c.arxiv_id && (
          <a href={`https://arxiv.org/abs/${c.arxiv_id}`} target="_blank" rel="noopener noreferrer"
            className="flex items-center gap-1 text-emerald-400/70 hover:text-emerald-300 transition shrink-0">
            arXiv <ExternalLink className="w-2.5 h-2.5" />
          </a>
        )}
      </div>
      {c.section_header && (
        <div className="text-slate-500 flex items-center gap-1">
          <span className="text-slate-600">§</span> {c.section_header}
        </div>
      )}
      <div className="flex items-center gap-3 text-slate-600">
        <span className="truncate">{c.source_filename}</span>
        {c.page_number && <span className="shrink-0">p.{c.page_number}</span>}
      </div>
      <ScoreBar score={c.score} />
    </div>
  );
}

function AssistantMessage({ msg, onCopy, copied, expanded, onToggleExpand }: {
  msg: Message;
  onCopy: (id: string, text: string) => void;
  copied: boolean;
  expanded: boolean;
  onToggleExpand: (id: string) => void;
}) {
  /* ── Backend error (network / server crash) ── */
  if (msg.error) {
    return (
      <div className="flex gap-3">
        <div className="w-8 h-8 bg-red-900/30 border border-red-700/30 rounded-lg flex items-center justify-center shrink-0 mt-0.5">
          <ServerCrash className="w-4 h-4 text-red-400" />
        </div>
        <div className="bg-red-500/8 border border-red-500/20 rounded-xl px-4 py-3 space-y-0.5">
          <p className="text-red-400 text-[15px] font-medium">Backend error</p>
          <p className="text-red-400/60 text-sm">Failed to connect or process the request. Check the Health badge — the server may be down.</p>
        </div>
      </div>
    );
  }

  /* ── No relevant documents found ── */
  if (msg.status === 'no_context') {
    return (
      <div className="flex gap-3">
        <div className="w-8 h-8 bg-slate-800/60 border border-slate-700/30 rounded-lg flex items-center justify-center shrink-0 mt-0.5">
          <Search className="w-4 h-4 text-slate-500" />
        </div>
        <div className="bg-slate-800/40 border border-slate-700/30 rounded-xl px-4 py-3 space-y-1">
          <p className="text-slate-300 text-[15px] font-medium">No relevant documents found</p>
          <p className="text-slate-500 text-sm leading-relaxed">Your indexed papers don't contain information matching this query. Try rephrasing or using different keywords.</p>
        </div>
      </div>
    );
  }

  /* ── Normal / partial answer ── */
  return (
    <div className="flex gap-3">
      <div className="w-8 h-8 bg-emerald-900/50 border border-emerald-700/30 rounded-lg flex items-center justify-center shrink-0 mt-0.5">
        <BookOpen className="w-4 h-4 text-emerald-400" />
      </div>
      <div className="flex-1 space-y-3 min-w-0">
        {msg.warning && (
          <div className="flex items-start gap-2.5 bg-amber-500/8 border border-amber-500/20 rounded-xl px-3.5 py-2.5">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
            <div className="min-w-0">
              <p className="text-amber-400 text-xs font-medium leading-tight">Partial results</p>
              <p className="text-amber-400/60 text-xs mt-0.5 leading-snug">{msg.warning}</p>
            </div>
          </div>
        )}
        <div className="relative group">
          <div className="bg-slate-800/60 border border-slate-700/40 rounded-xl px-4 py-3 text-slate-200 text-base leading-relaxed whitespace-pre-wrap">
            <MathRenderer text={msg.content} />
          </div>
          <button
            onClick={() => onCopy(msg.id, msg.content)}
            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-lg bg-slate-700/60 hover:bg-slate-600/60 text-slate-400 hover:text-slate-200"
          >
            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
          </button>
        </div>
        {msg.citations && msg.citations.length > 0 && (
          <div>
            <button
              onClick={() => onToggleExpand(msg.id)}
              className={`flex items-center gap-1.5 text-xs transition mb-2 ${
                msg.citations.length === 1
                  ? 'text-amber-500/70 hover:text-amber-400'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              {msg.citations.length === 1
                ? <><AlertTriangle className="w-3 h-3" /> 1 source — verify independently</>
                : `Sources (${msg.citations.length})`
              }
            </button>
            {expanded && (
              <div className="grid gap-2">
                {msg.citations.map((c, i) => <CitationCard key={i} c={c} />)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function QueryPage() {
  const { currentId, persistMessages } = useSession();

  const [messages, setMessages] = useState<Message[]>(() => {
    if (!currentId) return [];
    const session = readSessions().find((s) => s.id === currentId);
    return (session?.messages as Message[]) ?? [];
  });
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(5);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedCitations, setExpandedCitations] = useState<Record<string, boolean>>({});
  const [suggestedQueries, setSuggestedQueries] = useState<string[]>([]);
  const [suggestLoading, setSuggestLoading] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const MAX_TEXTAREA_HEIGHT = 140; /* ~5 lines */

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  /* Auto-focus on mount */
  useEffect(() => { textareaRef.current?.focus(); }, []);

  /* Auto-resize textarea */
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
  }, [input]);

  const loadSuggestedQueries = useCallback(async (refresh = false) => {
    setSuggestLoading(true);
    try {
      const res = await api.suggestQueries(refresh);
      setSuggestedQueries(res.questions.length >= 2 ? res.questions : FALLBACK_QUERIES);
    } catch {
      setSuggestedQueries(FALLBACK_QUERIES);
    } finally {
      setSuggestLoading(false);
    }
  }, []);

  useEffect(() => { void loadSuggestedQueries(); }, [loadSuggestedQueries]);

  const handleCopy = (id: string, text: string) => {
    void navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  const handleToggleExpand = (id: string) => {
    setExpandedCitations((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const q = input.trim();
    if (!q || loading) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: q };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await api.query(q, topK);
      const msgId = (Date.now() + 1).toString();
      const assistantMsg: Message = {
        id: msgId,
        role: 'assistant',
        content: res.answer,
        citations: res.citations,
        status: res.status,
        warning: res.warnings?.length ? res.warnings.join(' | ') : undefined,
      };
      setMessages((prev) => {
        const next = [...prev, assistantMsg];
        persistMessages(next);
        return next;
      });
      setExpandedCitations((prev) => ({ ...prev, [msgId]: false }));
    } catch {
      setMessages((prev) => {
        const next = [
          ...prev,
          { id: (Date.now() + 1).toString(), role: 'assistant' as const, content: '', error: true },
        ];
        persistMessages(next);
        return next;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setInput('');
      return;
    }
    const cmdEnter = (e.metaKey || e.ctrlKey) && e.key === 'Enter';
    const enterOnly = e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey;
    if (cmdEnter || enterOnly) {
      e.preventDefault();
      void handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60">
        <div>
          <h2 className="text-white font-semibold">Research Query</h2>
          <p className="text-slate-500 text-xs mt-0.5">Ask questions about your indexed papers</p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={() => { setMessages([]); setExpandedCitations({}); }}
            className="flex items-center gap-1.5 text-xs text-slate-600 hover:text-red-400 px-2.5 py-1.5 rounded-lg hover:bg-red-500/8 transition"
          >
            <Trash2 className="w-3.5 h-3.5" /> Clear
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center py-16">
            <div className="w-14 h-14 bg-emerald-900/40 border border-emerald-700/40 rounded-2xl flex items-center justify-center mb-4" style={{ boxShadow: '0 0 24px rgba(16,185,129,0.12)' }}>
              <BookOpen className="w-7 h-7 text-emerald-400" />
            </div>
            <h3 className="text-white font-medium">Ask anything about your papers</h3>
            <p className="text-slate-500 text-base mt-1.5 max-w-sm">Queries are answered using only your indexed documents with forced citations.</p>
            <div className="flex flex-wrap gap-2 justify-center mt-6 max-w-md">
              {suggestLoading ? (
                Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="h-7 w-44 rounded-lg bg-slate-800/60 animate-pulse" />
                ))
              ) : (
                suggestedQueries.map((q) => (
                  <button
                    key={q}
                    onClick={() => { setInput(q); textareaRef.current?.focus(); }}
                    className="suggest-chip px-3 py-1.5 text-sm rounded-lg border border-slate-700/60 text-slate-400 hover:text-slate-200 hover:border-slate-600 hover:bg-slate-800/60 transition text-left"
                  >
                    {q}
                  </button>
                ))
              )}
            </div>
            {!suggestLoading && (
              <button
                onClick={() => void loadSuggestedQueries(true)}
                className="mt-3 flex items-center gap-1.5 text-xs text-slate-600 hover:text-emerald-400 transition"
              >
                <Sparkles className="w-3 h-3" /> Regenerate suggestions
              </button>
            )}
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className="animate-appear">
            {msg.role === 'user' ? (
              <div className="flex justify-end">
                <div className="bg-slate-700 border border-slate-600/40 rounded-xl px-4 py-2.5 text-white text-[15px] max-w-[70%] whitespace-pre-wrap">
                  {msg.content}
                </div>
              </div>
            ) : (
              <AssistantMessage
                msg={msg}
                onCopy={handleCopy}
                copied={copiedId === msg.id}
                expanded={expandedCitations[msg.id] ?? false}
                onToggleExpand={handleToggleExpand}
              />
            )}
          </div>
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 bg-emerald-900/50 border border-emerald-700/30 rounded-lg flex items-center justify-center shrink-0">
              <BookOpen className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="bg-slate-800/60 border border-slate-700/40 rounded-xl px-4 py-3 flex items-center gap-2 text-slate-400 text-[15px]">
              <Loader2 className="w-4 h-4 animate-spin" /> Retrieving and generating…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="px-6 py-4 border-t border-slate-800/60">
        <form onSubmit={handleSubmit} className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question… (Enter to send, Shift+Enter for newline, Esc to clear)"
            rows={1}
            style={{ maxHeight: MAX_TEXTAREA_HEIGHT, overflowY: 'auto' }}
            className="flex-1 bg-slate-900/80 border border-slate-700/60 rounded-xl px-4 py-3 text-white placeholder-slate-600 text-[15px] focus:outline-none focus:ring-1 focus:ring-emerald-500/60 focus:border-emerald-500/40 resize-none transition"
          />
          <div className="flex flex-col gap-2 shrink-0">
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="bg-slate-800/80 border border-slate-700/60 rounded-lg px-2 py-1.5 text-slate-400 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500/60 cursor-pointer"
            >
              {[3, 5, 10, 15].map((v) => <option key={v} value={v}>K={v}</option>)}
            </select>
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="h-9 px-3 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg flex items-center justify-center transition"
            >
              <Send className="w-4 h-4 text-white" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
