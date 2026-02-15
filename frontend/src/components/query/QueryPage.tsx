import { useState, useRef, useEffect } from 'react';
import { Send, Key, AlertTriangle, Sparkles, User, Bot } from 'lucide-react';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Panel } from '../common/Panel';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export function QueryPage() {
  const { data, dataLoaded, geminiKey, setGeminiKey, setPage } = useStore();
  const [selectedCol, setSelectedCol] = useState('');
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chatEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const columns = data?.columns ?? [];

  const handleSend = async () => {
    if (!question.trim() && !selectedCol) return;
    if (!geminiKey) {
      setError('Enter your Gemini API key to use AI queries.');
      return;
    }
    if (!selectedCol) {
      setError('Select a column to query.');
      return;
    }

    const userMsg: Message = { role: 'user', content: question || 'Summarize this data', timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setQuestion('');
    setError(null);
    setLoading(true);

    try {
      const res = await api.queryGemini(question || '', selectedCol, geminiKey);
      const assistantMsg: Message = { role: 'assistant', content: res.response, timestamp: new Date() };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Query failed');
    } finally {
      setLoading(false);
    }
  };

  if (!dataLoaded) {
    return (
      <Panel title="No Data Loaded">
        <p className="text-sm text-text-muted">
          Load a dataset first from the{' '}
          <button onClick={() => setPage('dashboard')} className="text-accent hover:underline">
            Dashboard
          </button>.
        </p>
      </Panel>
    );
  }

  return (
    <div className="flex h-full gap-4">
      {/* Sidebar config */}
      <div className="w-72 shrink-0 space-y-4">
        <Panel title="Configuration">
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-xs text-text-muted">Gemini API Key</label>
              <div className="relative">
                <Key className="absolute left-2.5 top-2 h-3.5 w-3.5 text-text-muted" />
                <input
                  type="password"
                  value={geminiKey}
                  onChange={(e) => setGeminiKey(e.target.value)}
                  placeholder="Enter API key..."
                  className="w-full rounded border border-border bg-deep py-1.5 pl-8 pr-3 text-xs text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
                />
              </div>
            </div>
            <div>
              <label className="mb-1 block text-xs text-text-muted">Target Column</label>
              <select
                value={selectedCol}
                onChange={(e) => setSelectedCol(e.target.value)}
                className="w-full rounded border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
              >
                <option value="">Select a column...</option>
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </Panel>

        <Panel title="Quick Prompts">
          <div className="space-y-1.5">
            {[
              'Summarize the data in bullet points',
              'What are the most common patterns?',
              'Identify any anomalies or outliers',
              'What correlations exist in this data?',
              'Provide a statistical overview',
            ].map((prompt) => (
              <button
                key={prompt}
                onClick={() => setQuestion(prompt)}
                className="w-full rounded border border-border/50 bg-raised px-3 py-2 text-left text-xs text-text-secondary transition-colors hover:border-accent hover:bg-elevated"
              >
                {prompt}
              </button>
            ))}
          </div>
        </Panel>
      </div>

      {/* Chat area */}
      <div className="flex flex-1 flex-col rounded-lg border border-border bg-surface">
        {/* Header */}
        <div className="flex items-center gap-2 border-b border-border px-4 py-3">
          <Sparkles className="h-4 w-4 text-accent" />
          <span className="text-sm font-semibold text-text-primary">AI Intelligence Query</span>
          <span className="text-xs text-text-muted">Powered by Gemini</span>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
              <div className="rounded-full bg-elevated p-4">
                <Sparkles className="h-8 w-8 text-accent/50" />
              </div>
              <p className="text-sm text-text-muted">
                Ask questions about your data using natural language.
              </p>
              <p className="text-xs text-text-muted">
                Select a column and type your question below.
              </p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`mb-4 flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="mt-1 shrink-0 rounded-full bg-accent-dim/30 p-1.5">
                  <Bot className="h-3.5 w-3.5 text-accent" />
                </div>
              )}
              <div
                className={`max-w-[75%] rounded-lg px-4 py-2.5 text-sm ${
                  msg.role === 'user'
                    ? 'bg-accent-dim text-white'
                    : 'border border-border bg-raised text-text-primary'
                }`}
              >
                <div className="whitespace-pre-wrap">{msg.content}</div>
                <p className="mt-1 text-[10px] opacity-50">
                  {msg.timestamp.toLocaleTimeString()}
                </p>
              </div>
              {msg.role === 'user' && (
                <div className="mt-1 shrink-0 rounded-full bg-purple/20 p-1.5">
                  <User className="h-3.5 w-3.5 text-purple" />
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <div className="flex gap-1">
                <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-accent [animation-delay:-0.3s]" />
                <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-accent [animation-delay:-0.15s]" />
                <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-accent" />
              </div>
              Analyzing data...
            </div>
          )}
          <div ref={chatEnd} />
        </div>

        {/* Error */}
        {error && (
          <div className="mx-4 mb-2 flex items-center gap-2 rounded-md bg-danger/10 px-3 py-2 text-xs text-danger">
            <AlertTriangle className="h-3.5 w-3.5" /> {error}
          </div>
        )}

        {/* Input */}
        <div className="border-t border-border p-3">
          <div className="flex items-center gap-2">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="Ask about your data..."
              className="flex-1 rounded-md border border-border bg-deep px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
            />
            <button
              onClick={handleSend}
              disabled={loading || !geminiKey}
              className="rounded-md bg-accent-dim p-2 text-white transition-colors hover:bg-accent disabled:opacity-50"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
