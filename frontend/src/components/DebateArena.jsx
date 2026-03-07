import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function AgentMessage({ type, content, round }) {
  const [expanded, setExpanded] = useState(true);

  const isBull = type === 'bull';
  const isBear = type === 'bear';
  const isResearch = type === 'research';

  let cardClass = 'research-card';
  let avatar = '🕵️';
  let name = 'Market Researcher';
  let headerColor = '#6366f1';
  let roundLabel = 'Research Phase';

  if (isBull) {
    cardClass = 'bull-card';
    avatar = '🐂';
    name = 'Agent Bull';
    headerColor = '#00E676';
    roundLabel = `Round ${round} — Bullish Analysis`;
  } else if (isBear) {
    cardClass = 'bear-card';
    avatar = '🐻';
    name = 'Agent Bear';
    headerColor = '#FF1744';
    roundLabel = `Round ${round} — Bearish Counter-Analysis`;
  }

  return (
    <div className={`rounded-2xl overflow-hidden ${cardClass}`}>
      {/* Header toggle */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-5 py-3 text-left transition-colors"
        style={{ background: 'rgba(0,0,0,0.15)' }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-full flex items-center justify-center text-lg avatar-pulse"
            style={{ background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.15)' }}
          >
            {avatar}
          </div>
          <div>
            <div className="text-sm font-bold" style={{ color: headerColor }}>{name}</div>
            <div className="text-xs text-gray-400">{roundLabel}</div>
          </div>
        </div>
        <span className="text-gray-500 text-xs">{expanded ? '▲ Collapse' : '▼ Expand'}</span>
      </button>

      {/* Content */}
      {expanded && (
        <div className="px-5 py-4">
          <div className="markdown-content text-sm text-gray-300 leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {content}
            </ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}

function TypingIndicator({ label }) {
  return (
    <div
      className="rounded-2xl px-5 py-4 flex items-center gap-3"
      style={{
        background: 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.08)',
      }}
    >
      <div className="flex gap-1 items-end">
        <span className="w-2 h-2 rounded-full bg-indigo-400 typing-dot" />
        <span className="w-2 h-2 rounded-full bg-purple-400 typing-dot" />
        <span className="w-2 h-2 rounded-full bg-indigo-400 typing-dot" />
      </div>
      <span className="text-sm text-gray-400">{label}</span>
    </div>
  );
}

export default function DebateArena({ events, currentStatus, isLoading }) {
  if (events.length === 0 && !isLoading) return null;

  return (
    <div
      className="rounded-2xl p-5 space-y-4"
      style={{
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.07)',
      }}
    >
      {/* Section Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold gradient-text">🤖 Live Agentic Debate</h2>
          <p className="text-xs text-gray-500 mt-0.5">Powered by CrewAI + OpenAI</p>
        </div>
        {isLoading && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full" style={{ background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.3)' }}>
            <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
            <span className="text-xs text-indigo-300 font-medium">Live</span>
          </div>
        )}
      </div>

      {/* Status Message */}
      {currentStatus && (
        <div
          className="px-4 py-2.5 rounded-xl flex items-center gap-2"
          style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)' }}
        >
          <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
          <span className="text-xs text-indigo-200">{currentStatus}</span>
        </div>
      )}

      {/* Debate Messages */}
      <div className="space-y-3">
        {events.map((event, idx) => {
          if (event.type === 'research') {
            return (
              <AgentMessage key={idx} type="research" content={event.content} />
            );
          }
          if (event.type === 'bull') {
            return (
              <AgentMessage key={idx} type="bull" content={event.content} round={event.round} />
            );
          }
          if (event.type === 'bear') {
            return (
              <AgentMessage key={idx} type="bear" content={event.content} round={event.round} />
            );
          }
          return null;
        })}

        {/* Typing indicator for active loading */}
        {isLoading && currentStatus && (
          <TypingIndicator label={currentStatus} />
        )}
      </div>
    </div>
  );
}
