import { useState } from 'react';

function SentimentBadge({ sentiment }) {
  const s = (sentiment || '').toLowerCase();
  if (s === 'bullish' || s === 'positive') {
    return (
      <span className="px-2 py-0.5 rounded-full text-xs font-semibold" style={{ background: 'rgba(0,230,118,0.15)', color: '#00E676', border: '1px solid rgba(0,230,118,0.3)' }}>
        🟢 Bullish
      </span>
    );
  }
  if (s === 'bearish' || s === 'negative') {
    return (
      <span className="px-2 py-0.5 rounded-full text-xs font-semibold" style={{ background: 'rgba(255,23,68,0.15)', color: '#FF1744', border: '1px solid rgba(255,23,68,0.3)' }}>
        🔴 Bearish
      </span>
    );
  }
  return (
    <span className="px-2 py-0.5 rounded-full text-xs font-semibold" style={{ background: 'rgba(148,163,184,0.15)', color: '#94a3b8', border: '1px solid rgba(148,163,184,0.25)' }}>
      ⚪ Neutral
    </span>
  );
}

export function NewsSection({ newsSentiment }) {
  const [expanded, setExpanded] = useState(true);

  if (!newsSentiment) {
    return (
      <div className="rounded-2xl p-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <p className="text-sm text-gray-500">No news sentiment data available.</p>
      </div>
    );
  }

  const headlines = newsSentiment.headlines || [];
  const summary = newsSentiment.summary || '';

  return (
    <div className="rounded-2xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-5 py-3.5 text-left transition-colors hover:bg-white/5"
      >
        <div>
          <h3 className="text-base font-bold gradient-text">📰 Recent News Sentiment</h3>
          {summary && <p className="text-xs text-gray-500 mt-0.5 truncate max-w-xs">{summary.slice(0, 80)}...</p>}
        </div>
        <span className="text-gray-500 text-xs ml-3 shrink-0">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <div className="px-5 pb-5 space-y-3">
          {/* Summary */}
          {summary && (
            <div
              className="px-4 py-3 rounded-xl text-xs text-gray-300 leading-relaxed"
              style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}
            >
              {summary}
            </div>
          )}

          {/* Headlines */}
          {headlines.length > 0 && (
            <div className="space-y-2">
              {headlines.map((headline, i) => {
                const title = typeof headline === 'object' ? headline.title : headline;
                const publisher = typeof headline === 'object' ? headline.publisher : 'N/A';
                const sentiment = typeof headline === 'object' ? headline.sentiment : 'neutral';
                const link = typeof headline === 'object' ? headline.link : null;

                return (
                  <div
                    key={i}
                    className="px-4 py-3 rounded-xl flex items-start gap-3"
                    style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}
                  >
                    <div className="flex-1 min-w-0">
                      {link ? (
                        <a
                          href={link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-blue-300 hover:text-blue-200 font-medium line-clamp-2 transition-colors"
                        >
                          {title}
                        </a>
                      ) : (
                        <p className="text-xs text-gray-200 font-medium">{title}</p>
                      )}
                      <p className="text-xs text-gray-500 mt-0.5">{publisher}</p>
                    </div>
                    <SentimentBadge sentiment={sentiment} />
                  </div>
                );
              })}
            </div>
          )}

          {headlines.length === 0 && (
            <p className="text-sm text-gray-500">No headlines available.</p>
          )}
        </div>
      )}
    </div>
  );
}

export function RedditSection({ redditSentiment }) {
  const [expanded, setExpanded] = useState(true);
  const [expandedDiscussions, setExpandedDiscussions] = useState({});

  if (!redditSentiment || !redditSentiment.summary) {
    return (
      <div className="rounded-2xl p-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
        <p className="text-sm text-gray-500">No Reddit sentiment data available.</p>
      </div>
    );
  }

  const { summary, sentiment_breakdown, discussions = [] } = redditSentiment;

  const toggleDiscussion = (i) => {
    setExpandedDiscussions(prev => ({ ...prev, [i]: !prev[i] }));
  };

  return (
    <div className="rounded-2xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-5 py-3.5 text-left transition-colors hover:bg-white/5"
      >
        <div>
          <h3 className="text-base font-bold gradient-text">🤖 Reddit Community Sentiment</h3>
          {summary && <p className="text-xs text-gray-500 mt-0.5 truncate max-w-xs">{summary.slice(0, 80)}...</p>}
        </div>
        <span className="text-gray-500 text-xs ml-3 shrink-0">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <div className="px-5 pb-5 space-y-3">
          {/* Summary */}
          {summary && (
            <div
              className="px-4 py-3 rounded-xl text-xs text-gray-300 leading-relaxed"
              style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}
            >
              {summary}
            </div>
          )}

          {/* Sentiment Breakdown */}
          {sentiment_breakdown && (
            <div className="grid grid-cols-3 gap-2">
              <div className="rounded-xl p-3 text-center" style={{ background: 'rgba(0,230,118,0.08)', border: '1px solid rgba(0,230,118,0.2)' }}>
                <div className="text-xl font-bold text-green-400">{sentiment_breakdown.bullish ?? 0}</div>
                <div className="text-xs text-gray-400 mt-0.5">🟢 Bullish</div>
              </div>
              <div className="rounded-xl p-3 text-center" style={{ background: 'rgba(255,23,68,0.08)', border: '1px solid rgba(255,23,68,0.2)' }}>
                <div className="text-xl font-bold text-red-400">{sentiment_breakdown.bearish ?? 0}</div>
                <div className="text-xs text-gray-400 mt-0.5">🔴 Bearish</div>
              </div>
              <div className="rounded-xl p-3 text-center" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
                <div className="text-xl font-bold text-gray-300">{sentiment_breakdown.neutral ?? 0}</div>
                <div className="text-xs text-gray-400 mt-0.5">⚪ Neutral</div>
              </div>
            </div>
          )}

          {/* Discussions */}
          {discussions.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-400 mb-2">Top Reddit Discussions:</p>
              <div className="space-y-2">
                {discussions.slice(0, 5).map((d, i) => {
                  const isExpanded = expandedDiscussions[i];
                  return (
                    <div
                      key={i}
                      className="rounded-xl overflow-hidden"
                      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
                    >
                      <button
                        onClick={() => toggleDiscussion(i)}
                        className="w-full flex items-center gap-2 px-4 py-2.5 text-left hover:bg-white/5 transition-colors"
                      >
                        <SentimentBadge sentiment={d.sentiment} />
                        <span className="text-xs text-gray-300 flex-1 text-left line-clamp-1">
                          r/{d.subreddit} — {d.title}
                        </span>
                        <span className="text-xs text-gray-600 shrink-0">{isExpanded ? '▲' : '▼'}</span>
                      </button>
                      {isExpanded && (
                        <div className="px-4 pb-3 space-y-1.5">
                          <div className="text-xs text-gray-400">
                            <span className="font-medium text-gray-300">Score:</span> {d.score} upvotes
                          </div>
                          <div className="text-xs text-gray-400 leading-relaxed">
                            <span className="font-medium text-gray-300">Preview:</span> {d.content_preview}
                          </div>
                          {d.url && (
                            <a
                              href={d.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                            >
                              View on Reddit →
                            </a>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
