import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function FinalVerdict({ content }) {
  const [copied, setCopied] = useState(false);

  if (!content) return null;

  const handleCopy = () => {
    const clean = content.replace(/[*_#`]/g, '').replace(/\s+/g, ' ').trim();
    navigator.clipboard.writeText(clean).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    });
  };

  // Determine decision badge color
  const upperContent = content.toUpperCase();
  let decision = null;
  let decisionColor = '#FFD700';
  let decisionBg = 'rgba(255,215,0,0.15)';

  if (upperContent.includes('**DECISION**: BUY') || upperContent.includes('**DECISION**: BUY') || upperContent.match(/decision.*?buy/i)) {
    decision = 'BUY';
    decisionColor = '#00E676';
    decisionBg = 'rgba(0,230,118,0.15)';
  } else if (upperContent.match(/decision.*?sell/i) || upperContent.match(/\*\*decision\*\*:.*sell/i)) {
    decision = 'SELL';
    decisionColor = '#FF1744';
    decisionBg = 'rgba(255,23,68,0.15)';
  } else if (upperContent.match(/decision.*?hold/i) || upperContent.match(/\*\*decision\*\*:.*hold/i)) {
    decision = 'HOLD';
    decisionColor = '#FFD700';
    decisionBg = 'rgba(255,215,0,0.15)';
  }

  return (
    <div className="verdict-card rounded-2xl p-6 space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 relative z-10">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-2xl">⚖️</span>
            <h2 className="text-xl font-bold gradient-text-gold">Chief Risk Officer Verdict</h2>
          </div>
          <p className="text-xs text-yellow-600">Final AI Trading Recommendation</p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {decision && (
            <span
              className="px-4 py-1.5 rounded-full text-sm font-bold tracking-wider"
              style={{
                background: decisionBg,
                color: decisionColor,
                border: `1px solid ${decisionColor}50`,
              }}
            >
              {decision}
            </span>
          )}
          <button
            onClick={handleCopy}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all"
            style={{
              background: copied ? 'rgba(0,230,118,0.2)' : 'rgba(255,255,255,0.08)',
              border: copied ? '1px solid rgba(0,230,118,0.4)' : '1px solid rgba(255,255,255,0.12)',
              color: copied ? '#00E676' : '#94a3b8',
            }}
          >
            {copied ? '✅ Copied!' : '📋 Copy'}
          </button>
        </div>
      </div>

      {/* Divider */}
      <div style={{ height: '1px', background: 'rgba(255,215,0,0.2)' }} className="relative z-10" />

      {/* Content */}
      <div className="markdown-content text-sm text-gray-200 leading-relaxed relative z-10">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {content}
        </ReactMarkdown>
      </div>

      {/* Disclaimer */}
      <div
        className="px-4 py-2.5 rounded-xl text-xs text-center relative z-10"
        style={{
          background: 'rgba(255,82,82,0.1)',
          border: '1px solid rgba(255,82,82,0.2)',
          color: '#ff7070',
        }}
      >
        ⚠️ This is AI-generated analysis for educational purposes only. Not financial advice. Always do your own research.
      </div>
    </div>
  );
}
