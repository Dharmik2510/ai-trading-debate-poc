import { useState, useCallback, useRef } from 'react';
import ConfigPanel from './components/ConfigPanel';
import StockMetrics from './components/StockMetrics';
import DebateArena from './components/DebateArena';
import FinalVerdict from './components/FinalVerdict';
import { NewsSection, RedditSection } from './components/SentimentSections';

export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [stockData, setStockData] = useState(null);
  const [debateEvents, setDebateEvents] = useState([]);
  const [finalVerdict, setFinalVerdict] = useState(null);
  const [currentStatus, setCurrentStatus] = useState('');
  const [error, setError] = useState(null);
  const [debateComplete, setDebateComplete] = useState(false);
  const eventSourceRef = useRef(null);

  const startDebate = useCallback(async (config) => {
    // Reset state
    setIsLoading(true);
    setStockData(null);
    setDebateEvents([]);
    setFinalVerdict(null);
    setCurrentStatus('Initializing debate...');
    setError(null);
    setDebateComplete(false);

    // Close any existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.abort();
    }

    try {
      const controller = new AbortController();
      eventSourceRef.current = controller;

      const response = await fetch('/api/debate/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              handleEvent(event);
            } catch {
              // ignore parse errors
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message || 'An unexpected error occurred.');
        setCurrentStatus('');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleEvent = (event) => {
    switch (event.type) {
      case 'status':
        setCurrentStatus(event.message || '');
        break;
      case 'stock_data':
        setStockData(event.data);
        setCurrentStatus('Stock data loaded. Starting debate...');
        break;
      case 'research':
        setDebateEvents(prev => [...prev, { type: 'research', content: event.content }]);
        setCurrentStatus('Research complete. Starting debate rounds...');
        break;
      case 'bull':
        setDebateEvents(prev => [...prev, { type: 'bull', content: event.content, round: event.round }]);
        setCurrentStatus(`Round ${event.round}: Bear analyst is formulating counter-argument...`);
        break;
      case 'bear':
        setDebateEvents(prev => [...prev, { type: 'bear', content: event.content, round: event.round }]);
        setCurrentStatus(`Round ${event.round} complete. Preparing next round...`);
        break;
      case 'verdict':
        setFinalVerdict(event.content);
        setCurrentStatus('Debate complete!');
        break;
      case 'complete':
        setCurrentStatus('');
        setDebateComplete(true);
        setIsLoading(false);
        break;
      case 'error':
        setError(event.message || 'An error occurred during the debate.');
        setCurrentStatus('');
        setIsLoading(false);
        break;
    }
  };

  const hasContent = stockData || debateEvents.length > 0 || finalVerdict;

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: '#0a0b0f' }}>
      {/* Sidebar */}
      <ConfigPanel onStartDebate={startDebate} isLoading={isLoading} />

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        {/* Top Header Bar */}
        <div
          className="sticky top-0 z-10 px-6 py-4 flex items-center justify-between"
          style={{
            background: 'rgba(10,11,15,0.85)',
            backdropFilter: 'blur(20px)',
            borderBottom: '1px solid rgba(255,255,255,0.06)',
          }}
        >
          <div>
            <h1 className="text-2xl font-bold gradient-text">🚀 AI Trading Debate Platform</h1>
            <p className="text-xs text-gray-500 mt-0.5">Watch AI agents debate whether a stock is good for day trading</p>
          </div>
          {debateComplete && (
            <div
              className="px-3 py-1.5 rounded-full text-xs font-semibold"
              style={{ background: 'rgba(0,230,118,0.15)', color: '#00E676', border: '1px solid rgba(0,230,118,0.3)' }}
            >
              ✨ Debate Complete
            </div>
          )}
        </div>

        {/* Content Area */}
        <div className="p-6 space-y-5 max-w-5xl mx-auto">
          {/* Error Banner */}
          {error && (
            <div
              className="px-5 py-4 rounded-2xl"
              style={{ background: 'rgba(255,23,68,0.12)', border: '1px solid rgba(255,23,68,0.3)' }}
            >
              <div className="flex items-center gap-2">
                <span className="text-red-400 font-bold">⚠️ Error</span>
                <span className="text-sm text-red-300">{error}</span>
              </div>
            </div>
          )}

          {/* Welcome Screen */}
          {!hasContent && !isLoading && !error && (
            <WelcomeScreen />
          )}

          {/* Stock Metrics */}
          {stockData && (
            <StockMetrics stockData={stockData} />
          )}

          {/* News & Reddit Sentiment — show side-by-side */}
          {stockData && (stockData.news_sentiment || stockData.reddit_sentiment) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              <NewsSection newsSentiment={stockData.news_sentiment} />
              <RedditSection redditSentiment={stockData.reddit_sentiment} />
            </div>
          )}

          {/* Debate Arena */}
          {(debateEvents.length > 0 || (isLoading && currentStatus)) && (
            <DebateArena
              events={debateEvents}
              currentStatus={currentStatus}
              isLoading={isLoading}
            />
          )}

          {/* Final Verdict */}
          {finalVerdict && (
            <FinalVerdict content={finalVerdict} />
          )}

          {/* Footer */}
          {hasContent && (
            <div className="text-center py-6">
              <p className="text-xs text-gray-600">
                AI Trading Debate Platform • Built by Dharmik Soni, 2025
              </p>
              <p className="text-xs text-red-700 mt-1 font-semibold">
                ⚠️ For educational purposes only. Not financial advice. Do your own research.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function WelcomeScreen() {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center space-y-6">
      {/* Big Icon */}
      <div
        className="w-24 h-24 rounded-full flex items-center justify-center text-5xl"
        style={{
          background: 'linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15))',
          border: '1px solid rgba(99,102,241,0.3)',
          boxShadow: '0 0 40px rgba(99,102,241,0.2)',
        }}
      >
        📈
      </div>

      <div>
        <h2 className="text-3xl font-bold gradient-text mb-2">Welcome to AI Trading Debate</h2>
        <p className="text-gray-400 text-sm max-w-md mx-auto leading-relaxed">
          Configure your settings in the sidebar and click <strong className="text-indigo-400">🎯 Start Debate</strong> to
          watch AI agents (Bull 🐂 vs Bear 🐻) debate whether a stock is good for day trading.
        </p>
      </div>

      {/* Feature Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6 w-full max-w-2xl">
        {[
          { icon: '📊', title: 'Technical Analysis', desc: 'RSI, MACD, Bollinger Bands, Support & Resistance' },
          { icon: '📰', title: 'News Sentiment', desc: 'Real-time Yahoo Finance news with AI sentiment analysis' },
          { icon: '🤝', title: 'Multi-Agent Debate', desc: 'CrewAI-powered bull vs bear debate with final verdict' },
        ].map((f, i) => (
          <div
            key={i}
            className="rounded-2xl p-4 text-left"
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.07)',
            }}
          >
            <div className="text-2xl mb-2">{f.icon}</div>
            <div className="text-sm font-semibold text-gray-200 mb-1">{f.title}</div>
            <div className="text-xs text-gray-500 leading-relaxed">{f.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
