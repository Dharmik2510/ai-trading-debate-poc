export default function StockMetrics({ stockData }) {
  if (!stockData) return null;

  const {
    symbol, price, change_pct, volume, rsi, macd,
    ma_20, ma_50, support, resistance, bb_upper, bb_lower, atr
  } = stockData;

  const isPositive = change_pct >= 0;

  const metrics = [
    {
      label: 'Price',
      value: `$${price?.toFixed(2)}`,
      sub: change_pct !== undefined ? `${change_pct >= 0 ? '+' : ''}${change_pct?.toFixed(2)}%` : null,
      subColor: isPositive ? '#00E676' : '#FF1744',
      icon: '💲',
    },
    {
      label: 'RSI (14)',
      value: rsi?.toFixed(1),
      sub: rsi > 70 ? 'Overbought' : rsi < 30 ? 'Oversold' : 'Neutral',
      subColor: rsi > 70 ? '#FF1744' : rsi < 30 ? '#00E676' : '#94a3b8',
      icon: '📊',
    },
    {
      label: 'MACD',
      value: macd?.toFixed(3),
      sub: macd > 0 ? 'Bullish' : 'Bearish',
      subColor: macd > 0 ? '#00E676' : '#FF1744',
      icon: '📉',
    },
    {
      label: 'Support',
      value: `$${support?.toFixed(2)}`,
      sub: 'Floor Level',
      subColor: '#94a3b8',
      icon: '🟢',
    },
    {
      label: 'Resistance',
      value: `$${resistance?.toFixed(2)}`,
      sub: 'Ceiling Level',
      subColor: '#94a3b8',
      icon: '🔴',
    },
    {
      label: 'MA 20',
      value: `$${ma_20?.toFixed(2)}`,
      sub: price > ma_20 ? 'Above MA' : 'Below MA',
      subColor: price > ma_20 ? '#00E676' : '#FF1744',
      icon: '📈',
    },
    {
      label: 'MA 50',
      value: `$${ma_50?.toFixed(2)}`,
      sub: price > ma_50 ? 'Above MA' : 'Below MA',
      subColor: price > ma_50 ? '#00E676' : '#FF1744',
      icon: '📈',
    },
    {
      label: 'ATR',
      value: atr?.toFixed(2),
      sub: 'Volatility',
      subColor: '#94a3b8',
      icon: '⚡',
    },
    {
      label: 'BB Upper',
      value: `$${bb_upper?.toFixed(2)}`,
      sub: price > bb_upper ? 'Above Band' : 'In Range',
      subColor: price > bb_upper ? '#FF1744' : '#94a3b8',
      icon: '📐',
    },
    {
      label: 'BB Lower',
      value: `$${bb_lower?.toFixed(2)}`,
      sub: price < bb_lower ? 'Below Band' : 'In Range',
      subColor: price < bb_lower ? '#00E676' : '#94a3b8',
      icon: '📐',
    },
  ];

  return (
    <div
      className="rounded-2xl p-5"
      style={{
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-bold gradient-text">📊 {symbol} Analysis</h2>
          <p className="text-xs text-gray-500 mt-0.5">Real-time technical indicators</p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-white">${price?.toFixed(2)}</div>
          <div
            className="text-sm font-semibold"
            style={{ color: isPositive ? '#00E676' : '#FF1744' }}
          >
            {isPositive ? '▲' : '▼'} {Math.abs(change_pct)?.toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Volume */}
      <div
        className="flex items-center gap-2 mb-4 px-3 py-2 rounded-lg"
        style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)' }}
      >
        <span className="text-gray-400 text-xs">📦 Volume:</span>
        <span className="text-gray-200 text-xs font-semibold">{volume?.toLocaleString()}</span>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
        {metrics.slice(0, 5).map((m, i) => (
          <MetricCard key={i} {...m} />
        ))}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 mt-3">
        {metrics.slice(5).map((m, i) => (
          <MetricCard key={i} {...m} />
        ))}
      </div>
    </div>
  );
}

function MetricCard({ label, value, sub, subColor, icon }) {
  return (
    <div
      className="rounded-xl p-3 transition-all duration-200 hover:scale-105"
      style={{
        background: 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.07)',
      }}
    >
      <div className="text-xs text-gray-500 mb-1">{icon} {label}</div>
      <div className="text-base font-bold text-gray-100 leading-tight">{value}</div>
      {sub && (
        <div className="text-xs font-medium mt-0.5" style={{ color: subColor }}>
          {sub}
        </div>
      )}
    </div>
  );
}
