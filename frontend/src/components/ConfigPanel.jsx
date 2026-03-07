import { useState } from 'react';

const DEFAULT_BULL_PERSONALITY = {
  risk_tolerance: 'Medium-High',
  focus_areas: 'Breakouts, Momentum, Volume Spikes, Positive News Catalysts, Support Levels',
  style: 'Opportunistic growth seeker, quick to capitalize on upward trends.',
  beliefs: 'The market generally trends higher, and pullbacks are buying opportunities.',
};

const DEFAULT_BEAR_PERSONALITY = {
  risk_tolerance: 'Low-Medium',
  focus_areas: 'Risk Factors, Overbought Signals (RSI > 70), Resistance Levels, Negative News, Volume Declines',
  style: 'Defensive risk manager, identifies potential reversals and shorting opportunities.',
  beliefs: 'Markets are prone to corrections, and caution is paramount to preserve capital.',
};

const RISK_LEVELS = ['Low', 'Medium', 'Medium-High', 'High'];
const CHART_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '5y'];
const CHART_PERIOD_LABELS = {
  '1d': '1 Day', '5d': '5 Days', '1mo': '1 Month',
  '3mo': '3 Months', '6mo': '6 Months', '1y': '1 Year', '5y': '5 Years'
};

function CollapsibleSection({ title, emoji, children }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl overflow-hidden" style={{ border: '1px solid rgba(255,255,255,0.08)' }}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 text-left transition-colors hover:bg-white/5"
        style={{ background: 'rgba(255,255,255,0.03)' }}
      >
        <span className="text-sm font-semibold text-gray-200">{emoji} {title}</span>
        <span className="text-gray-400 text-xs">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="p-4 space-y-3" style={{ background: 'rgba(0,0,0,0.2)' }}>
          {children}
        </div>
      )}
    </div>
  );
}

function InputField({ label, type = 'text', value, onChange, placeholder, maxLength }) {
  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1 font-medium">{label}</label>
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        maxLength={maxLength}
        className="w-full px-3 py-2 rounded-lg text-sm text-gray-100 placeholder-gray-500 input-glow transition-all"
        style={{
          background: 'rgba(255,255,255,0.06)',
          border: '1px solid rgba(255,255,255,0.1)',
        }}
      />
    </div>
  );
}

function TextAreaField({ label, value, onChange, rows = 2 }) {
  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1 font-medium">{label}</label>
      <textarea
        value={value}
        onChange={e => onChange(e.target.value)}
        rows={rows}
        className="w-full px-3 py-2 rounded-lg text-sm text-gray-100 placeholder-gray-500 input-glow transition-all resize-none"
        style={{
          background: 'rgba(255,255,255,0.06)',
          border: '1px solid rgba(255,255,255,0.1)',
        }}
      />
    </div>
  );
}

function SelectField({ label, value, onChange, options }) {
  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1 font-medium">{label}</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full px-3 py-2 rounded-lg text-sm text-gray-100 input-glow transition-all"
        style={{
          background: 'rgba(20,20,30,0.9)',
          border: '1px solid rgba(255,255,255,0.1)',
        }}
      >
        {options.map(opt => (
          <option key={typeof opt === 'string' ? opt : opt.value} value={typeof opt === 'string' ? opt : opt.value}>
            {typeof opt === 'string' ? opt : opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export default function ConfigPanel({ onStartDebate, isLoading }) {
  const [openaiKey, setOpenaiKey] = useState('');
  const [redditClientId, setRedditClientId] = useState('');
  const [redditClientSecret, setRedditClientSecret] = useState('');
  const [symbol, setSymbol] = useState('AAPL');
  const [maxRounds, setMaxRounds] = useState(3);
  const [chartPeriod, setChartPeriod] = useState('1mo');

  const [bullRisk, setBullRisk] = useState(DEFAULT_BULL_PERSONALITY.risk_tolerance);
  const [bullFocus, setBullFocus] = useState(DEFAULT_BULL_PERSONALITY.focus_areas);
  const [bullStyle, setBullStyle] = useState(DEFAULT_BULL_PERSONALITY.style);
  const [bullBeliefs, setBullBeliefs] = useState(DEFAULT_BULL_PERSONALITY.beliefs);

  const [bearRisk, setBearRisk] = useState(DEFAULT_BEAR_PERSONALITY.risk_tolerance);
  const [bearFocus, setBearFocus] = useState(DEFAULT_BEAR_PERSONALITY.focus_areas);
  const [bearStyle, setBearStyle] = useState(DEFAULT_BEAR_PERSONALITY.style);
  const [bearBeliefs, setBearBeliefs] = useState(DEFAULT_BEAR_PERSONALITY.beliefs);

  const handleSubmit = () => {
    if (!openaiKey.trim()) {
      alert('Please enter your OpenAI API key.');
      return;
    }
    onStartDebate({
      openai_key: openaiKey,
      reddit_client_id: redditClientId,
      reddit_client_secret: redditClientSecret,
      symbol: symbol.toUpperCase(),
      max_rounds: maxRounds,
      chart_period: chartPeriod,
      bull_risk_tolerance: bullRisk,
      bull_focus_areas: bullFocus.split(',').map(s => s.trim()).filter(Boolean),
      bull_style: bullStyle,
      bull_beliefs: bullBeliefs,
      bear_risk_tolerance: bearRisk,
      bear_focus_areas: bearFocus.split(',').map(s => s.trim()).filter(Boolean),
      bear_style: bearStyle,
      bear_beliefs: bearBeliefs,
    });
  };

  return (
    <aside
      className="h-full overflow-y-auto flex flex-col gap-4 p-4"
      style={{
        background: 'rgba(255,255,255,0.02)',
        borderRight: '1px solid rgba(255,255,255,0.07)',
        width: '320px',
        minWidth: '280px',
      }}
    >
      {/* Header */}
      <div className="pt-2">
        <h2 className="text-lg font-bold gradient-text">⚙️ Configuration</h2>
        <p className="text-xs text-gray-500 mt-0.5">Configure your debate settings</p>
      </div>

      {/* Divider */}
      <div style={{ height: '1px', background: 'rgba(255,255,255,0.06)' }} />

      {/* OpenAI Key */}
      <div className="space-y-2">
        <InputField
          label="🔑 OpenAI API Key"
          type="password"
          value={openaiKey}
          onChange={setOpenaiKey}
          placeholder="sk-..."
        />
        {openaiKey ? (
          <p className="text-xs text-green-400">✅ API key configured</p>
        ) : (
          <p className="text-xs text-yellow-500">⚠️ Required to start debate</p>
        )}
      </div>

      <div style={{ height: '1px', background: 'rgba(255,255,255,0.06)' }} />

      {/* Reddit (optional) */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-gray-300">🤖 Reddit Analysis <span className="text-gray-500">(Optional)</span></p>
        <InputField
          label="Reddit Client ID"
          type="password"
          value={redditClientId}
          onChange={setRedditClientId}
          placeholder="Leave blank to skip"
        />
        <InputField
          label="Reddit Client Secret"
          type="password"
          value={redditClientSecret}
          onChange={setRedditClientSecret}
          placeholder="Leave blank to skip"
        />
        {redditClientId && redditClientSecret ? (
          <p className="text-xs text-green-400">✅ Reddit configured</p>
        ) : (
          <p className="text-xs text-gray-500">Reddit analysis will be skipped</p>
        )}
      </div>

      <div style={{ height: '1px', background: 'rgba(255,255,255,0.06)' }} />

      {/* Stock & Debate Settings */}
      <div className="space-y-3">
        <InputField
          label="📈 Stock Symbol"
          value={symbol}
          onChange={v => setSymbol(v.toUpperCase())}
          placeholder="AAPL"
          maxLength={5}
        />
        <div>
          <label className="block text-xs text-gray-400 mb-1 font-medium">🥊 Max Debate Rounds: <span className="text-indigo-400">{maxRounds}</span></label>
          <input
            type="range"
            min={1}
            max={5}
            value={maxRounds}
            onChange={e => setMaxRounds(Number(e.target.value))}
            className="w-full accent-indigo-500"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-0.5">
            <span>1</span><span>5</span>
          </div>
        </div>
        <SelectField
          label="📅 Chart Period"
          value={chartPeriod}
          onChange={setChartPeriod}
          options={CHART_PERIODS.map(p => ({ value: p, label: CHART_PERIOD_LABELS[p] }))}
        />
      </div>

      <div style={{ height: '1px', background: 'rgba(255,255,255,0.06)' }} />

      {/* Agent Personalities */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-gray-300">🧑‍💻 Customize Agent Personalities</p>
        <CollapsibleSection title="Agent Bull Personality" emoji="🐂">
          <SelectField label="Risk Tolerance" value={bullRisk} onChange={setBullRisk} options={RISK_LEVELS} />
          <TextAreaField label="Focus Areas (comma-separated)" value={bullFocus} onChange={setBullFocus} rows={2} />
          <TextAreaField label="Trading Style" value={bullStyle} onChange={setBullStyle} rows={2} />
          <TextAreaField label="Key Beliefs" value={bullBeliefs} onChange={setBullBeliefs} rows={2} />
        </CollapsibleSection>
        <CollapsibleSection title="Agent Bear Personality" emoji="🐻">
          <SelectField label="Risk Tolerance" value={bearRisk} onChange={setBearRisk} options={RISK_LEVELS} />
          <TextAreaField label="Focus Areas (comma-separated)" value={bearFocus} onChange={setBearFocus} rows={2} />
          <TextAreaField label="Trading Style" value={bearStyle} onChange={setBearStyle} rows={2} />
          <TextAreaField label="Key Beliefs" value={bearBeliefs} onChange={setBearBeliefs} rows={2} />
        </CollapsibleSection>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Start Debate Button */}
      <button
        onClick={handleSubmit}
        disabled={isLoading}
        className="w-full py-3 px-6 rounded-xl font-bold text-white text-sm tracking-wide transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
        style={{
          background: isLoading
            ? 'rgba(99,102,241,0.4)'
            : 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
          boxShadow: isLoading ? 'none' : '0 10px 20px rgba(99,102,241,0.35)',
          transform: isLoading ? 'none' : undefined,
        }}
        onMouseEnter={e => { if (!isLoading) e.currentTarget.style.transform = 'translateY(-2px)'; }}
        onMouseLeave={e => { e.currentTarget.style.transform = 'none'; }}
      >
        {isLoading ? '⏳ Debate in Progress...' : '🎯 Start Debate'}
      </button>

      <p className="text-xs text-gray-600 text-center pb-2">
        ⚠️ For educational purposes only. Not financial advice.
      </p>
    </aside>
  );
}
