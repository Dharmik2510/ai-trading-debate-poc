# AI Trading Debate Platform

🚀 **AI Trading Debate Platform** is an interactive web application where two AI agents (Bull 🐂 and Bear 🐻) debate whether a stock is a good candidate for day trading, using real-time data, technical indicators, and news sentiment analysis powered by OpenAI's GPT models.

**Built with React.js + Tailwind CSS (frontend) and FastAPI (backend).**

## Features

- **AI Debate:** Watch two AI agents (Bull and Bear) debate the merits and risks of day trading a stock, powered by CrewAI multi-agent framework.
- **Customizable Personalities:** Configure each agent's risk tolerance, style, and focus areas.
- **Real-Time Data:** Fetches live stock data, technical indicators (RSI, MACD, MAs, Bollinger Bands, ATR), and recent news.
- **News Sentiment Analysis:** Uses OpenAI LLM to analyze news headlines for bullish/bearish/neutral sentiment.
- **Reddit Sentiment:** Optional Reddit community sentiment analysis (requires Reddit API credentials).
- **Live Streaming Debate:** Real-time SSE streaming of each debate round as it unfolds.
- **Final Verdict:** Chief Risk Officer agent renders a final BUY/SELL/HOLD recommendation.
- **Beautiful UI:** Glassmorphism dark theme with animated agent cards and a responsive layout.

## Architecture

```
ai-trading-debate-poc/
├── src/                    # Python Backend (FastAPI)
│   ├── api.py              # FastAPI application with SSE streaming
│   ├── agents.py           # TradingAgent class and personalities
│   ├── crew_agentic_workflow.py  # CrewAI agent definitions and tools
│   ├── data_fetcher.py     # Stock data, news, Reddit sentiment
│   ├── llm_utils.py        # LLM helper functions
│   ├── ui_utils.py         # Legacy UI utilities (for Streamlit app)
│   ├── app.py              # Legacy Streamlit application (still functional)
│   └── requirements.txt    # Python dependencies
└── frontend/               # React.js Frontend
    ├── src/
    │   ├── App.jsx          # Main app with SSE streaming
    │   ├── index.css        # Tailwind CSS + custom styles
    │   └── components/
    │       ├── ConfigPanel.jsx      # Sidebar configuration
    │       ├── StockMetrics.jsx     # Technical indicators dashboard
    │       ├── DebateArena.jsx      # Live debate display
    │       ├── FinalVerdict.jsx     # Chief Risk Officer verdict
    │       └── SentimentSections.jsx # News & Reddit sentiment
    ├── package.json
    └── vite.config.js       # Vite + Tailwind CSS config
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API key (from [platform.openai.com](https://platform.openai.com/))
- Reddit API credentials (optional, from [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps))

### 1. Backend Setup

```bash
cd src
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### 3. Using the App

1. Enter your **OpenAI API key** in the sidebar
2. Optionally enter **Reddit API credentials** for community sentiment
3. Enter a **stock symbol** (e.g., AAPL, TSLA, GOOGL)
4. Configure **debate rounds** (1-5) and **chart period**
5. Optionally customize **agent personalities**
6. Click **🎯 Start Debate** and watch in real-time!

## Legacy Streamlit App

The original Streamlit app is still available:

```bash
cd src
pip install streamlit plotly
streamlit run app.py
```

## Notes

- ⚠️ **For educational/demo use only.** Not financial advice.
- Requires a valid OpenAI API key for all LLM-powered features.
- News sentiment and technical indicators are for demonstration purposes.
- Day trading carries significant financial risk.

## License

MIT License

---

*Created by Dharmik Soni, 2025.*
