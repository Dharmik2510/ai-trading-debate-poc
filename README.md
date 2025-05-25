# AI Trading Debate Platform

🚀 **AI Trading Debate Platform** is an interactive Streamlit web app where two AI agents (Bull and Bear) debate whether a stock is a good candidate for day trading, using real-time data, technical indicators, and news sentiment analysis powered by OpenAI's GPT models.

## Features

- **AI Debate:** Watch two AI agents (Bull and Bear) debate the merits and risks of day trading a stock.
- **Customizable Personalities:** Configure each agent's risk tolerance, style, and focus areas.
- **Real-Time Data:** Fetches live stock data, technical indicators (RSI, MACD, MAs, Bollinger Bands, ATR), and recent news.
- **News Sentiment Analysis:** Uses OpenAI LLM to analyze news headlines for bullish/bearish/neutral sentiment.
- **Debate Summary & Recommendation:** Generates a neutral summary and a final actionable trading recommendation.
- **Beautiful UI:** Modern, responsive design with clear message cards and copy-to-clipboard functionality.

## How It Works

1. **Enter your OpenAI API key** in the sidebar (required for LLM features).
2. **Choose a stock symbol** (e.g., AAPL, TSLA, GOOGL).
3. **Configure agent personalities** (optional).
4. **Start the debate** and watch the agents analyze, counter, and summarize.
5. **View the final recommendation** and copy it for your notes.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ai-trading-debate-poc/src
```

### 2. Install Dependencies
Install Python packages (preferably in a virtual environment):
```bash
pip install streamlit openai yfinance plotly pandas
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Get an OpenAI API Key
- Sign up at [OpenAI Platform](https://platform.openai.com/)
- Create an API key and paste it in the app sidebar

## File Structure

- `app.py` — Main Streamlit app
- `agents.py` — AI agent logic and personalities
- `data_fetcher.py` — Stock data, indicators, and news sentiment
- `llm_utils.py` — LLM prompt helpers and summary/recommendation logic
- `ui_utils.py` — UI components and charting

## Notes
- **For educational/demo use only.** Not financial advice.
- Requires a valid OpenAI API key for all LLM-powered features.
- News sentiment and technical indicators are for demonstration and may not reflect real market conditions.

## License
MIT License

---

*Created by Dharmik Soni, 2025.*
