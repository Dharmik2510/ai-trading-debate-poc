# AI Trading Debate Platform — Architecture

> **ai-trading-debate-poc** is an interactive AI-powered financial debate platform where two autonomous agents (🐂 Bull and 🐻 Bear) debate whether a stock is suitable for day trading, powered by real-time market data, technical indicators, news sentiment, and Reddit community analysis.

## Visual Diagram

The full-resolution system architecture diagram is available at:

![Architecture Diagram](docs/architecture.png)

---

## System Layers

The platform is organized into **five distinct layers**:

| Layer | Files | Responsibility |
|---|---|---|
| Presentation | `app.py`, `ui_utils.py` | Streamlit UI, user inputs, debate display |
| Application | `app.py` (orchestration), `llm_utils.py` | Debate coordination, summary, recommendations |
| Agent | `agents.py` | Bull/Bear agent logic, GPT-powered debate turns |
| Data | `data_fetcher.py` | Market data, technical indicators, sentiment |
| External Services | — | OpenAI, Yahoo Finance, Reddit, Web Scraping |

---

## Component Overview

```mermaid
graph TD
    User["👤 User / Browser"]

    subgraph APP["app.py — Streamlit Web Application"]
        Sidebar["🎛️ Sidebar Config<br/>API Keys · Stock Symbol<br/>Debate Rounds · Agent Personalities"]
        Metrics["📊 Metrics Panel<br/>Price · RSI · MACD<br/>Support / Resistance"]
        DebateRenderer["💬 Debate Renderer<br/>Round-by-round messages<br/>Progress tracking"]
        OutputPanel["📋 Output Panel<br/>Summary · Recommendation<br/>Copy-to-Clipboard"]
        SessionState["🗂️ Session State<br/>debate_history · stock_data · recs"]
        UIUtils["ui_utils.py<br/>display_message · CSS styling"]
    end

    subgraph ORCH["Application Orchestration"]
        Orchestrator["🔄 Debate Orchestrator<br/>app.py — round coordination<br/>history accumulation"]
        LLMUtils["🧠 llm_utils.py<br/>generate_debate_summary()<br/>generate_final_recommendation()"]
    end

    subgraph AGENTS["agents.py"]
        BullAgent["🐂 Bull Agent<br/>BullishTrader<br/>analyze() · respond_to()"]
        BearAgent["🐻 Bear Agent<br/>BearishTrader<br/>analyze() · respond_to()"]
        Factory["create_agents() factory"]
    end

    subgraph DATA["data_fetcher.py — StockDataFetcher"]
        GetStockData["get_stock_data()<br/>Main orchestration"]
        TechIndicators["📈 Technical Indicators<br/>RSI · MACD · BB · ATR<br/>MA20/50 · Support/Resistance"]
        NewsAnalysis["📰 News Analysis<br/>fetch_enhanced_news_sentiment()<br/>NewsContentExtractor"]
        RedditAnalysis["💬 Reddit Sentiment<br/>fetch_reddit_sentiment()<br/>RedditNewsExtractor"]
        StockDataObj["📦 StockData dataclass<br/>All metrics + sentiment"]
    end

    subgraph EXT["External Services"]
        OpenAI["🤖 OpenAI GPT-3.5-turbo<br/>Agent debates<br/>Sentiment analysis<br/>Summary + Recommendation"]
        YFinance["📉 Yahoo Finance<br/>yfinance library<br/>Prices · History · News links"]
        WebScrape["🌐 Web Scraping<br/>BeautifulSoup4 + requests<br/>Full article extraction"]
        Reddit["🔴 Reddit PRAW<br/>Discussions · Comments<br/>5 subreddits"]
    end

    User -->|"input / view"| APP
    APP --> Orchestrator
    Orchestrator -->|"analyze() / respond_to()"| BullAgent
    Orchestrator -->|"analyze() / respond_to()"| BearAgent
    Orchestrator -->|"debate history + stock_data"| LLMUtils
    BullAgent -->|"_call_llm()"| OpenAI
    BearAgent -->|"_call_llm()"| OpenAI
    LLMUtils -->|"_call_openai_llm()"| OpenAI
    Factory --> BullAgent
    Factory --> BearAgent
    GetStockData --> TechIndicators
    GetStockData --> NewsAnalysis
    GetStockData --> RedditAnalysis
    TechIndicators --> YFinance
    NewsAnalysis --> YFinance
    NewsAnalysis --> WebScrape
    NewsAnalysis -->|"sentiment analysis"| OpenAI
    RedditAnalysis --> Reddit
    RedditAnalysis -->|"sentiment analysis"| OpenAI
    GetStockData --> StockDataObj
    StockDataObj -->|"stock_data"| Orchestrator
    StockDataObj -->|"stock_data"| BullAgent
    StockDataObj -->|"stock_data"| BearAgent
```

---

## Data Flow

```mermaid
sequenceDiagram
    actor User
    participant App as app.py<br/>(Streamlit)
    participant Fetcher as data_fetcher.py
    participant Agents as agents.py<br/>(Bull + Bear)
    participant LLM as llm_utils.py
    participant OpenAI as OpenAI API
    participant YFinance as Yahoo Finance
    participant Reddit as Reddit PRAW

    User->>App: Enter stock symbol, API keys, config
    App->>Fetcher: get_stock_data(symbol, openai_key, reddit_creds)
    Fetcher->>YFinance: yf.Ticker(symbol) — prices, history, news links
    Fetcher->>YFinance: fetch news article URLs
    Fetcher->>Fetcher: NewsContentExtractor — scrape article content
    Fetcher->>OpenAI: _analyze_comprehensive_sentiment(news text)
    Fetcher->>Reddit: RedditNewsExtractor — fetch discussions
    Fetcher->>OpenAI: _analyze_comprehensive_sentiment(reddit text)
    Fetcher-->>App: StockData (prices + technicals + news + reddit sentiment)

    App->>App: Display stock metrics, news, Reddit sentiment

    loop Each Debate Round
        App->>Agents: bull_agent.analyze(stock_data) [Round 1]
        Agents->>OpenAI: ChatCompletion.create() — bullish analysis
        OpenAI-->>Agents: bullish argument text
        Agents-->>App: Bull message + sentiment

        App->>Agents: bear_agent.analyze(stock_data) [Round 1]
        Agents->>OpenAI: ChatCompletion.create() — bearish analysis
        OpenAI-->>Agents: bearish argument text
        Agents-->>App: Bear message + sentiment

        App->>Agents: bull_agent.respond_to(bear_msg) [Rounds 2+]
        App->>Agents: bear_agent.respond_to(bull_msg) [Rounds 2+]
    end

    App->>LLM: generate_debate_summary(debate_history, stock_data)
    LLM->>OpenAI: ChatCompletion — neutral summary
    OpenAI-->>LLM: summary text
    LLM-->>App: debate summary

    App->>LLM: generate_final_recommendation(debate_history, stock_data)
    LLM->>OpenAI: ChatCompletion — BUY/SELL/HOLD recommendation
    OpenAI-->>LLM: structured recommendation
    LLM-->>App: recommendation (decision, confidence, entry, stop-loss, target)

    App-->>User: Render full debate + summary + recommendation
```

---

## Module Details

### `app.py` — Streamlit Application (757 lines)

The main entry point and UI orchestrator.

| Section | Lines | Responsibility |
|---|---|---|
| Sidebar Config | 256–372 | OpenAI key, Reddit credentials, stock symbol, debate rounds, agent personality customization |
| Data Fetching | 406–436 | Calls `StockDataFetcher.get_stock_data()` with credentials |
| Stock Metrics Display | 442–452 | Price, RSI, MACD, Support, Resistance panels |
| News & Reddit Sentiment | 457–513 | Sentiment breakdown, top discussions |
| Debate Execution | 520–595 | Orchestrates rounds with spinners and progress tracking |
| Summary & Recommendation | 597–628 | Generates and displays final analysis |

**Session State keys**: `debate_history`, `stock_data`, `bull_agent`, `bear_agent`, `final_recommendation`, `debate_summary`

---

### `agents.py` — AI Trading Agents (264 lines)

```mermaid
classDiagram
    class AgentPersonality {
        +str risk_tolerance
        +List~str~ focus_areas
        +str style
        +str beliefs
    }

    class TradingAgent {
        +str name
        +str role
        +str avatar
        +str color
        +AgentPersonality personality
        +str openai_api_key
        +List conversation_history
        +get_system_prompt() str
        +analyze(stock_data) dict
        +respond_to(opponent_name, msg, stock_data) dict
        +_format_news_context(stock_data) str
        +_infer_sentiment(text) str
        +_call_llm(prompt) str
    }

    class BullAgent {
        role = "bullish"
        color = "#03ad2b"
    }

    class BearAgent {
        role = "bearish"
        color = "#e60017"
    }

    TradingAgent --> AgentPersonality
    TradingAgent <|-- BullAgent
    TradingAgent <|-- BearAgent
```

**Key behaviors**:
- Each agent maintains a `conversation_history` list for multi-turn context
- `_infer_sentiment()` uses keyword lists to classify text as bullish / bearish / neutral
- `_call_llm()` wraps `openai.ChatCompletion.create()` with error handling
- `_format_news_context()` injects Yahoo Finance news and Reddit discussions into the GPT prompt

---

### `data_fetcher.py` — Data Collection & Analysis (525 lines)

```mermaid
classDiagram
    class StockData {
        +str symbol
        +float price
        +float change_pct
        +int volume
        +float rsi
        +float macd
        +float ma_20
        +float ma_50
        +float support
        +float resistance
        +float bb_upper
        +float bb_lower
        +float atr
        +dict news_sentiment
        +dict reddit_sentiment
    }

    class NewsItem {
        +str title
        +str content
        +str url
        +str sentiment
        +str source
    }

    class NewsContentExtractor {
        +extract_content_from_url(url) str
    }

    class RedditNewsExtractor {
        +praw.Reddit reddit_client
        +fetch_reddit_discussions(symbol) List
    }

    class StockDataFetcher {
        +get_stock_data(symbol, openai_key, reddit_creds)$ StockData
        +_calculate_rsi(prices, period)$ float
        +_calculate_macd(prices)$ float
        +_calculate_bollinger_bands(prices)$ tuple
        +_calculate_atr(high, low, close)$ float
        +fetch_enhanced_news_sentiment(ticker, openai_key)$ dict
        +fetch_reddit_sentiment(symbol, openai_key, reddit_creds)$ dict
        +_analyze_comprehensive_sentiment(text, openai_key)$ str
    }

    StockDataFetcher --> StockData
    StockDataFetcher --> NewsItem
    StockDataFetcher --> NewsContentExtractor
    StockDataFetcher --> RedditNewsExtractor
```

**Technical indicators computed**:

| Indicator | Period | Method |
|---|---|---|
| RSI | 14 | Relative Strength Index via price deltas |
| MACD | 12/26/9 EMA | Exponential Moving Average crossover |
| Bollinger Bands | 20-period, 2σ | Standard deviation bands |
| ATR | 14 | Average True Range |
| Moving Averages | 20, 50 | Simple Moving Average |
| Support / Resistance | — | Recent low / high over lookback window |

---

### `llm_utils.py` — LLM Utilities (124 lines)

| Function | Model | Temp | Max Tokens | Output |
|---|---|---|---|---|
| `generate_debate_summary()` | GPT-3.5-turbo | 0.4 | 600 | Neutral summary of both sides |
| `generate_final_recommendation()` | GPT-3.5-turbo | 0.5 | 500 | BUY/SELL/HOLD + confidence + entry/stop/target |

**Recommendation structure returned**:
```
Decision:   BUY / SELL / HOLD
Confidence: 1–10
Entry:      $XXX.XX – $XXX.XX
Stop-Loss:  $XXX.XX
Target:     $XXX.XX
Risk Level: 1–10
Reasoning:  <brief synthesis from both agents>
```

---

### `ui_utils.py` — UI Components (64 lines)

| Function | Description |
|---|---|
| `display_message(agent_info, message, sentiment)` | Renders styled agent message card with gradient, color, sentiment emoji |
| `copy_to_clipboard_button(text)` | JavaScript clipboard integration with ✅ feedback animation |

**Color scheme**:
- 🐂 Bull: `#03ad2b` (green)
- 🐻 Bear: `#e60017` (red)
- 📋 Recommendation: `#00bcd4` (cyan)

---

## Technology Stack

```mermaid
graph LR
    subgraph Frontend
        ST["Streamlit<br/>Web Framework"]
    end

    subgraph LLMs
        OAI["OpenAI GPT-3.5-turbo<br/>Debate · Sentiment · Recommendations"]
    end

    subgraph DataSources
        YF["yfinance<br/>Stock Data"]
        BS4["BeautifulSoup4<br/>Web Scraping"]
        PRAW["PRAW<br/>Reddit API"]
    end

    subgraph Processing
        PD["Pandas<br/>Time Series"]
        NP["NumPy<br/>Calculations"]
    end

    ST --> OAI
    ST --> YF
    ST --> BS4
    ST --> PRAW
    ST --> PD
    PD --> NP
```

| Component | Library | Min Version | Auth Required |
|---|---|---|---|
| Web Framework | streamlit | ≥ 1.20.0 | — |
| LLM | openai | ≥ 1.0.0 | ✅ API Key |
| Stock Data | yfinance | ≥ 0.2.0 | — |
| Data Processing | pandas | ≥ 1.5.0 | — |
| Web Scraping | beautifulsoup4, requests | ≥ 4.11.0 | — |
| Reddit | praw | ≥ 7.0.0 | ⚠️ Optional credentials |
| Visualization | plotly | ≥ 5.0.0 | — |
| Language | Python | 3.7+ | — |

---

## Agent Personality Configuration

Users can customize both agents via the sidebar. Each `AgentPersonality` controls:

```mermaid
mindmap
  root((AgentPersonality))
    risk_tolerance
      Low
      Medium
      Medium-High
      High
    focus_areas
      Technical Analysis
      Fundamental Analysis
      Sentiment Analysis
      Volume Analysis
      Momentum
      News Catalyst
    style
      Day Trading
      Swing Trading
      Momentum Trading
    beliefs
      Custom philosophy string
```

---

## Error Handling & Safety

```mermaid
flowchart TD
    A[User submits request] --> B{OpenAI API key\nprovided?}
    B -- No --> C[Show error message\nStop execution]
    B -- Yes --> D[Fetch stock data]
    D --> E{yfinance returns\nvalid data?}
    E -- No --> F[Show error\nEmpty data warning]
    E -- Yes --> G[Fetch news & Reddit]
    G --> H{Reddit credentials\nprovided?}
    H -- No --> I[Skip Reddit\nUse news only]
    H -- Yes --> J[Fetch Reddit data]
    I --> K[Run debate]
    J --> K
    K --> L{LLM call\nsucceeds?}
    L -- Rate limit --> M[Show rate limit error]
    L -- Auth error --> N[Show auth error]
    L -- Yes --> O[Display result]
    O --> P[⚠️ Disclaimer:\nEducational use only]
```

---

## Quick Start Reference

```bash
# Install dependencies
pip install streamlit openai yfinance plotly pandas beautifulsoup4 praw requests

# Run the application
streamlit run src/app.py
```

**Required configuration (sidebar)**:
1. OpenAI API Key
2. Stock Symbol (e.g., `AAPL`, `TSLA`, `NVDA`)
3. Debate Rounds (1–5)
4. *(Optional)* Reddit Client ID + Secret

---

*This document was generated to accompany the architecture diagram at [`docs/architecture.png`](docs/architecture.png).*
