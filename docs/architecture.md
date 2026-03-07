# AI Trading Debate Platform — Architecture

This document describes how the multi-agent orchestration is structured, what each agent does, and how the live debate is handled end-to-end.

## Visual Overview

![Architecture Diagram](architecture.png)

---

## System Components

### (1) User & React Frontend

| Component | File | Purpose |
|-----------|------|---------|
| **ConfigPanel** | `ConfigPanel.jsx` | API keys, stock symbol, debate rounds, agent personality sliders |
| **StockMetrics** | `StockMetrics.jsx` | Colour-coded dashboard: RSI, MACD, MA20/50, Bollinger Bands, ATR |
| **DebateArena** | `DebateArena.jsx` | Live feed of researcher report and bull/bear round messages |
| **FinalVerdict** | `FinalVerdict.jsx` | Chief Risk Officer BUY / SELL / HOLD recommendation |
| **SentimentSections** | `SentimentSections.jsx` | News articles and Reddit posts with sentiment badges |

The frontend uses a **custom SSE parser** (`App.jsx`) that reads the streaming response body with `fetch()` and incrementally updates each component as events arrive.

---

### (2) FastAPI Backend & SSE Streaming

```
POST /api/debate/stream
  └─> StreamingResponse (text/event-stream)
        └─> Async Generator  (asyncio.run_in_executor)
              └─> yield  "data: {json}\n\n"
```

The backend never blocks the event loop — each synchronous LLM call is offloaded with `asyncio.run_in_executor` so the SSE connection stays alive throughout the entire debate.

**SSE Event Types**

| Event | Payload | When |
|-------|---------|------|
| `status` | `{message}` | Progress updates throughout |
| `stock_data` | Full `StockData` object | After data fetch |
| `research` | Researcher's report text | After Task 1 |
| `bull` | `{round, content}` | After each bull turn |
| `bear` | `{round, content}` | After each bear turn |
| `verdict` | CRO recommendation text | After final round |
| `complete` | `{}` | Debate finished |
| `error` | `{message}` | On any exception |

---

### (3) Data Fetcher (`data_fetcher.py`)

```
StockDataFetcher.get_stock_data(symbol)
  ├── yfinance           →  60-day OHLCV + real-time quote
  ├── Yahoo Finance RSS  →  BeautifulSoup article scraping
  │    └── OpenAI GPT-3.5-turbo  →  headline sentiment score
  └── Reddit PRAW        →  5 subreddits, top posts & comments
       └── OpenAI GPT-3.5-turbo  →  community sentiment score

Technical indicators computed locally from OHLCV:
  RSI (14)  |  MACD (12/26/9)  |  MA20 / MA50
  Bollinger Bands (20, ±2σ)  |  ATR (14)
  Support (20-period low)  |  Resistance (20-period high)

Returns: StockData dataclass
  price, change_pct, volume, RSI, MACD, MA20, MA50,
  bollinger_upper/lower, ATR, support, resistance,
  news_sentiment{}, reddit_sentiment{}
```

---

### (4) CrewAI Agent Orchestration (`crew_agentic_workflow.py`)

```
CrewAI Orchestrator  (gpt-4o-mini, temperature=0.7)
│
├── Task 1 — Researcher Agent
│     Role  : Market Data Researcher
│     Tools : get_technical_indicators()
│             get_recent_news()
│             get_reddit_sentiment()
│     Output: Comprehensive research report (passed as context to all agents)
│
├── Task 2 — Debate Loop  (Rounds 1 … max_rounds)
│     ┌─ Bull Agent  (Bullish Trading Analyst)
│     │    Task  : Formulate strongest BUY / LONG argument
│     │    Input : Research report + Bear's previous argument (from round N-1)
│     │    Output: 2-3 paragraph bullish thesis with entry price suggestion
│     │    SSE   : bull event streamed to frontend
│     │
│     └─ Bear Agent  (Bearish Trading Analyst)
│          Task  : Formulate strongest SELL / SHORT argument
│          Input : Research report + Bull's argument (current round)
│          Output: 2-3 paragraph bearish thesis with risk / exit levels
│          SSE   : bear event streamed to frontend
│
└── Task 3 — Judge Agent  (Chief Risk Officer)
      Input : Final bull & bear arguments + full research context
      Output: Final recommendation
        - Decision    : BUY / SELL / HOLD
        - Confidence  : 1-10
        - Entry price
        - Target price
        - Stop-loss level
        - Risk level  : 1-10
        - Reasoning   : concise summary
      SSE   : verdict event streamed to frontend
```

#### Agent Personalities (configurable via API)

| Parameter | Bull default | Bear default |
|-----------|-------------|-------------|
| Risk tolerance | Medium-High | Low |
| Trading style | Aggressive momentum | Conservative risk-aware |
| Focus areas | Breakouts, upside momentum | Resistance, overbought, news risks |

Personalities are passed in the `DebateRequest` body (`bull_risk_tolerance`, `bull_focus_areas`, `bull_style`, `bull_beliefs`, and matching bear fields).

---

### (5) External APIs & AI Models

| Service | Used By | Purpose |
|---------|---------|---------|
| OpenAI GPT-4o-mini | Researcher, Bull, Bear, Judge | Reasoning and argument generation |
| OpenAI GPT-3.5-turbo | Data Fetcher | News & Reddit sentiment scoring |
| yfinance | Data Fetcher | Historical OHLCV + real-time quote |
| Yahoo Finance RSS | Data Fetcher | News headlines + scraped article content |
| Reddit API (PRAW) | Data Fetcher | Community posts from r/stocks, r/investing, r/StockMarket, r/SecurityAnalysis, r/wallstreetbets |

---

## End-to-End Live Debate Flow

```
User clicks "Start Debate"
        │
        ▼
POST /api/debate/stream
        │
        ├─ status: "Fetching stock data for <SYMBOL>..."
        │
        ├─ StockDataFetcher.get_stock_data()
        │      (yfinance + Yahoo News + Reddit + GPT sentiment)
        │
        ├─ stock_data: { price, RSI, MACD, ... }
        │
        ├─ status: "Initialising CrewAI agents..."
        ├─ status: "Researcher is gathering data..."
        │
        ├─ Researcher Agent runs (tools: indicators, news, reddit)
        │
        ├─ research: "<full research report>"
        │
        ├─ FOR round = 1 to max_rounds:
        │     │
        │     ├─ status: "Agent Bull formulating case (Round N)..."
        │     ├─ Bull Agent generates bullish argument
        │     ├─ bull: { round: N, content: "..." }
        │     │
        │     ├─ status: "Agent Bear formulating counter-argument (Round N)..."
        │     ├─ Bear Agent reacts to Bull's argument
        │     └─ bear: { round: N, content: "..." }
        │
        ├─ status: "Chief Risk Officer rendering final verdict..."
        ├─ Judge Agent synthesises all arguments
        ├─ verdict: "Decision: BUY\nConfidence: 7/10\nEntry: ..."
        │
        └─ complete: {}
```

---

## Sequence Diagram

```mermaid
sequenceDiagram
    participant U  as User (React)
    participant F  as FastAPI
    participant D  as DataFetcher
    participant O  as CrewAI Orchestrator
    participant R  as Researcher Agent
    participant Bu as Bull Agent
    participant Be as Bear Agent
    participant J  as Judge Agent

    U->>F: POST /api/debate/stream
    F-->>U: SSE: status "Fetching data..."
    F->>D: get_stock_data(symbol)
    D-->>F: StockData
    F-->>U: SSE: stock_data

    F->>O: run_crew(StockData, rounds)
    O->>R: Task 1 — gather data
    R->>R: get_technical_indicators()
    R->>R: get_recent_news()
    R->>R: get_reddit_sentiment()
    R-->>O: research report
    O-->>F: research report
    F-->>U: SSE: research

    loop Debate rounds
        O->>Bu: Task 2a — argue BUY (context = research + prev bear)
        Bu-->>O: bullish argument
        O-->>F: bull output
        F-->>U: SSE: bull

        O->>Be: Task 2b — argue SELL (context = research + bull)
        Be-->>O: bearish argument
        O-->>F: bear output
        F-->>U: SSE: bear
    end

    O->>J: Task 3 — render verdict
    J-->>O: BUY/SELL/HOLD + levels
    O-->>F: final verdict
    F-->>U: SSE: verdict
    F-->>U: SSE: complete
```

---

## Agent Orchestration Diagram

```mermaid
flowchart TD
    subgraph Input["Data Input"]
        YF[yfinance\nOHLCV data]
        YN[Yahoo Finance News\nBeautifulSoup + GPT-3.5]
        RD[Reddit PRAW\n5 subreddits + GPT-3.5]
    end

    subgraph SD["StockData"]
        SD1[price · RSI · MACD\nMA20 · MA50 · Bollinger\nATR · Support · Resistance\nnews_sentiment · reddit_sentiment]
    end

    subgraph Crew["CrewAI Orchestration (gpt-4o-mini)"]
        R[Researcher Agent\nTools: indicators + news + reddit\nOutput: research report]
        Bu[Bull Agent\nBUY / LONG argument\n2-3 paragraphs per round]
        Be[Bear Agent\nSELL / SHORT argument\n2-3 paragraphs per round]
        J[Judge Agent\nChief Risk Officer\nFinal recommendation]
    end

    subgraph Verdict["Final Output"]
        V[Decision: BUY / SELL / HOLD\nConfidence · Entry · Target\nStop-Loss · Reasoning]
    end

    YF --> SD1
    YN --> SD1
    RD --> SD1
    SD1 --> R
    R -- research context --> Bu
    R -- research context --> Be
    Bu <--> Be
    Bu --> J
    Be --> J
    J --> V

    style Crew fill:#1a1a2e,stroke:#f59e0b,color:#fff
    style Input fill:#0a2a1a,stroke:#10b981,color:#fff
    style SD fill:#0a1a2a,stroke:#3b82f6,color:#fff
    style Verdict fill:#1a1a3e,stroke:#6366f1,color:#fff
```

---

*For educational / demonstration use only — not financial advice.*
