import os
from textwrap import dedent
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai.tools import tool
import yfinance as yf
import pandas as pd
from data_fetcher import NewsContentExtractor, RedditNewsExtractor, StockDataFetcher

# ==========================================
# 1. DEFINE TOOLS FOR AGENTS
# ==========================================

@tool("Get Technical Indicators")
def get_technical_indicators(symbol: str) -> str:
    """Useful to get the current price, volume, RSI, MACD, Support/Resistance and Moving Averages for a stock."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="6mo")
        if hist.empty:
            return f"Error: No historical data found for {symbol}."
            
        current_price = hist['Close'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        
        # Calculate RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not rsi.empty else 0.0
        
        # MACD
        ema_fast = hist['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        current_macd = macd_line.iloc[-1] if not macd_line.empty else 0.0

        ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        
        support = hist['Low'].rolling(window=20).min().iloc[-1]
        resistance = hist['High'].rolling(window=20).max().iloc[-1]

        return dedent(f"""
            Technical Data for {symbol}:
            - Current Price: ${current_price:.2f}
            - Volume: {volume:,}
            - RSI (14): {current_rsi:.1f}
            - MACD: {current_macd:.3f}
            - 20-day MA: ${ma_20:.2f}
            - 50-day MA: ${ma_50:.2f}
            - Support Level: ${support:.2f}
            - Resistance Level: ${resistance:.2f}
        """)
    except Exception as e:
        return f"Error fetching technicals for {symbol}: {str(e)}"

@tool("Get Recent News")
def get_recent_news(symbol: str) -> str:
    """Useful to get the latest news headlines and context for a stock."""
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        if not news:
            return f"No recent news found for {symbol}."
        
        content_extractor = NewsContentExtractor()
        news_summary = f"Recent News for {symbol}:\n"
        
        for item in news[:4]:
            content_data = item.get('content', {})
            title = content_data.get('title', item.get('title', 'No title available'))
            
            link = None
            if content_data.get('clickThroughUrl', {}).get('url'):
                link = content_data['clickThroughUrl']['url']
            elif content_data.get('canonicalUrl', {}).get('url'):
                link = content_data['canonicalUrl']['url']
            else:
                link = item.get('link', '#')
                
            publisher = content_data.get('provider', {}).get('displayName', item.get('publisher', 'Unknown'))
            
            if link and link != '#':
                content = content_extractor.extract_content_from_url(link)
                content_preview = content[:200] + "..." if len(content) > 200 else content
            else:
                content_preview = "Content not extractable."
                
            news_summary += f"\n- Title: {title}\n  Publisher: {publisher}\n  Preview: {content_preview}\n"
            
        return news_summary
    except Exception as e:
        return f"Error fetching news for {symbol}: {str(e)}"

@tool("Get Reddit Sentiment")
def get_reddit_sentiment(symbol: str) -> str:
    """Useful to get the current retail sentiment and discussions from Reddit for a stock."""
    try:
        # We try to use read-only Reddit API or just simulate if credentials aren't set up optimally
        extractor = RedditNewsExtractor()
        discussions = extractor.fetch_reddit_discussions(symbol, limit=4)
        if not discussions:
            return f"No recent Reddit discussions found for {symbol}."
            
        res = f"Reddit Discussions for {symbol}:\n"
        for d in discussions:
            res += f"- r/{d['subreddit']} | Score: {d['score']} | {d['title']}\n  {d['content'][:150]}...\n"
        return res
    except Exception as e:
        return f"Error fetching Reddit data: {str(e)}"

# ==========================================
# 2. DEFINE AGENTS
# ==========================================

def create_trading_agents(bull_focus: str, bear_focus: str, api_key: str):
    """Creates and returns the CrewAI agents."""
    
    # Set the LLM up explicitly
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7) # Using a slightly better model default if available
    
    # 1. Researcher / Data Gatherer Agent
    researcher = Agent(
        role='Market Data Researcher',
        goal='Gather all comprehensive data (Technicals, News, Sentiment) for the requested stock.',
        backstory="""You are an elite data scientist working at a hedge fund. Your job is to scour 
                     markets, news, and social media to compile a perfect dossier of information 
                     so analysts can make accurate predictions.""",
        verbose=True,
        allow_delegation=False,
        tools=[get_technical_indicators, get_recent_news, get_reddit_sentiment],
        llm=llm
    )
    
    # 2. Bullish Analyst
    bull_agent = Agent(
        role='Bullish Trading Analyst',
        goal='Formulate a strong bullish argument for day trading the stock based on data.',
        backstory=f"""You are an aggressive, optimistic trader. You look for upside potential, breakouts, 
                      and positive catalysts. Your focus areas are: {bull_focus}. You always look for the bull case.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    # 3. Bearish Analyst
    bear_agent = Agent(
        role='Bearish Trading Analyst',
        goal='Formulate a robust bearish argument and risk analysis for day trading the stock based on data.',
        backstory=f"""You are a defensive, skeptical risk manager. You identify overhead resistance, 
                      overbought conditions, and negative catalysts. Your focus areas are: {bear_focus}. 
                      You emphasize capital preservation and shorting opportunities.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    # 4. Chief Risk Officer (Judge)
    judge_agent = Agent(
        role='Chief Risk Officer',
        goal='Weigh the bull and bear arguments and provide a final actionable trading recommendation.',
        backstory="""You are the ultimate decision maker at the firm. You hear the arguments from both 
                     the bullish and bearish analysts, strip away the emotion, verify the data, and make 
                     the final call on what the firm should trade today.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    return researcher, bull_agent, bear_agent, judge_agent

