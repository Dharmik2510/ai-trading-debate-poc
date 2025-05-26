import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
import openai
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin, urlparse
import praw  # Reddit API wrapper

@dataclass
class NewsItem:
    """Represents a single news item with full content and analysis"""
    title: str
    publisher: str
    link: str
    content: str
    sentiment: str
    source_type: str  # 'yahoo_finance' or 'reddit'
    score: int = 0  # For Reddit posts/comments
    
@dataclass
class StockData:
    """
    Dataclass to hold all relevant stock information and calculated technical indicators.
    """
    symbol: str
    price: float
    change_pct: float
    volume: int
    rsi: float
    macd: float
    ma_20: float
    ma_50: float
    support: float
    resistance: float
    bb_upper: float
    bb_lower: float
    atr: float
    news_sentiment: dict = None
    reddit_sentiment: dict = None

class NewsContentExtractor:
    """Handles extracting and parsing news content from various sources"""
    
    @staticmethod
    def extract_content_from_url(url: str, timeout: int = 10) -> str:
        """
        Extracts the main text content from a news article URL.
        Returns cleaned text content or empty string if extraction fails.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Try to find the main content using common article selectors
            content_selectors = [
                'article', 
                '[role="main"]',
                '.article-content',
                '.story-content', 
                '.entry-content',
                '.post-content',
                '.article-body',
                '.story-body',
                '.content-body',
                'main',
                '.main-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # Get text from all matching elements
                    for element in elements:
                        content += element.get_text(separator=' ', strip=True) + " "
                    break
            
            # Fallback: get all paragraph text if no main content found
            if not content.strip():
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Clean and limit content
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Limit content length to avoid token limits (keep first 2000 characters)
            if len(content) > 2000:
                content = content[:2000] + "..."
                
            return content
            
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return ""

class RedditNewsExtractor:
    """Handles fetching and analyzing Reddit discussions about stocks"""
    
    def __init__(self, client_id: str = None, client_secret: str = None, user_agent: str = "StockDebateBot"):
        """
        Initialize Reddit API client. If credentials not provided, will use read-only mode.
        """
        self.reddit = None
        try:
            if client_id and client_secret:
                print(f"[DEBUG] Initializing Reddit with credentials: {client_id[:8]}...")
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
                # Test the connection
                print(f"[DEBUG] Reddit connection test - read_only: {self.reddit.read_only}")
            else:
                print("[DEBUG] No Reddit credentials provided, skipping Reddit analysis")
                self.reddit = None
        except Exception as e:
            print(f"[ERROR] Reddit API initialization failed: {e}")
            self.reddit = None
    
    def fetch_reddit_discussions(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Fetches recent Reddit discussions about a stock symbol from relevant subreddits.
        Returns list of discussion items with content and metadata.
        """
        if not self.reddit:
            print("[DEBUG] Reddit client not available, returning empty discussions")
            return []
            
        discussions = []
        
        # Relevant subreddits for stock discussions
        subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'StockMarket', 'wallstreetbets']
        
        try:
            for subreddit_name in subreddits[:3]:  # Limit to first 3 to avoid rate limits
                try:
                    print(f"[DEBUG] Searching r/{subreddit_name} for {symbol}")
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts mentioning the stock symbol
                    search_query = f"${symbol} OR {symbol}"
                    print(f"[DEBUG] Search query: {search_query}")
                    
                    search_results = list(subreddit.search(search_query, limit=5, time_filter='week'))
                    print(f"[DEBUG] Search results: {search_results}")
                    print(f"[DEBUG] Found {len(search_results)} results in r/{subreddit_name}")
                    
                    for submission in search_results:
                        if submission.selftext or submission.title:
                            content = f"{submission.title}\n\n{submission.selftext}"
                            
                            discussions.append({
                                'title': submission.title,
                                'content': content[:1500],  # Limit content length
                                'score': submission.score,
                                'subreddit': subreddit_name,
                                'url': f"https://reddit.com{submission.permalink}",
                                'created_utc': submission.created_utc
                            })
                            
                            # Get top comments for additional context
                            try:
                                submission.comments.replace_more(limit=0)
                                for comment in submission.comments[:2]:  # Top 2 comments
                                    if hasattr(comment, 'body') and len(comment.body) > 50:  # Only meaningful comments
                                        discussions.append({
                                            'title': f"Comment on: {submission.title[:50]}...",
                                            'content': comment.body[:1000],
                                            'score': comment.score,
                                            'subreddit': subreddit_name,
                                            'url': f"https://reddit.com{comment.permalink}",
                                            'created_utc': comment.created_utc
                                        })
                            except Exception as comment_error:
                                print(f"[DEBUG] Error processing comments: {comment_error}")
                                continue
                                    
                except Exception as e:
                    print(f"[ERROR] Error fetching from r/{subreddit_name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"[ERROR] Error in Reddit discussions fetch: {e}")
            
        # Sort by score (popularity) and recency
        discussions.sort(key=lambda x: (x['score'], x['created_utc']), reverse=True)
        final_discussions = discussions[:limit]
        print(f"[DEBUG] Returning {len(final_discussions)} total discussions")
        return final_discussions

class StockDataFetcher:
    """
    Enhanced stock data fetcher with comprehensive news analysis from multiple sources.
    """
    
    @staticmethod
    def get_stock_data(symbol: str, period: str = "1y", api_key: str = "", reddit_credentials: dict = None) -> StockData:
        """
        Fetches comprehensive stock data including technical indicators and multi-source news analysis.
        
        Args:
            symbol: Stock symbol
            period: Historical data period
            api_key: OpenAI API key for LLM analysis
            reddit_credentials: Dict with 'client_id' and 'client_secret' for Reddit API
        """
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            if hist.empty:
                print(f"Error: No historical data found for {symbol} for the period {period}.")
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price 
            change_pct = ((current_price - prev_price) / prev_price) * 100
            volume = hist['Volume'].iloc[-1]
            
            # Technical indicators
            rsi = StockDataFetcher._calculate_rsi(hist['Close'])
            macd = StockDataFetcher._calculate_macd(hist['Close'])
            ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            
            # Simple Support and Resistance
            support = hist['Low'].rolling(window=20).min().iloc[-1]
            resistance = hist['High'].rolling(window=20).max().iloc[-1]

            # Bollinger Bands
            bb_upper, bb_lower = StockDataFetcher._calculate_bollinger_bands(hist['Close'])

            # ATR
            atr = StockDataFetcher._calculate_atr(hist)

            # Fetch and analyze Yahoo Finance news with full content parsing
            news_data = StockDataFetcher.fetch_enhanced_news_sentiment(symbol, api_key=api_key)
            
            # Fetch and analyze Reddit discussions
            reddit_data = StockDataFetcher.fetch_reddit_sentiment(symbol, api_key=api_key, reddit_credentials=reddit_credentials)
            
            return StockData(
                symbol=symbol.upper(),
                price=float(current_price),
                change_pct=float(change_pct),
                volume=int(volume),
                rsi=float(rsi),
                macd=float(macd),
                ma_20=float(ma_20),
                ma_50=float(ma_50),
                support=float(support),
                resistance=float(resistance),
                bb_upper=float(bb_upper),
                bb_lower=float(bb_lower),
                atr=float(atr),
                news_sentiment=news_data,
                reddit_sentiment=reddit_data
            )
        except Exception as e:
            print(f"Error fetching or calculating data for {symbol}: {str(e)}")
            return None

    @staticmethod
    def fetch_enhanced_news_sentiment(symbol: str, limit: int = 5, api_key: str = "") -> dict:
        """
        Fetches Yahoo Finance news and performs deep content analysis with LLM sentiment analysis.
        """
        try:
            stock = yf.Ticker(symbol)
            news = stock.news
            print(f"[DEBUG] yfinance news for {symbol}: Found {len(news) if news else 0} articles")
            
            if not news:
                return {"summary": "No recent news available.", "articles": [], "sentiment_breakdown": {"bullish": 0, "bearish": 0, "neutral": 0}, "headlines": []}

            news_items = []
            sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
            content_extractor = NewsContentExtractor()
            headlines = []

            for item in news[:limit]:
                # Yahoo's new API structure: news item is a dict with 'content' key
                content_data = item.get('content', {})
                title = content_data.get('title', item.get('title', 'No title available')).strip()
                
                # Try clickThroughUrl, canonicalUrl, or fallback to None
                link = None
                if content_data.get('clickThroughUrl', {}).get('url'):
                    link = content_data['clickThroughUrl']['url']
                elif content_data.get('canonicalUrl', {}).get('url'):
                    link = content_data['canonicalUrl']['url']
                else:
                    link = item.get('link', '#').strip() if item.get('link') else None
                
                publisher = content_data.get('provider', {}).get('displayName', item.get('publisher', 'Unknown Publisher')).strip()
                
                # Skip if link is missing or invalid
                if not link or link == '#':
                    continue
                    
                headlines.append({
                    'title': title,
                    'publisher': publisher,
                    'link': link,
                    'sentiment': 'neutral'  # Will be updated below
                })
                
                # Extract full article content
                full_content = content_extractor.extract_content_from_url(link)
                
                # Perform comprehensive sentiment analysis on full content
                sentiment = "neutral"
                if api_key and (title or full_content):
                    analysis_text = f"Title: {title}\n\nContent: {full_content[:1500]}" if full_content else title
                    sentiment = StockDataFetcher._analyze_comprehensive_sentiment(analysis_text, symbol, api_key)
                
                # Update headline sentiment
                headlines[-1]['sentiment'] = sentiment
                sentiment_counts[sentiment] += 1
                
                news_item = NewsItem(
                    title=title,
                    publisher=publisher,
                    link=link,
                    content=full_content[:500] + "..." if len(full_content) > 500 else full_content,
                    sentiment=sentiment,
                    source_type="yahoo_finance"
                )
                
                news_items.append({
                    "title": news_item.title,
                    "publisher": news_item.publisher,
                    "link": news_item.link,
                    "content_preview": news_item.content,
                    "sentiment": news_item.sentiment,
                    "source": "Yahoo Finance"
                })
                
                time.sleep(1)  # Rate limiting
            
            total_articles = len(news_items)
            summary = f"Analyzed {total_articles} Yahoo Finance articles with full content. "
            summary += f"Bullish: {sentiment_counts['bullish']}, Bearish: {sentiment_counts['bearish']}, Neutral: {sentiment_counts['neutral']}."

            return {
                "summary": summary,
                "articles": news_items,
                "sentiment_breakdown": sentiment_counts,
                "total_articles": total_articles,
                "headlines": headlines
            }
            
        except Exception as e:
            print(f"Error in enhanced news sentiment analysis for {symbol}: {str(e)}")
            return {"summary": "Could not fetch enhanced news.", "articles": [], "sentiment_breakdown": {"bullish": 0, "bearish": 0, "neutral": 0}, "headlines": []}

    @staticmethod
    def fetch_reddit_sentiment(symbol: str, api_key: str = "", reddit_credentials: dict = None) -> dict:
        """
        Fetches Reddit discussions and performs sentiment analysis.
        """
        try:
            print(f"[DEBUG] Starting Reddit sentiment analysis for {symbol}")
            print(f"[DEBUG] Reddit credentials provided: {reddit_credentials is not None}")
            
            # Initialize Reddit extractor with provided credentials
            client_id = None
            client_secret = None
            
            if reddit_credentials:
                client_id = reddit_credentials.get('client_id')
                client_secret = reddit_credentials.get('client_secret')
                print(f"[DEBUG] Using provided credentials: {client_id[:8] if client_id else 'None'}...")
            
            reddit_extractor = RedditNewsExtractor(
                client_id=client_id,
                client_secret=client_secret
            )
            
            discussions = reddit_extractor.fetch_reddit_discussions(symbol, limit=10)
            print(f"[DEBUG] Fetched {len(discussions)} Reddit discussions")
            
            if not discussions:
                return {"summary": "No Reddit discussions found.", "discussions": [], "sentiment_breakdown": {"bullish": 0, "bearish": 0, "neutral": 0}}
            
            reddit_items = []
            sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
            
            for discussion in discussions:
                sentiment = "neutral"
                if api_key:
                    analysis_text = f"Title: {discussion['title']}\n\nContent: {discussion['content']}"
                    sentiment = StockDataFetcher._analyze_comprehensive_sentiment(analysis_text, symbol, api_key)
                
                sentiment_counts[sentiment] += 1
                
                reddit_items.append({
                    "title": discussion['title'],
                    "content_preview": discussion['content'][:300] + "..." if len(discussion['content']) > 300 else discussion['content'],
                    "score": discussion['score'],
                    "subreddit": discussion['subreddit'],
                    "url": discussion['url'],
                    "sentiment": sentiment,
                    "source": "Reddit"
                })
            
            total_discussions = len(reddit_items)
            summary = f"Analyzed {total_discussions} Reddit discussions. "
            summary += f"Bullish: {sentiment_counts['bullish']}, Bearish: {sentiment_counts['bearish']}, Neutral: {sentiment_counts['neutral']}."
            
            print(f"[DEBUG] Reddit sentiment analysis complete: {summary}")
            
            return {
                "summary": summary,
                "discussions": reddit_items,
                "sentiment_breakdown": sentiment_counts,
                "total_discussions": total_discussions
            }
            
        except Exception as e:
            print(f"[ERROR] Error in Reddit sentiment analysis for {symbol}: {str(e)}")
            return {"summary": f"Could not fetch Reddit discussions: {str(e)}", "discussions": [], "sentiment_breakdown": {"bullish": 0, "bearish": 0, "neutral": 0}}

    @staticmethod
    def _analyze_comprehensive_sentiment(text: str, symbol: str, api_key: str) -> str:
        """
        Performs comprehensive sentiment analysis using OpenAI LLM with full context.
        """
        if not api_key:
            return "neutral"
        try:
            openai.api_key = api_key
            prompt = f"""
            Analyze the sentiment of this financial content regarding {symbol} stock:

            {text}

            Consider:
            1. Overall tone towards the stock/company
            2. Mention of financial metrics, growth, or performance
            3. Market outlook and future prospects
            4. Risk factors or concerns mentioned
            5. Recommendations or predictions

            Classify the sentiment as:
            - "bullish" if the content is positive/optimistic about the stock
            - "bearish" if the content is negative/pessimistic about the stock  
            - "neutral" if the content is balanced or doesn't express clear direction

            Respond with only one word: bullish, bearish, or neutral.
            """
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert financial sentiment analyst. Analyze the given content and classify sentiment accurately."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            sentiment = response.choices[0].message.content.strip().lower()
            return sentiment if sentiment in ["bullish", "bearish", "neutral"] else "neutral"
        except Exception as e:
            print(f"Error in comprehensive sentiment analysis: {e}")
            return "neutral"

    # Keep existing technical indicator methods
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculates the Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rs = rs.replace([float('inf'), -float('inf')], 0)
        
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 0.0
    
    @staticmethod
    def _calculate_macd(prices, span_fast=12, span_slow=26, span_signal=9):
        """Calculates MACD."""
        ema_fast = prices.ewm(span=span_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=span_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        return macd_line.iloc[-1] if not macd_line.empty else 0.0

    @staticmethod
    def _calculate_bollinger_bands(prices, window=20, num_std_dev=2):
        """Calculates Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)
        return upper_band.iloc[-1] if not upper_band.empty else 0.0, \
               lower_band.iloc[-1] if not lower_band.empty else 0.0

    @staticmethod
    def _calculate_atr(df, window=14):
        """Calculates Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr.iloc[-1] if not atr.empty else 0.0