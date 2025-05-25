import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
import openai # Import OpenAI for LLM calls

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
    news_sentiment: dict = None # New field for news sentiment summary

class StockDataFetcher:
    """
    Handles fetching historical stock data, technical indicator calculations, and news.
    """
    @staticmethod
    def get_stock_data(symbol: str, period: str = "1y", api_key: str = "") -> StockData:
        """
        Fetches stock data for a given symbol and calculates various technical indicators.
        Returns a StockData object or None if fetching fails.
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
            
            # Simple Support and Resistance (20-period min/max)
            support = hist['Low'].rolling(window=20).min().iloc[-1]
            resistance = hist['High'].rolling(window=20).max().iloc[-1]

            # Bollinger Bands
            bb_upper, bb_lower = StockDataFetcher._calculate_bollinger_bands(hist['Close'])

            # ATR
            atr = StockDataFetcher._calculate_atr(hist)

            # Fetch news and perform LLM-based sentiment analysis using OpenAI
            news_data = StockDataFetcher.fetch_news_sentiment(symbol, api_key=api_key)
            
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
                news_sentiment=news_data # Store news sentiment
            )
        except Exception as e:
            print(f"Error fetching or calculating data for {symbol}: {str(e)}")
            return None
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculates the Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rs = rs.replace([float('inf'), -float('inf')], 0) # Replace inf with 0 to avoid NaN in RSI calculation
        
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 0.0
    
    @staticmethod
    def _calculate_macd(prices, span_fast=12, span_slow=26, span_signal=9):
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        ema_fast = prices.ewm(span=span_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=span_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        return macd_line.iloc[-1] if not macd_line.empty else 0.0

    @staticmethod
    def _calculate_bollinger_bands(prices, window=20, num_std_dev=2):
        """Calculates Bollinger Bands (Upper and Lower)."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)
        return upper_band.iloc[-1] if not upper_band.empty else 0.0, \
               lower_band.iloc[-1] if not lower_band.empty else 0.0

    @staticmethod
    def _calculate_atr(df, window=14):
        """Calculates Average True Range (ATR)."""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr.iloc[-1] if not atr.empty else 0.0

    @staticmethod
    def _call_openai_llm_for_sentiment(prompt: str, api_key: str) -> str:
        """
        Makes a call to the OpenAI LLM for sentiment analysis.
        Returns 'bullish', 'bearish', 'neutral', or 'error'.
        """
        if not api_key:
            return "error" # Indicate API key is missing

        try:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", # Using gpt-3.5-turbo as specified for general LLM calls
                messages=[
                    {"role": "system", "content": "You are a highly accurate sentiment analysis AI. You will be given a news headline and must classify its sentiment as 'bullish', 'bearish', or 'neutral' regarding a stock. Respond with only one word: 'bullish', 'bearish', or 'neutral'."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10, # Keep tokens low as we expect a single word response
                temperature=0.0 # Low temperature for consistent classification
            )
            text = response.choices[0].message.content.strip().lower()
            if "bullish" in text:
                return "bullish"
            elif "bearish" in text:
                return "bearish"
            elif "neutral" in text:
                return "neutral"
            else:
                print(f"LLM sentiment response ambiguous: {text}")
                return "neutral" # Default to neutral if LLM response is ambiguous
        except openai.error.AuthenticationError:
            print("OpenAI Authentication Error: Invalid API key.")
            return "error"
        except openai.error.RateLimitError:
            print("OpenAI Rate Limit Exceeded: Please wait and try again.")
            return "error"
        except Exception as e:
            print(f"Unexpected error calling OpenAI API for sentiment: {e}")
            return "error"

    @staticmethod
    def fetch_news_sentiment(symbol: str, limit: int = 5, api_key: str = "") -> dict:
        """
        Fetches recent news headlines for a symbol and performs LLM-based sentiment analysis using OpenAI.
        Returns a dictionary summarizing news sentiment.
        """
        try:
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                return {"summary": "No recent news available.", "bullish_count": 0, "bearish_count": 0, "neutral_count": 0, "headlines": []}

            headlines_with_sentiment = []
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0

            for item in news[:limit]:
                # Debug: Print raw news item structure
                print(f"Raw news item: {item}")  # Debugging line
                
                # Extract content from the nested structure
                content = item.get('content', item)  # Use entire item if no 'content' key
                
                # Get title - try multiple fields
                title = content.get('title', 'No title available').strip()
                if title == 'No title available':
                    title = content.get('summary', 'No title available').split('.')[0].strip()
                
                # Get publisher - handle both direct strings and nested provider objects
                publisher = 'Unknown Publisher'
                if 'publisher' in content:
                    if isinstance(content['publisher'], str):
                        publisher = content['publisher']
                    elif isinstance(content['publisher'], dict):
                        publisher = content['publisher'].get('displayName', 'Unknown Publisher')
                elif 'provider' in content:
                    if isinstance(content['provider'], str):
                        publisher = content['provider']
                    elif isinstance(content['provider'], dict):
                        publisher = content['provider'].get('displayName', 'Unknown Publisher')
                
                # Get link - try multiple fields
                link = '#'
                if 'link' in content and content['link']:
                    link = content['link']
                elif 'canonicalUrl' in content and isinstance(content['canonicalUrl'], dict):
                    link = content['canonicalUrl'].get('url', '#')
                elif 'url' in content:
                    link = content['url']
                
                # Use LLM for sentiment analysis if a title is available and API key is provided
                sentiment = "neutral"  # Default
                
                if title and title != 'No title available' and api_key:
                    sentiment_prompt = (
                        f"Analyze the sentiment of this financial news headline regarding stocks or the market: '{title}'. "
                        "Is it bullish (positive for stocks), bearish (negative for stocks), or neutral? "
                        "Respond with only one word: 'bullish', 'bearish', or 'neutral'."
                    )
                    print(f"Sending to LLM: {sentiment_prompt}")  # Debugging line
                    
                    llm_sentiment = StockDataFetcher._call_openai_llm_for_sentiment(sentiment_prompt, api_key)
                    print(f"LLM response: {llm_sentiment}")  # Debugging line
                    
                    if llm_sentiment in ["bullish", "bearish", "neutral"]:
                        sentiment = llm_sentiment
                    else:
                        print(f"LLM returned unexpected sentiment: {llm_sentiment}")
                
                # Update counts based on determined sentiment
                if sentiment == "bullish":
                    bullish_count += 1
                elif sentiment == "bearish":
                    bearish_count += 1
                else:
                    neutral_count += 1
                
                headlines_with_sentiment.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "sentiment": sentiment
                })
            
            total_news = len(headlines_with_sentiment)
            summary_message = f"Analyzed {total_news} recent news articles. "
            if total_news > 0:
                summary_message += f"Bullish: {bullish_count}, Bearish: {bearish_count}, Neutral: {neutral_count}."

            return {
                "summary": summary_message,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "neutral_count": neutral_count,
                "headlines": headlines_with_sentiment
            }
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            return {"summary": "Could not fetch news.", "bullish_count": 0, "bearish_count": 0, "neutral_count": 0, "headlines": []}
        """
        Fetches recent news headlines for a symbol and performs LLM-based sentiment analysis using OpenAI.
        Returns a dictionary summarizing news sentiment.
        """
        try:
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                return {"summary": "No recent news available.", "bullish_count": 0, "bearish_count": 0, "neutral_count": 0, "headlines": []}

            headlines_with_sentiment = []
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0

            for item in news[:limit]:
                # Extract content from the nested structure (yfinance news sometimes has 'content' dict)
                # Prioritize 'title', then 'summary' from the main item or 'content' dict
                title = item.get('title', '').strip()
                if not title:
                    title = item.get('content', {}).get('title', '').strip()
                if not title:
                    title = item.get('content', {}).get('summary', '').strip()
                if not title:
                    title = item.get('summary', '').strip() # Fallback to summary if no title

                link = item.get('link', '#').strip()
                if not link: # Try canonicalUrl or clickThroughUrl if 'link' is empty
                    link = item.get('canonicalUrl', {}).get('url', '#').strip()
                if not link:
                    link = item.get('clickThroughUrl', {}).get('url', '#').strip()
                if not link:
                    link = '#' # Default if no link found

                # Attempt to get publisher from various keys, prioritizing common ones
                publisher = item.get('publisher', '').strip()
                if not publisher: 
                    publisher = item.get('providerDisplayName', '').strip()
                if not publisher:
                    publisher = item.get('source', '').strip() # 'source' can also be a key
                if not publisher: # If still no publisher, set to a generic placeholder
                    publisher = 'N/A Publisher'

                # Use LLM for sentiment analysis if a title is available and API key is provided
                sentiment = "neutral" # Default if title is empty or LLM fails
                if title and api_key:
                    sentiment_prompt = f"Analyze the sentiment of this news headline regarding a stock: '{title}'. Is it bullish, bearish, or neutral? Respond with only 'bullish', 'bearish', or 'neutral'."
                    llm_sentiment = StockDataFetcher._call_openai_llm_for_sentiment(sentiment_prompt, api_key)
                    if llm_sentiment in ["bullish", "bearish", "neutral"]:
                        sentiment = llm_sentiment
                    else:
                        print(f"LLM sentiment analysis failed for title: '{title}'. LLM response: '{llm_sentiment}'. Defaulting to neutral.")
                        sentiment = "neutral" # Fallback if LLM returns 'error' or unexpected
                elif not api_key:
                    print("API key not provided for LLM sentiment analysis. Defaulting to neutral for news.")
                    sentiment = "neutral" # Default if API key is missing

                # Update counts based on determined sentiment
                if sentiment == "bullish":
                    bullish_count += 1
                elif sentiment == "bearish":
                    bearish_count += 1
                else:
                    neutral_count += 1
                
                headlines_with_sentiment.append({
                    "title": title if title else "No title available", # Store original title or fallback
                    "publisher": publisher,
                    "link": link,
                    "sentiment": sentiment
                })
            
            total_news = len(headlines_with_sentiment)
            summary_message = f"Analyzed {total_news} recent news articles. "
            if total_news > 0:
                summary_message += f"Bullish: {bullish_count}, Bearish: {bearish_count}, Neutral: {neutral_count}."
            else:
                summary_message = "No recent news available to analyze."

            return {
                "summary": summary_message,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "neutral_count": neutral_count,
                "headlines": headlines_with_sentiment
            }
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            return {"summary": "Could not fetch news.", "bullish_count": 0, "bearish_count": 0, "neutral_count": 0, "headlines": []}
