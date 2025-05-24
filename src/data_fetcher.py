# data_fetcher.py
import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime

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
    def get_stock_data(symbol: str, period: str = "1y") -> StockData:
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

            # Fetch news and perform basic sentiment analysis
            news_data = StockDataFetcher.fetch_news_sentiment(symbol)
            
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
    def fetch_news_sentiment(symbol: str, limit: int = 5) -> dict:
        """
        Fetches recent news headlines for a symbol and performs a basic sentiment analysis.
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

            # Simple keyword-based sentiment for news
            bullish_keywords = ["gain", "rise", "grow", "positive", "strong", "beats", "upgrade", 
                            "acquisition", "partnership", "increase", "profit", "expansion", 
                            "record", "surge", "rally", "buy", "outperform", "success"]
            bearish_keywords = ["drop", "fall", "lose", "negative", "weak", "misses", "downgrade", 
                            "investigation", "lawsuit", "cut", "decline", "loss", "recession", 
                            "slump", "plunge", "sell", "underperform", "tariff", "threat"]

            for item in news[:limit]:
                # Extract content from the nested structure
                content = item.get('content', {})
                
                # Get title - try multiple fields
                title = content.get('title', 'No title available').strip()
                if not title:
                    title = content.get('summary', 'No title available').split('.')[0].strip()
                
                # Get publisher - try multiple fields
                publisher = 'Unknown Publisher'
                provider = content.get('provider', {})
                if isinstance(provider, dict):
                    publisher = provider.get('displayName', 'Unknown Publisher')
                elif isinstance(provider, str):
                    publisher = provider
                
                # Get link - try multiple fields
                link = content.get('canonicalUrl', {}).get('url', '#')
                if link == '#':
                    link = content.get('clickThroughUrl', {}).get('url', '#')
                
                # Sentiment analysis
                sentiment = "neutral"
                title_lower = title.lower()
                
                # Check for bullish keywords
                if any(keyword in title_lower for keyword in bullish_keywords):
                    sentiment = "bullish"
                    bullish_count += 1
                # Check for bearish keywords
                elif any(keyword in title_lower for keyword in bearish_keywords):
                    sentiment = "bearish"
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