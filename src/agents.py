# agents.py
import openai
from openai.error import AuthenticationError, RateLimitError
from dataclasses import dataclass
from typing import List, Dict, Any

# Assuming StockData is imported or defined elsewhere if this were a standalone module
# For the purpose of this example, we'll assume StockData is available via data_fetcher
from data_fetcher import StockData # Import StockData from the new data_fetcher.py

@dataclass
class AgentPersonality:
    risk_tolerance: str
    focus_areas: List[str]
    style: str
    beliefs: str

class TradingAgent:
    """
    Enhanced AI trading agent that analyzes technical data, Yahoo Finance news content, 
    and Reddit discussions to provide comprehensive trading insights.
    """
    def __init__(self, name: str, role: str, avatar: str, color: str, personality: AgentPersonality):
        self.name = name
        self.role = role
        self.avatar = avatar
        self.color = color
        self.personality = personality
        self.conversation_memory = []

    def get_system_prompt(self) -> str:
        """
        Enhanced system prompt that includes multi-source analysis capabilities.
        """
        focus_areas_str = ", ".join(self.personality.focus_areas)
        return f"""
        You are {self.name}, an expert day trader and market analyst with a {self.role} perspective.
        
        Your personality:
        - Risk tolerance: {self.personality.risk_tolerance}
        - Focus areas: {focus_areas_str}
        - Trading style: {self.personality.style}
        - Key beliefs: {self.personality.beliefs}
        
        Your analysis capabilities:
        1. Technical Analysis: RSI, MACD, Moving Averages, Bollinger Bands, Support/Resistance, ATR
        2. News Analysis: Full article content from Yahoo Finance with sentiment analysis
        3. Social Sentiment: Reddit discussions from investing communities
        4. Market Context: Integration of all data sources for comprehensive insights
        
        Rules for debate:
        1. Stay in character - maintain your {self.role} perspective consistently
        2. Use specific data points from technical indicators, news content, and social sentiment
        3. Reference actual news headlines and Reddit discussions when making arguments
        4. Address opponent's points directly with counter-evidence
        5. Provide actionable trading insights with entry/exit points and risk management
        6. Keep responses focused and professional (2-3 paragraphs max)
        7. Distinguish between different types of evidence (technical vs. fundamental vs. sentiment)
        
        Response structure:
        - Lead with your key argument from your perspective
        - Support with specific evidence from technical, news, or social data
        - Provide actionable trading insight or risk assessment
        - Counter opponent's argument if applicable
        """
    
    def _format_news_context(self, stock_data: StockData) -> str:
        """
        Formats news and Reddit data into a comprehensive context string.
        """
        context_parts = []
        
        # Yahoo Finance News
        if stock_data.news_sentiment and stock_data.news_sentiment.get('articles'):
            context_parts.append("=== YAHOO FINANCE NEWS ===")
            context_parts.append(stock_data.news_sentiment['summary'])
            
            for i, article in enumerate(stock_data.news_sentiment['articles'][:3], 1):
                context_parts.append(f"\nArticle {i}: '{article['title']}' ({article['sentiment'].upper()})")
                context_parts.append(f"Publisher: {article['publisher']}")
                if article.get('content_preview'):
                    context_parts.append(f"Content: {article['content_preview'][:200]}...")
        
        # Reddit Discussions
        if stock_data.reddit_sentiment and stock_data.reddit_sentiment.get('discussions'):
            context_parts.append("\n=== REDDIT DISCUSSIONS ===")
            context_parts.append(stock_data.reddit_sentiment['summary'])
            
            for i, discussion in enumerate(stock_data.reddit_sentiment['discussions'][:3], 1):
                context_parts.append(f"\nDiscussion {i}: '{discussion['title']}' ({discussion['sentiment'].upper()})")
                context_parts.append(f"From r/{discussion['subreddit']} | Score: {discussion['score']}")
                if discussion.get('content_preview'):
                    context_parts.append(f"Content: {discussion['content_preview'][:150]}...")
        
        return "\n".join(context_parts) if context_parts else "No comprehensive news/social data available."

    def analyze(self, stock_data: StockData, context: str = "") -> tuple[str, str]:
        """
        Enhanced analysis incorporating technical data, news content, and Reddit discussions.
        """
        # Format comprehensive news and social context
        news_social_context = self._format_news_context(stock_data)
        
        # Calculate additional technical insights
        ma_signal = "above" if stock_data.price > stock_data.ma_20 else "below"
        rsi_condition = "overbought" if stock_data.rsi > 70 else "oversold" if stock_data.rsi < 30 else "neutral"
        bb_position = "upper band" if stock_data.price > stock_data.bb_upper else "lower band" if stock_data.price < stock_data.bb_lower else "middle range"
        
        prompt = f"""
        Provide a comprehensive day trading analysis for {stock_data.symbol} from your {self.role} perspective.
        
        TECHNICAL DATA:
        - Current Price: ${stock_data.price:.2f} ({stock_data.change_pct:+.2f}%)
        - Price vs 20-day MA: {ma_signal} (${stock_data.ma_20:.2f})
        - RSI: {stock_data.rsi:.1f} ({rsi_condition})
        - MACD: {stock_data.macd:.3f}
        - Bollinger Bands: Price near {bb_position} (Upper: ${stock_data.bb_upper:.2f}, Lower: ${stock_data.bb_lower:.2f})
        - Support/Resistance: ${stock_data.support:.2f} / ${stock_data.resistance:.2f}
        - ATR (Volatility): {stock_data.atr:.2f}
        - Volume: {stock_data.volume:,}
        
        NEWS & SOCIAL SENTIMENT:
        {news_social_context}
        
        Additional Context: {context}
        
        Analyze this data from your {self.role} perspective and provide:
        1. Your overall stance on day trading {stock_data.symbol} today
        2. Key supporting evidence from technical indicators
        3. How news sentiment and Reddit discussions support/contradict your view
        4. Specific entry/exit levels and risk management if recommending a trade
        
        Remember: You are {self.role}, so interpret ambiguous signals through that lens while being objective about the data.
        """
        
        response_text = self._call_llm(prompt)
        sentiment = self._infer_sentiment(response_text)
        return response_text, sentiment
    
    def respond_to(self, opponent_name: str, opponent_message: str, stock_data: StockData) -> tuple[str, str]:
        """
        Enhanced counter-argument incorporating all available data sources.
        """
        news_social_context = self._format_news_context(stock_data)
        
        prompt = f"""
        {opponent_name} just argued:
        "{opponent_message}"
        
        Counter their argument about {stock_data.symbol} while staying true to your {self.role} perspective.
        
        CURRENT MARKET DATA:
        - Price: ${stock_data.price:.2f} ({stock_data.change_pct:+.2f}%)
        - Technical: RSI {stock_data.rsi:.1f} | MACD {stock_data.macd:.3f} | ATR {stock_data.atr:.2f}
        - Key Levels: Support ${stock_data.support:.2f} | Resistance ${stock_data.resistance:.2f}
        - Bollinger Bands: ${stock_data.bb_lower:.2f} - ${stock_data.bb_upper:.2f}
        
        NEWS & SOCIAL SENTIMENT:
        {news_social_context}
        
        Your response should:
        1. Directly address their specific points with counter-evidence
        2. Use technical data, news content, or Reddit sentiment to support your view
        3. Highlight data points they may have overlooked or misinterpreted
        4. Maintain your {self.role} perspective while being factual
        5. Provide specific trading insights that contradict their recommendation
        
        Stay professional but be persuasive with your counter-argument.
        """
        
        response_text = self._call_llm(prompt)
        sentiment = self._infer_sentiment(response_text)
        
        self.conversation_memory.append({
            "opponent": opponent_name,
            "opponent_said": opponent_message,
            "my_response": response_text,
            "sentiment": sentiment
        })
        
        return response_text, sentiment
    
    def _infer_sentiment(self, text: str) -> str:
        """
        Enhanced sentiment inference with more comprehensive keyword analysis.
        """
        text_lower = text.lower()
        
        bullish_keywords = [
            "buy", "long", "breakout", "momentum", "strong", "bullish", "opportunity", 
            "upside", "growth", "support holds", "uptrend", "positive", "increase", 
            "gain", "rally", "bounce", "accumulate", "oversold bounce", "reversal up"
        ]
        
        bearish_keywords = [
            "sell", "short", "bearish", "risk", "overbought", "warning", "downside", 
            "resistance holds", "downtrend", "consolidation", "negative", "decrease", 
            "drop", "decline", "correction", "pullback", "distribution", "weakness"
        ]
        
        bullish_score = sum(1 for keyword in bullish_keywords if keyword in text_lower)
        bearish_score = sum(1 for keyword in bearish_keywords if keyword in text_lower)
        
        if bullish_score > bearish_score:
            return "bullish"
        elif bearish_score > bullish_score:
            return "bearish"
        else:
            return "neutral"

    
    def _call_llm(self, prompt: str) -> str:
        """
        Makes a call to the OpenAI LLM with the given prompt.
        Includes error handling for API issues.
        """
        try:
            # Assuming openai_key is set globally via Streamlit session state in app.py
            import streamlit as st # Only import here if not globally available
            if not st.session_state.get('openai_key'):
                return "⚠️ OpenAI API key required. Please add it in the sidebar."
            openai.api_key = st.session_state.openai_key
            prompt = prompt.strip()
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
            else:
                return "Error: No response from OpenAI."
        except AuthenticationError:
            return "Error: Invalid OpenAI API key. Please check your key in the sidebar."
        except RateLimitError:
            return "Error: OpenAI API rate limit exceeded. Please wait a moment and try again."
        except Exception as e:
            print(f"Error communicating with OpenAI: {e}")
            return f"Error communicating with OpenAI: {str(e)}"

def create_agents(bull_personality: AgentPersonality, bear_personality: AgentPersonality):
    """Initializes and returns the Bull and Bear trading agents with custom personalities."""
    bull_agent = TradingAgent(
        name="Agent Bull 🐂",
        role="bullish",
        avatar="🐂",
        color="#03ad2b",
        personality=bull_personality
    )
    
    bear_agent = TradingAgent(
        name="Agent Bear 🐻",
        role="bearish", 
        avatar="🐻",
        color="#e60017",
        personality=bear_personality
    )
    
    return bull_agent, bear_agent