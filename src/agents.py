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
    Represents an AI trading agent (Bull or Bear) with a specific personality
    and methods for analysis and debate.
    """
    def __init__(self, name: str, role: str, avatar: str, color: str, personality: AgentPersonality):
        self.name = name
        self.role = role
        self.avatar = avatar
        self.color = color
        self.personality = personality
        self.conversation_memory = [] # Stores past interactions for potential future use

    def get_system_prompt(self) -> str:
        """
        Generates the system prompt for the OpenAI LLM, defining the agent's persona.
        """
        focus_areas_str = ", ".join(self.personality.focus_areas)
        return f"""
        You are {self.name}, an expert day trader with a {self.role} perspective.
        
        Your personality:
        - Risk tolerance: {self.personality.risk_tolerance}
        - Focus areas: {focus_areas_str}
        - Trading style: {self.personality.style}
        - Key beliefs: {self.personality.beliefs}
        
        Rules for debate:
        1. Stay in character - maintain your {self.role} perspective.
        2. Use specific data points, technical indicators, and recent news sentiment to support arguments.
        3. Address opponent's points directly and professionally.
        4. Provide actionable trading insights (e.g., potential entry/exit points, risks).
        5. Keep responses concise (2-3 paragraphs max).
        
        Always structure responses with:
        - Key point/counterpoint
        - Supporting evidence from data/news
        - Specific trading action/insight (if applicable)
        """
    
    def _infer_sentiment(self, text: str) -> str:
        """
        Infers the sentiment (bullish, bearish, neutral) of a given text based on keywords.
        This is a simple rule-based sentiment analysis.
        """
        text_lower = text.lower()
        bullish_keywords = ["buy", "long", "breakout", "momentum", "strong", "bullish", "opportunity", "upside", "growth", "support holds", "uptrend", "positive", "increase", "gain"]
        bearish_keywords = ["sell", "short", "bearish", "risk", "overbought", "warning", "downside", "resistance holds", "downtrend", "consolidation", "negative", "decrease", "drop"]
        
        if any(keyword in text_lower for keyword in bullish_keywords):
            return "bullish"
        elif any(keyword in text_lower for keyword in bearish_keywords):
            return "bearish"
        else:
            return "neutral"

    def analyze(self, stock_data: StockData, context: str = "") -> tuple[str, str]:
        """
        Generates an initial analysis of the stock from the agent's perspective.
        Returns the analysis text and its inferred sentiment.
        """
        news_context = ""
        if stock_data.news_sentiment and stock_data.news_sentiment['summary'] != "No recent news available.":
            news_context = f"Recent News: {stock_data.news_sentiment['summary']}. Top headlines: " + \
                           "; ".join([f"'{h['title']}' ({h['sentiment']})" for h in stock_data.news_sentiment['headlines'][:2]]) + "."

        prompt = f"""
        Analyze {stock_data.symbol} for day trading from your {self.role} perspective.
        
        Current Data:
        - Price: ${stock_data.price:.2f} ({stock_data.change_pct:+.2f}%)
        - Volume: {stock_data.volume:,}
        - RSI: {stock_data.rsi:.1f}
        - MACD: {stock_data.macd:.3f}
        - 20-day MA: ${stock_data.ma_20:.2f}
        - 50-day MA: ${stock_data.ma_50:.2f}
        - Support: ${stock_data.support:.2f}
        - Resistance: ${stock_data.resistance:.2f}
        - Bollinger Bands: Upper: ${stock_data.bb_upper:.2f}, Lower: ${stock_data.bb_lower:.2f}
        - ATR: {stock_data.atr:.2f}
        
        {news_context}
        
        Context: {context}
        
        Given this data, should we day trade {stock_data.symbol} today? Give your {self.role} analysis and rationale.
        """
        
        response_text = self._call_llm(prompt)
        sentiment = self._infer_sentiment(response_text)
        return response_text, sentiment
    
    def respond_to(self, opponent_name: str, opponent_message: str, stock_data: StockData) -> tuple[str, str]:
        """
        Generates a counter-argument to the opponent's message, maintaining the agent's perspective.
        Returns the response text and its inferred sentiment.
        """
        news_context = ""
        if stock_data.news_sentiment and stock_data.news_sentiment['summary'] != "No recent news available.":
            news_context = f"Recent News: {stock_data.news_sentiment['summary']}. Top headlines: " + \
                           "; ".join([f"'{h['title']}' ({h['sentiment']})" for h in stock_data.news_sentiment['headlines'][:2]]) + "."

        prompt = f"""
        {opponent_name} just said:
        "{opponent_message}"
        
        Counter their argument about {stock_data.symbol} while staying true to your {self.role} perspective.
        
        Current market data for reference:
        - Price: ${stock_data.price:.2f} ({stock_data.change_pct:+.2f}%)
        - RSI: {stock_data.rsi:.1f} | MACD: {stock_data.macd:.3f}
        - Support/Resistance: ${stock_data.support:.2f}/${stock_data.resistance:.2f}
        - Bollinger Bands: Upper: ${stock_data.bb_upper:.2f}, Lower: ${stock_data.bb_lower:.2f}
        - ATR: {stock_data.atr:.2f}
        
        {news_context}

        Address their specific points but provide your alternative interpretation using data and news.
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
    
    def _call_llm(self, prompt: str) -> str:
        """
        Makes a call to the OpenAI LLM with the given prompt.
        Includes error handling for API issues.
        """
        try:
            # Assuming openai_key is set globally via Streamlit session state in app.py
            # and accessible here, or passed as an argument.
            # For modularity, it's better to pass it, but for simplicity
            # we'll assume Streamlit's session_state global access for now.
            import streamlit as st # Only import here if not globally available
            if not st.session_state.get('openai_key'):
                return "⚠️ OpenAI API key required. Please add it in the sidebar."
            
            openai.api_key = st.session_state.openai_key
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except AuthenticationError:
            return "Error: Invalid OpenAI API key. Please check your key in the sidebar."
        except RateLimitError:
            return "Error: OpenAI API rate limit exceeded. Please wait a moment and try again."
        except Exception as e:
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