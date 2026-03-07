# llm_utils.py
import openai
from openai import AuthenticationError, RateLimitError
from typing import List, Dict, Any

from data_fetcher import StockData

def _call_openai_llm(messages: List[Dict], openai_key: str, model: str = "gpt-3.5-turbo", max_tokens: int = 400, temperature: float = 0.7) -> str:
    """
    Helper function to make a call to the OpenAI LLM.
    Includes error handling for API issues.
    """
    try:
        if not openai_key:
            return "⚠️ OpenAI API key required."
        client = openai.OpenAI(api_key=openai_key)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except AuthenticationError:
        return "Error: Invalid OpenAI API key."
    except RateLimitError:
        return "Error: OpenAI API rate limit exceeded. Please wait a moment and try again."
    except Exception as e:
        return f"Error communicating with OpenAI: {str(e)}"

def generate_debate_summary(debate_history: List[Dict], stock_data: StockData, openai_key: str) -> str:
    """
    Generates a neutral summary of the debate, highlighting key bullish and bearish points.
    """
    summary_prompt_instruction = f"Summarize the key arguments from the following trading debate about {stock_data.symbol}. Highlight the main bullish and bearish points without making a recommendation. Focus on the core reasons each agent supports their stance."

    debate_text = "\n".join([
        f"Round {entry['round']} - {entry['agent']}: {entry['message']}" for entry in debate_history
    ])

    news_context = ""
    if stock_data.news_sentiment and stock_data.news_sentiment['summary'] != "No recent news available.":
        news_context = f"Recent News: {stock_data.news_sentiment['summary']}. Relevant headlines: " + \
                       "; ".join([f"'{h['title']}' ({h['sentiment']})" for h in stock_data.news_sentiment['headlines'][:2]]) + "."

    prompt = f"""
    {summary_prompt_instruction}

    Debate Transcript:
    {debate_text}

    Current Stock Data for context:
    - Price: ${stock_data.price:.2f}
    - RSI: {stock_data.rsi:.1f}
    - MACD: {stock_data.macd:.3f}
    - Support/Resistance: ${stock_data.support:.2f}/${stock_data.resistance:.2f}
    {news_context}
    """

    messages = [
        {"role": "system", "content": "You are a neutral financial analyst summarizing a trading debate."},
        {"role": "user", "content": prompt}
    ]

    return _call_openai_llm(messages, openai_key=openai_key, max_tokens=300, temperature=0.4)

def generate_final_recommendation(debate_history: List[Dict], stock_data: StockData, openai_key: str) -> str:
    """
    Generates the final trading recommendation based on the entire debate history and stock data.
    """
    debate_summary_for_recommendation = "\n".join([
        f"{entry['agent']}: {entry['message']}" for entry in debate_history
    ])

    news_context = ""
    if stock_data.news_sentiment and stock_data.news_sentiment['summary'] != "No recent news available.":
        news_context = f"Recent News Summary: {stock_data.news_sentiment['summary']}. Top headlines considered: " + \
                       "; ".join([f"'{h['title']}' ({h['sentiment']})" for h in stock_data.news_sentiment['headlines'][:3]]) + "."

    prompt = f"""
    Based on the full debate between Bull and Bear agents about {stock_data.symbol},
    and the provided current market data, provide a final, concise, and actionable day trading recommendation.

    Debate Summary:
    {debate_summary_for_recommendation}

    Current Stock Data:
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

    Provide a final recommendation with:
    1.  **Decision**: BUY/SELL/HOLD for day trading.
    2.  **Confidence**: (1-10 scale, where 10 is highest).
    3.  **Entry Price Suggestion**: (e.g., "$X.XX - $Y.YY").
    4.  **Stop-Loss Level**: (e.g., "$Z.ZZ").
    5.  **Target Price**: (e.g., "$A.AA").
    6.  **Risk Level**: (1-10 scale, where 10 is highest risk).
    7.  **Brief Reasoning**: Synthesize key points from both agents, including news sentiment, and explain the rationale for the final decision, considering pros and cons.
    8.  **Important Note**: Remind the user about market volatility and the inherent risks of day trading.

    Format as a clear, actionable recommendation, using markdown for readability.
    """

    messages = [
        {"role": "system", "content": "You are a neutral, experienced trading analyst who synthesizes different perspectives into a balanced, actionable recommendation."},
        {"role": "user", "content": prompt}
    ]

    return _call_openai_llm(messages, openai_key=openai_key, max_tokens=500, temperature=0.5)
