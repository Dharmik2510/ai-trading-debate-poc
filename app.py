import streamlit as st
import openai
import yfinance as yf
import pandas as pd
import time
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="AI Trading Debate",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.debate-container {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    background-color: #f8f9fa;
}

.bull-message {
    background: #03ad2b;
    border-left: 4px solid #28a745;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
}

.bear-message {
    background: #e60017;
    border-left: 4px solid #dc3545;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
}

.final-recommendation {
    background: blue;
    border: 2px solid #17a2b8;
    padding: 20px;
    margin: 20px 0;
    border-radius: 10px;
    font-weight: bold;
}

.agent-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

@dataclass
class StockData:
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

class TradingAgent:
    def __init__(self, name: str, role: str, avatar: str, color: str, personality: Dict):
        self.name = name
        self.role = role
        self.avatar = avatar
        self.color = color
        self.personality = personality
        self.conversation_memory = []
        
    def get_system_prompt(self) -> str:
        return f"""
        You are {self.name}, an expert day trader with a {self.role} perspective.
        
        Your personality:
        - Risk tolerance: {self.personality['risk_tolerance']}
        - Focus areas: {', '.join(self.personality['focus_areas'])}
        - Trading style: {self.personality['style']}
        
        Rules for debate:
        1. Stay in character - maintain your {self.role} perspective
        2. Use specific data points to support arguments
        3. Address opponent's points directly
        4. Provide actionable trading insights
        5. Be professional but assertive
        6. Keep responses concise (2-3 paragraphs max)
        
        Always structure responses with:
        - Key point/counterpoint
        - Supporting evidence from data
        - Specific trading action (if applicable)
        """
    
    def analyze(self, stock_data: StockData, context: str = "") -> str:
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
        
        Context: {context}
        
        Should we day trade {stock_data.symbol} today? Give your {self.role} analysis.
        """
        
        return self._call_llm(prompt)
    
    def respond_to(self, opponent_name: str, opponent_message: str, stock_data: StockData) -> str:
        prompt = f"""
        {opponent_name} just said:
        "{opponent_message}"
        
        Counter their argument about {stock_data.symbol} while staying true to your {self.role} perspective.
        
        Current market data for reference:
        - Price: ${stock_data.price:.2f} ({stock_data.change_pct:+.2f}%)
        - RSI: {stock_data.rsi:.1f} | MACD: {stock_data.macd:.3f}
        - Support/Resistance: ${stock_data.support:.2f}/${stock_data.resistance:.2f}
        
        Address their specific points but provide your alternative interpretation.
        """
        
        response = self._call_llm(prompt)
        
        # Store in memory
        self.conversation_memory.append({
            "opponent": opponent_name,
            "opponent_said": opponent_message,
            "my_response": response
        })
        
        return response
    
    def _call_llm(self, prompt: str) -> str:
        try:
            if not st.session_state.get('openai_key'):
                return "⚠️ OpenAI API key required. Please add it in the sidebar."
            
            openai.api_key = st.session_state.openai_key
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

class StockDataFetcher:
    @staticmethod
    def get_stock_data(symbol: str) -> StockData:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="3mo")
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            change_pct = ((current_price - prev_price) / prev_price) * 100
            volume = hist['Volume'].iloc[-1]
            
            # Technical indicators
            rsi = StockDataFetcher._calculate_rsi(hist['Close'])
            macd = StockDataFetcher._calculate_macd(hist['Close'])
            ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            support = hist['Low'].rolling(20).min().iloc[-1]
            resistance = hist['High'].rolling(20).max().iloc[-1]
            
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
                resistance=float(resistance)
            )
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    @staticmethod
    def _calculate_macd(prices):
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd.iloc[-1]

def create_agents():
    bull_agent = TradingAgent(
        name="Agent Bull 🐂",
        role="bullish",
        avatar="🐂",
        color="#03ad2b",
        personality={
            "risk_tolerance": "Medium-High",
            "focus_areas": ["Breakouts", "Momentum", "Volume", "Catalysts"],
            "style": "Opportunistic growth seeker"
        }
    )
    
    bear_agent = TradingAgent(
        name="Agent Bear 🐻",
        role="bearish", 
        avatar="🐻",
        color="#e60017",
        personality={
            "risk_tolerance": "Low-Medium",
            "focus_areas": ["Risk factors", "Overbought signals", "Market warnings"],
            "style": "Defensive risk manager"
        }
    )
    
    return bull_agent, bear_agent

def display_message(agent: TradingAgent, message: str, message_type: str = ""):
    css_class = "bull-message" if "Bull" in agent.name else "bear-message"
    
    st.markdown(f"""
    <div class="{css_class}">
        <strong>{agent.avatar} {agent.name}</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def create_stock_chart(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1mo")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(x=hist.index, y=hist['Volume'], name="Volume", marker_color='lightblue'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{symbol} - Last 30 Days",
            xaxis_rangeslider_visible=False,
            height=400
        )
        
        return fig
    except:
        return None

def generate_final_recommendation(debate_history: List, stock_data: StockData) -> str:
    if not st.session_state.get('openai_key'):
        return "⚠️ OpenAI API key required for final recommendation."
    
    try:
        openai.api_key = st.session_state.openai_key
        
        # Compile debate history
        debate_summary = "\n".join([
            f"{entry['agent']}: {entry['message']}" for entry in debate_history
        ])
        
        prompt = f"""
        Based on the debate between Bull and Bear agents about {stock_data.symbol}, 
        provide a final trading recommendation.
        
        Debate Summary:
        {debate_summary}
        
        Current Stock Data:
        - Price: ${stock_data.price:.2f} ({stock_data.change_pct:+.2f}%)
        - RSI: {stock_data.rsi:.1f}
        - MACD: {stock_data.macd:.3f}
        - Support/Resistance: ${stock_data.support:.2f}/${stock_data.resistance:.2f}
        
        Provide a final recommendation with:
        1. Decision: BUY/SELL/HOLD for day trading
        2. Confidence: 1-10 scale
        3. Entry price suggestion
        4. Stop-loss level
        5. Target price
        6. Risk level: 1-10 scale
        7. Brief reasoning that considers both perspectives
        
        Format as a clear, actionable recommendation.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a neutral trading analyst who synthesizes different perspectives into actionable recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.5
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating recommendation: {str(e)}"

# Main Streamlit App
def main():
    st.title("🚀 AI Trading Debate Platform")
    st.markdown("Watch AI agents debate whether a stock is good for day trading!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your API key from platform.openai.com"
        )
        if openai_key:
            st.session_state.openai_key = openai_key
            st.success("API key configured!")
        
        st.divider()
        
        # Stock input
        stock_symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter stock ticker (e.g., AAPL, TSLA, GOOGL)"
        ).upper()
        
        # Debate settings
        max_rounds = st.slider("Max Debate Rounds", 1, 5, 3)
        
        # Start debate button
        start_debate = st.button("🎯 Start Debate", type="primary", use_container_width=True)
    
    # Initialize session state
    if 'debate_history' not in st.session_state:
        st.session_state.debate_history = []
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 0
    if 'debate_active' not in st.session_state:
        st.session_state.debate_active = False
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if start_debate and openai_key:
            # Reset for new debate
            st.session_state.debate_history = []
            st.session_state.current_round = 0
            st.session_state.debate_active = True
            
            # Fetch stock data
            with st.spinner(f"Fetching data for {stock_symbol}..."):
                stock_data = StockDataFetcher.get_stock_data(stock_symbol)
                
            if not stock_data:
                st.error("Could not fetch stock data. Please check the symbol.")
                st.stop()
            
            st.session_state.stock_data = stock_data
            
            # Display stock info
            st.subheader(f"📊 {stock_data.symbol} Analysis")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("Price", f"${stock_data.price:.2f}", f"{stock_data.change_pct:+.2f}%")
            with metrics_col2:
                st.metric("RSI", f"{stock_data.rsi:.1f}")
            with metrics_col3:
                st.metric("Support", f"${stock_data.support:.2f}")
            with metrics_col4:
                st.metric("Resistance", f"${stock_data.resistance:.2f}")
            
            # Initialize agents
            bull_agent, bear_agent = create_agents()
            
            # Start the debate
            st.subheader("🥊 Live Debate")
            
            debate_container = st.container()
            
            with debate_container:
                # Round 1: Initial positions
                st.markdown("### Round 1: Initial Analysis")
                
                with st.spinner("Agent Bull is analyzing..."):
                    bull_initial = bull_agent.analyze(stock_data)
                    time.sleep(1)  # Simulate thinking time
                
                display_message(bull_agent, bull_initial)
                st.session_state.debate_history.append({
                    "round": 1,
                    "agent": bull_agent.name,
                    "message": bull_initial
                })
                
                with st.spinner("Agent Bear is analyzing..."):
                    bear_initial = bear_agent.analyze(stock_data)
                    time.sleep(1)
                
                display_message(bear_agent, bear_initial)
                st.session_state.debate_history.append({
                    "round": 1,
                    "agent": bear_agent.name,
                    "message": bear_initial
                })
                
                # Additional rounds
                last_bull_message = bull_initial
                last_bear_message = bear_initial
                
                for round_num in range(2, max_rounds + 1):
                    st.markdown(f"### Round {round_num}: Counter-Arguments")
                    
                    # Bull responds to Bear
                    with st.spinner(f"Agent Bull is preparing counter-argument..."):
                        bull_counter = bull_agent.respond_to(bear_agent.name, last_bear_message, stock_data)
                        time.sleep(1)
                    
                    display_message(bull_agent, bull_counter)
                    st.session_state.debate_history.append({
                        "round": round_num,
                        "agent": bull_agent.name,
                        "message": bull_counter
                    })
                    
                    # Bear responds to Bull  
                    with st.spinner(f"Agent Bear is preparing counter-argument..."):
                        bear_counter = bear_agent.respond_to(bull_agent.name, last_bull_message, stock_data)
                        time.sleep(1)
                    
                    display_message(bear_agent, bear_counter)
                    st.session_state.debate_history.append({
                        "round": round_num,
                        "agent": bear_agent.name,
                        "message": bear_counter
                    })
                    
                    last_bull_message = bull_counter
                    last_bear_message = bear_counter
                
                # Final recommendation
                st.markdown("### 🎯 Final Recommendation")
                
                with st.spinner("Generating final recommendation..."):
                    final_rec = generate_final_recommendation(st.session_state.debate_history, stock_data)
                    time.sleep(1)
                
                st.markdown(f"""
                <div class="final-recommendation">
                    <h4>⚖️ Consensus Decision</h4>
                    {final_rec}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.stock_data:
            st.subheader("📈 Chart")
            
            chart = create_stock_chart(st.session_state.stock_data.symbol)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Technical indicators
            st.subheader("📊 Technical Data")
            st.json({
                "RSI": round(st.session_state.stock_data.rsi, 2),
                "MACD": round(st.session_state.stock_data.macd, 4),
                "20-day MA": round(st.session_state.stock_data.ma_20, 2),
                "50-day MA": round(st.session_state.stock_data.ma_50, 2),
                "Volume": f"{st.session_state.stock_data.volume:,}"
            })
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🤖 Powered by OpenAI GPT-3.5 | 📈 Market data from Yahoo Finance
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()