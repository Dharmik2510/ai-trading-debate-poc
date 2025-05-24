# app.py
import streamlit as st
import time
import json
import yfinance as yf # Keep yfinance here for direct history fetching
import pandas as pd # Keep pandas here for DataFrame operations

# Import modules
from data_fetcher import StockData, StockDataFetcher
from agents import TradingAgent, AgentPersonality, create_agents
from llm_utils import generate_debate_summary, generate_final_recommendation
from ui_utils import display_message, create_stock_chart, copy_to_clipboard_button

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
    background: green; 
    border-left: 4px solid #03ad2b;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
}

.bear-message {
    background: red; 
    border-left: 4px solid #e60017;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
}

.final-recommendation {
    background: #e0f7fa; /* Light blue */
    border: 2px solid #00bcd4;
    padding: 20px;
    margin: 20px 0;
    border-radius: 10px;
    font-weight: bold;
    color: #333;
}

.agent-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 10px;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
    cursor: pointer;
}
.stButton>button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)


# Main Streamlit App function
def main():
    st.title("🚀 AI Trading Debate Platform")
    st.markdown("Watch AI agents debate whether a stock is good for day trading!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key input
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your API key from platform.openai.com"
        )
        if openai_key:
            st.session_state.openai_key = openai_key
            st.success("API key configured!")
        else:
            st.warning("Please enter your OpenAI API key to start the debate.")
        
        st.divider()
        
        # Stock symbol input
        stock_symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            max_chars=5, # Limit input length for typical tickers
            help="Enter stock ticker (e.g., AAPL, TSLA, GOOGL)"
        ).upper()
        
        # Debate settings
        max_rounds = st.slider("Max Debate Rounds", 1, 5, 3, help="More rounds allow for deeper counter-arguments and analysis.")
        
        # Chart historical period selection
        chart_period_options = {
            "1 Day": "1d", # Added 1 Day for very short-term view
            "5 Days": "5d",
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "5 Years": "5y"
        }
        selected_chart_period_display = st.selectbox(
            "Chart Historical Period",
            list(chart_period_options.keys()),
            index=2, # Default to 1 Month
            help="Select the historical period for the stock chart."
        )
        selected_chart_period = chart_period_options[selected_chart_period_display]

        st.divider()
        st.subheader("🧑‍💻 Customize Agent Personalities")

        # Default personalities for reference/reset
        default_bull_personality = AgentPersonality(
            risk_tolerance="Medium-High",
            focus_areas=["Breakouts", "Momentum", "Volume Spikes", "Positive News Catalysts", "Support Levels"],
            style="Opportunistic growth seeker, quick to capitalize on upward trends.",
            beliefs="The market generally trends higher, and pullbacks are buying opportunities."
        )
        default_bear_personality = AgentPersonality(
            risk_tolerance="Low-Medium",
            focus_areas=["Risk Factors", "Overbought Signals (RSI > 70)", "Resistance Levels", "Negative News", "Volume Declines"],
            style="Defensive risk manager, identifies potential reversals and shorting opportunities.",
            beliefs="Markets are prone to corrections, and caution is paramount to preserve capital."
        )

        with st.expander("Agent Bull Personality"):
            bull_risk_tolerance = st.selectbox("Bull Risk Tolerance", ["Low", "Medium", "Medium-High", "High"], index=2, key="bull_risk")
            bull_focus_areas_str = st.text_area("Bull Focus Areas (comma-separated)", ", ".join(default_bull_personality.focus_areas), key="bull_focus")
            bull_style = st.text_area("Bull Trading Style", default_bull_personality.style, key="bull_style")
            bull_beliefs = st.text_area("Bull Key Beliefs", default_bull_personality.beliefs, key="bull_beliefs")
            
            bull_personality = AgentPersonality(
                risk_tolerance=bull_risk_tolerance,
                focus_areas=[f.strip() for f in bull_focus_areas_str.split(",")] if bull_focus_areas_str else [],
                style=bull_style,
                beliefs=bull_beliefs
            )

        with st.expander("Agent Bear Personality"):
            bear_risk_tolerance = st.selectbox("Bear Risk Tolerance", ["Low", "Medium", "Medium-High", "High"], index=1, key="bear_risk")
            bear_focus_areas_str = st.text_area("Bear Focus Areas (comma-separated)", ", ".join(default_bear_personality.focus_areas), key="bear_focus")
            bear_style = st.text_area("Bear Trading Style", default_bear_personality.style, key="bear_style")
            bear_beliefs = st.text_area("Bear Key Beliefs", default_bear_personality.beliefs, key="bear_beliefs")
            
            bear_personality = AgentPersonality(
                risk_tolerance=bear_risk_tolerance,
                focus_areas=[f.strip() for f in bear_focus_areas_str.split(",")] if bear_focus_areas_str else [],
                style=bear_style,
                beliefs=bear_beliefs
            )

        st.divider()
        start_debate = st.button("🎯 Start Debate", type="primary", use_container_width=True)
    
    # Initialize session state variables if they don't exist
    if 'debate_history' not in st.session_state:
        st.session_state.debate_history = []
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 0
    if 'debate_active' not in st.session_state:
        st.session_state.debate_active = False
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'final_recommendation_text' not in st.session_state:
        st.session_state.final_recommendation_text = ""
    if 'chart_historical_data' not in st.session_state:
        st.session_state.chart_historical_data = None

    # Main content area layout
    col1, col2 = st.columns([2, 1]) # Column for debate, Column for chart/tech data
    
    with col1:
        if start_debate:
            # Check for API key before proceeding
            if not openai_key:
                st.error("Please provide your OpenAI API key in the sidebar to start the debate.")
                st.stop()

            # Reset session state for a new debate
            st.session_state.debate_history = []
            st.session_state.current_round = 0
            st.session_state.debate_active = True
            st.session_state.stock_data = None # Clear previous data
            st.session_state.final_recommendation_text = "" # Clear previous recommendation
            st.session_state.chart_historical_data = None # Clear previous chart data
            
            # Fetch primary stock data with technicals and news sentiment
            with st.spinner(f"Fetching data for {stock_symbol}..."):
                # Fetch a longer period (e.g., "1y") to ensure enough data for MAs, BBs, ATR calculations
                stock_data = StockDataFetcher.get_stock_data(stock_symbol, period="1y") 
                
            if not stock_data:
                st.error(f"Could not fetch data for {stock_symbol}. Please check the symbol and try again.")
                st.session_state.debate_active = False # Stop debate if data fetch fails
                st.stop() # Halt execution if data is not available
            
            st.session_state.stock_data = stock_data # Store fetched data in session state
            
            # Fetch historical data specifically for charting (uses user-selected period)
            with st.spinner(f"Fetching historical chart data for {stock_symbol} ({selected_chart_period})..."):
                try:
                    # yfinance.history() directly returns a pandas DataFrame
                    chart_hist_data = yf.Ticker(stock_symbol).history(period=selected_chart_period)
                    if not chart_hist_data.empty:
                        st.session_state.chart_historical_data = chart_hist_data
                    else:
                        st.warning(f"No historical chart data available for {stock_symbol} for the period {selected_chart_period}.")
                        st.session_state.chart_historical_data = pd.DataFrame() # Ensure it's an empty DataFrame
                except Exception as e:
                    st.error(f"Error fetching historical chart data for {stock_symbol}: {e}")
                    st.session_state.chart_historical_data = pd.DataFrame() # Ensure it's an empty DataFrame


            # Display current stock metrics
            st.subheader(f"📊 {stock_data.symbol} Analysis")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
            with metrics_col1:
                st.metric("Price", f"${stock_data.price:.2f}", f"{stock_data.change_pct:+.2f}%")
            with metrics_col2:
                st.metric("RSI", f"{stock_data.rsi:.1f}")
            with metrics_col3:
                st.metric("MACD", f"{stock_data.macd:.3f}")
            with metrics_col4:
                st.metric("Support", f"${stock_data.support:.2f}")
            with metrics_col5:
                st.metric("Resistance", f"${stock_data.resistance:.2f}")

            # Display news sentiment
            st.markdown("---")
            st.subheader("📰 Recent News Sentiment")
            if stock_data.news_sentiment and stock_data.news_sentiment['summary']:
                st.write(stock_data.news_sentiment['summary'])
                if stock_data.news_sentiment['headlines']:
                    for i, headline in enumerate(stock_data.news_sentiment['headlines']):
                        # Ensure title and publisher are not empty before displaying
                        title_display = headline['title'] if headline['title'] else "N/A Title"
                        publisher_display = headline['publisher'] if headline['publisher'] else "N/A Publisher"
                        st.markdown(f"- **{headline['sentiment'].capitalize()}**: [{title_display}]({headline['link']}) ({publisher_display})")
            else:
                st.write("No news sentiment data available.")

            st.markdown("---") # Visual separator

            # Initialize trading agents with user-defined personalities
            bull_agent, bear_agent = create_agents(bull_personality, bear_personality)
            
            st.subheader("🥊 Live Debate")
            
            # Progress bar for the debate
            debate_progress_bar = st.progress(0)
            
            # Debate rounds container
            debate_container = st.container()
            with debate_container:
                # Round 1: Initial positions
                st.session_state.current_round = 1
                debate_progress_bar.progress(st.session_state.current_round / (max_rounds + 1))
                with st.expander(f"Round {st.session_state.current_round}: Initial Analysis", expanded=True):
                    
                    with st.spinner("Agent Bull is analyzing..."):
                        bull_initial, bull_sentiment = bull_agent.analyze(stock_data)
                        time.sleep(0.5)  # Simulate thinking time
                    
                    display_message({"name": bull_agent.name, "avatar": bull_agent.avatar, "color": bull_agent.color}, bull_initial, bull_sentiment)
                    st.session_state.debate_history.append({
                        "round": st.session_state.current_round,
                        "agent": bull_agent.name,
                        "message": bull_initial,
                        "sentiment": bull_sentiment
                    })
                    
                    with st.spinner("Agent Bear is analyzing..."):
                        bear_initial, bear_sentiment = bear_agent.analyze(stock_data)
                        time.sleep(0.5)
                    
                    display_message({"name": bear_agent.name, "avatar": bear_agent.avatar, "color": bear_agent.color}, bear_initial, bear_sentiment)
                    st.session_state.debate_history.append({
                        "round": st.session_state.current_round,
                        "agent": bear_agent.name,
                        "message": bear_initial,
                        "sentiment": bear_sentiment
                    })
                
                # Subsequent rounds: agents counter-argue
                last_bull_message = bull_initial
                last_bear_message = bear_initial
                
                for round_num in range(2, max_rounds + 1):
                    st.session_state.current_round = round_num
                    debate_progress_bar.progress(st.session_state.current_round / (max_rounds + 1))
                    with st.expander(f"Round {st.session_state.current_round}: Counter-Arguments", expanded=True):
                        # Bull responds to Bear
                        with st.spinner(f"Agent Bull is preparing counter-argument (Round {st.session_state.current_round})..."):
                            bull_counter, bull_sentiment = bull_agent.respond_to(bear_agent.name, last_bear_message, stock_data)
                            time.sleep(0.5)
                        
                        display_message({"name": bull_agent.name, "avatar": bull_agent.avatar, "color": bull_agent.color}, bull_counter, bull_sentiment)
                        st.session_state.debate_history.append({
                            "round": st.session_state.current_round,
                            "agent": bull_agent.name,
                            "message": bull_counter,
                            "sentiment": bull_sentiment
                        })
                        
                        # Bear responds to Bull  
                        with st.spinner(f"Agent Bear is preparing counter-argument (Round {st.session_state.current_round})..."):
                            bear_counter, bear_sentiment = bear_agent.respond_to(bull_agent.name, last_bull_message, stock_data)
                            time.sleep(0.5)
                        
                        display_message({"name": bear_agent.name, "avatar": bear_agent.avatar, "color": bear_agent.color}, bear_counter, bear_sentiment)
                        st.session_state.debate_history.append({
                            "round": st.session_state.current_round,
                            "agent": bear_agent.name,
                            "message": bear_counter,
                            "sentiment": bear_sentiment
                        })
                        
                        last_bull_message = bull_counter
                        last_bear_message = bear_counter
                
                debate_progress_bar.progress(1.0) # Complete progress bar
                st.success("Debate Concluded!")

                st.markdown("---")
                st.subheader("⚖️ Debate Summary")
                with st.spinner("Generating debate summary..."):
                    summary = generate_debate_summary(st.session_state.debate_history, stock_data)
                    st.markdown(summary)
                
                st.markdown("---")
                st.subheader("🎯 Final Recommendation")
                
                with st.spinner("Generating final recommendation..."):
                    final_rec = generate_final_recommendation(st.session_state.debate_history, stock_data)
                    st.session_state.final_recommendation_text = final_rec # Store for re-runs
                
                st.markdown(f"""
                <div class="final-recommendation">
                    {st.session_state.final_recommendation_text}
                </div>
                """, unsafe_allow_html=True)
                
                # Add copy to clipboard button for the recommendation
                if st.session_state.final_recommendation_text:
                    copy_to_clipboard_button(st.session_state.final_recommendation_text)

        # This block handles re-runs of the app if a debate was already active
        elif st.session_state.debate_active and st.session_state.stock_data:
            st.subheader(f"📊 {st.session_state.stock_data.symbol} Analysis")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
            with metrics_col1:
                st.metric("Price", f"${st.session_state.stock_data.price:.2f}", f"{st.session_state.stock_data.change_pct:+.2f}%")
            with metrics_col2:
                st.metric("RSI", f"{st.session_state.stock_data.rsi:.1f}")
            with metrics_col3:
                st.metric("MACD", f"{st.session_state.stock_data.macd:.3f}")
            with metrics_col4:
                st.metric("Support", f"${st.session_state.stock_data.support:.2f}")
            with metrics_col5:
                st.metric("Resistance", f"${st.session_state.stock_data.resistance:.2f}")
            
            st.markdown("---")
            st.subheader("📰 Recent News Sentiment")
            if st.session_state.stock_data.news_sentiment and st.session_state.stock_data.news_sentiment['summary']:
                st.write(st.session_state.stock_data.news_sentiment['summary'])
                if st.session_state.stock_data.news_sentiment['headlines']:
                    for i, headline in enumerate(st.session_state.stock_data.news_sentiment['headlines']):
                        title_display = headline['title'] if headline['title'] else "N/A Title"
                        publisher_display = headline['publisher'] if headline['publisher'] else "N/A Publisher"
                        st.markdown(f"- **{headline['sentiment'].capitalize()}**: [{title_display}]({headline['link']}) ({publisher_display})")
            else:
                st.write("No news sentiment data available.")

            st.markdown("---")
            st.subheader("🥊 Live Debate")
            st.progress(1.0) # Show full progress if debate is already concluded from a previous run
            
            with st.container():
                for entry in st.session_state.debate_history:
                    agent_name = entry['agent']
                    message = entry['message']
                    sentiment = entry.get('sentiment', 'neutral') # Use .get for robustness
                    
                    # Re-create minimal agent_info dict to pass to display_message
                    agent_info_for_display = {}
                    if "Bull" in agent_name:
                        agent_info_for_display = {"name": "Agent Bull 🐂", "avatar": "🐂", "color": "#03ad2b"}
                    else:
                        agent_info_for_display = {"name": "Agent Bear 🐻", "avatar": "🐻", "color": "#e60017"}
                    
                    with st.expander(f"Round {entry['round']} - {agent_name}", expanded=False):
                        display_message(agent_info_for_display, message, sentiment)
            
            st.markdown("---")
            st.subheader("⚖️ Debate Summary")
            with st.spinner("Generating debate summary..."):
                summary = generate_debate_summary(st.session_state.debate_history, st.session_state.stock_data)
                st.markdown(summary)

            st.markdown("---")
            st.subheader("🎯 Final Recommendation")
            st.markdown(f"""
            <div class="final-recommendation">
                {st.session_state.final_recommendation_text}
            </div>
            """, unsafe_allow_html=True)
            if st.session_state.final_recommendation_text:
                copy_to_clipboard_button(st.session_state.final_recommendation_text)


    # Right column for chart and technical data
    with col2:
        if st.session_state.stock_data and st.session_state.chart_historical_data is not None:
            st.subheader("📈 Chart")
            
            # Pass the selected chart period and historical data to the chart function
            chart = create_stock_chart(st.session_state.stock_data.symbol, st.session_state.chart_historical_data, selected_chart_period)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info(f"Chart could not be generated for {st.session_state.stock_data.symbol} for the selected period ({selected_chart_period_display}). Please try a different period.")
            
            # Technical indicators display
            st.subheader("📊 Technical Data")
            st.json({
                "RSI": round(st.session_state.stock_data.rsi, 2),
                "MACD": round(st.session_state.stock_data.macd, 4),
                "20-day MA": round(st.session_state.stock_data.ma_20, 2),
                "50-day MA": round(st.session_state.stock_data.ma_50, 2),
                "Bollinger Upper": round(st.session_state.stock_data.bb_upper, 2),
                "Bollinger Lower": round(st.session_state.stock_data.bb_lower, 2),
                "ATR": round(st.session_state.stock_data.atr, 2),
                "Volume": f"{st.session_state.stock_data.volume:,}"
            })
    
    # Footer with disclaimer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        🤖 Powered by OpenAI GPT-3.5 | 📈 Market data from Yahoo Finance<br>
        <span style='color: red; font-weight: bold;'>Disclaimer: This is for educational and entertainment purposes only and not financial advice. Day trading involves substantial risk.</span>
    </div>
    """, unsafe_allow_html=True)

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()