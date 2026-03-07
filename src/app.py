import streamlit as st
import time
import json
import yfinance as yf # Keep yfinance here for direct history fetching
import pandas as pd # Keep pandas here for DataFrame operations

# Import modules
from data_fetcher import StockData, StockDataFetcher
from agents import TradingAgent, AgentPersonality, create_agents
from llm_utils import generate_debate_summary, generate_final_recommendation
from ui_utils import display_message, copy_to_clipboard_button
from crewai import Crew, Task
import crew_agentic_workflow

# Page config
st.set_page_config(
    page_title="AI Trading Debate",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
# Custom CSS for better UI - FIXED VERSION
st.markdown("""
<style>
/* Base UI Settings */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphism General Container */
.debate-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 24px;
    margin: 15px 0;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
}

/* Base Message Card Settings */
.base-message {
    padding: 24px;
    margin: 20px 0;
    border-radius: 16px;
    word-wrap: break-word;
    word-break: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    box-sizing: border-box;
    width: 100%;
    font-size: 15px;
    line-height: 1.7;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    color: #f1f1f1; /* Assumes Dark Mode preference for trading apps */
}

/* Hover Animation for Cards */
.base-message:hover {
    transform: translateY(-5px) scale(1.01);
}

/* Bull Message Theme (Neon Green/Teal Glass) */
.bull-message {
    background: linear-gradient(135deg, rgba(3, 173, 43, 0.15) 0%, rgba(0, 150, 136, 0.05) 100%);
    border-left: 4px solid #00E676;
    border-right: 1px solid rgba(0, 230, 118, 0.2);
    border-top: 1px solid rgba(0, 230, 118, 0.2);
    border-bottom: 1px solid rgba(0, 230, 118, 0.2);
    box-shadow: 0 10px 30px rgba(0, 230, 118, 0.1), inset 0 0 20px rgba(0, 230, 118, 0.05);
}

.bull-message:hover {
    box-shadow: 0 15px 35px rgba(0, 230, 118, 0.2), inset 0 0 20px rgba(0, 230, 118, 0.1);
}

/* Bear Message Theme (Neon Red/Orange Glass) */
.bear-message {
    background: linear-gradient(135deg, rgba(230, 0, 23, 0.15) 0%, rgba(255, 87, 34, 0.05) 100%);
    border-left: 4px solid #FF1744;
    border-right: 1px solid rgba(255, 23, 68, 0.2);
    border-top: 1px solid rgba(255, 23, 68, 0.2);
    border-bottom: 1px solid rgba(255, 23, 68, 0.2);
    box-shadow: 0 10px 30px rgba(255, 23, 68, 0.1), inset 0 0 20px rgba(255, 23, 68, 0.05);
}

.bear-message:hover {
    box-shadow: 0 15px 35px rgba(255, 23, 68, 0.2), inset 0 0 20px rgba(255, 23, 68, 0.1);
}

/* Final Recommendation Theme (Gold/Purple Premium Glass) */
.final-recommendation {
    background: linear-gradient(135deg, rgba(138, 43, 226, 0.15) 0%, rgba(255, 215, 0, 0.1) 100%);
    border: 1px solid rgba(255, 215, 0, 0.4);
    padding: 30px;
    margin: 30px 0;
    border-radius: 20px;
    font-weight: 500;
    color: #fff;
    font-size: 16px;
    box-shadow: 0 15px 35px rgba(138, 43, 226, 0.2), inset 0 0 30px rgba(255, 215, 0, 0.1);
    word-wrap: break-word;
    word-break: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

/* Shimmer Effect on Final Recommendation */
.final-recommendation::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 50%;
    height: 100%;
    background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0) 100%);
    transform: skewX(-25deg);
    animation: shimmer 6s infinite;
}

@keyframes shimmer {
    0% { left: -100%; }
    20% { left: 200%; }
    100% { left: 200%; }
}

/* Metrics Styling overhaul */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: -webkit-linear-gradient(#fff, #b3b3b3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Primary Button Styling */
.stButton>button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3) !important;
    text-transform: uppercase;
}

.stButton>button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 25px rgba(99, 102, 241, 0.4) !important;
}

.stButton>button:active {
    transform: translateY(1px) !important;
}

/* Expander Overrides */
.streamlit-expanderHeader {
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    background: rgba(255, 255, 255, 0.03) !important;
}

/* Ensure code blocks look premium */
code, pre {
    background: #1e1e24 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e0e0e0 !important;
}

/* Agent Avatar Pulsing Animation */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(255, 255, 255, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }
}

.agent-avatar {
    width: 45px;
    height: 45px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-right: 15px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    animation: pulse 2s infinite;
}

/* Clean up footer disclaimer */
.footer-disclaimer {
    text-align: center;
    color: #888 !important;
    font-size: 0.9rem !important;
    padding: 24px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.footer-disclaimer .disclaimer-text {
    color: #ff5252 !important;
    font-weight: 600 !important;
    margin-top: 10px !important;
    opacity: 0.8;
}

/* Headers */
h1, h2, h3 {
    background: -webkit-linear-gradient(45deg, #ffffff, #a8a8a8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
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
            help="Get your API key from platform.openai.com. This key will be used for both agent debates (via OpenAI API) and news sentiment analysis (via OpenAI API)."
        )
        if openai_key:
            st.session_state.openai_key = openai_key
            st.success("API key configured!")
        else:
            st.warning("Please enter your OpenAI API key to start the debate.")
        
        st.divider()

         # Reddit API Configuration (Optional)
        st.subheader("🤖 Reddit Analysis (Optional)")
        st.markdown("*Leave blank to skip Reddit analysis*")
        
        reddit_client_id = st.text_input(
            "Reddit Client ID",
            type="password",
            help="Get Reddit API credentials from https://www.reddit.com/prefs/apps"
        )
        
        reddit_client_secret = st.text_input(
            "Reddit Client Secret", 
            type="password",
            help="Your Reddit API client secret"
        )
        
        if reddit_client_id and reddit_client_secret:
            st.session_state.reddit_credentials = {
                'client_id': reddit_client_id,
                'client_secret': reddit_client_secret
            }
            st.success("Reddit API configured!")
        else:
            st.session_state.reddit_credentials = None
            st.info("Reddit analysis will be skipped without API credentials")
        
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
                # Pass both API key and Reddit credentials to the data fetcher
                reddit_creds = getattr(st.session_state, 'reddit_credentials', None)
                stock_data = StockDataFetcher.get_stock_data(
                    stock_symbol, 
                    period="1y", 
                    api_key=openai_key,
                    reddit_credentials=reddit_creds
                )
                
            if not stock_data:
                st.error(f"Could not fetch data for {stock_symbol}. Please check the symbol and try again.")
                st.session_state.debate_active = False # Stop debate if data fetch fails
                st.stop() # Halt execution if data is not available
            
            st.session_state.stock_data = stock_data # Store fetched data in session state
            
            # Fetch historical data specifically for charting (uses user-selected period)
            with st.spinner(f"Fetching historical chart data for {stock_symbol} ({selected_chart_period_display})..."):
                try:
                    # yfinance.history() directly returns a pandas DataFrame
                    chart_hist_data = yf.Ticker(stock_symbol).history(period=selected_chart_period)
                    if not chart_hist_data.empty:
                        st.session_state.chart_historical_data = chart_hist_data
                    else:
                        st.warning(f"No historical chart data available for {stock_symbol} for the period {selected_chart_period_display}.")
                        st.session_state.chart_historical_data = pd.DataFrame() # Ensure it's an empty DataFrame
                except Exception as e:
                    st.error(f"Error fetching historical chart data for {stock_symbol}: {e}")
                    st.session_state.chart_historical_data = pd.DataFrame() # Ensure it's an empty DataFrame


            # Display current stock metrics
            st.subheader(f"� {stock_data.symbol} Analysis")
            
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
            # Add this section after the Yahoo Finance news sentiment display in your app.py

            # Display news sentiment
            st.markdown("---")
            st.subheader("📰 Recent News Sentiment")
            if stock_data.news_sentiment and stock_data.news_sentiment['summary']:
                st.write(stock_data.news_sentiment['summary'])
                if stock_data.news_sentiment['headlines']:
                    for i, headline in enumerate(stock_data.news_sentiment['headlines']):
                        # If headline is a dict, get 'title' and 'publisher'; if string, use as is
                        if isinstance(headline, dict):
                            title_display = headline.get('title', 'No title available')
                            publisher_display = headline.get('publisher', 'N/A Publisher')
                            sentiment_display = headline.get('sentiment', 'Neutral').capitalize()
                            link_display = headline.get('link', None)
                        else:
                            title_display = headline if headline else 'No title available'
                            publisher_display = 'N/A Publisher'
                            sentiment_display = 'Neutral'
                            link_display = None
                        if link_display:
                            st.markdown(f"- **{sentiment_display}**: [{title_display}]({link_display}) ({publisher_display})")
                        else:
                            st.markdown(f"- **{sentiment_display}**: {title_display} ({publisher_display})")
            else:
                st.write("No news sentiment data available.")

            # ADD THIS NEW SECTION FOR REDDIT ANALYSIS
            st.markdown("---")
            st.subheader("🤖 Reddit Community Sentiment")
            if stock_data.reddit_sentiment and stock_data.reddit_sentiment['summary']:
                st.write(stock_data.reddit_sentiment['summary'])
                
                # Display sentiment breakdown
                sentiment_breakdown = stock_data.reddit_sentiment.get('sentiment_breakdown', {})
                if sentiment_breakdown:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🟢 Bullish", sentiment_breakdown.get('bullish', 0))
                    with col2:
                        st.metric("🔴 Bearish", sentiment_breakdown.get('bearish', 0))
                    with col3:
                        st.metric("⚪ Neutral", sentiment_breakdown.get('neutral', 0))
                
                # Display top discussions
                discussions = stock_data.reddit_sentiment.get('discussions', [])
                if discussions:
                    st.write("**Top Reddit Discussions:**")
                    for i, discussion in enumerate(discussions[:5]):  # Show top 5
                        sentiment_emoji = "🟢" if discussion['sentiment'] == 'bullish' else "🔴" if discussion['sentiment'] == 'bearish' else "⚪"
                        
                        with st.expander(f"{sentiment_emoji} r/{discussion['subreddit']} - {discussion['title'][:60]}..." if len(discussion['title']) > 60 else f"{sentiment_emoji} r/{discussion['subreddit']} - {discussion['title']}"):
                            st.write(f"**Score:** {discussion['score']} upvotes")
                            st.write(f"**Sentiment:** {discussion['sentiment'].capitalize()}")
                            st.write(f"**Preview:** {discussion['content_preview']}")
                            if discussion.get('url'):
                                st.markdown(f"[View on Reddit]({discussion['url']})")
            else:
                st.write("No Reddit sentiment data available.")

            st.markdown("---") # Visual separator

            # ===============================================
            # AGENTIC WORKFLOW POWERED BY CREWAI
            # ===============================================
            st.subheader("🤖 Live Agentic Debate")
            
            # Setup agent container
            workflow_container = st.container()
            with workflow_container:
                with st.spinner("Initializing CrewAI Agents and Tools..."):
                    # Create the agents
                    researcher, bull_agent, bear_agent, judge_agent = crew_agentic_workflow.create_trading_agents(
                        bull_focus=", ".join(bull_personality.focus_areas),
                        bear_focus=", ".join(bear_personality.focus_areas),
                        api_key=openai_key
                    )
                
                # We will keep the history in session state
                st.session_state.debate_history = []
                
                # --- Step 1: Research ---
                with st.spinner(f"🕵️ Researcher is gathering comprehensive data for {stock_symbol}..."):
                    research_task = Task(
                        description=f"""Use your tools to gather all available technical data, recent news, and Reddit sentiment for the stock {stock_symbol}. 
                                        Compile this into a comprehensive research report for the analysts.""",
                        expected_output=f"A detailed dossier containing technical levels, recent news context, and social sentiment for {stock_symbol}.",
                        agent=researcher
                    )
                    Crew(agents=[researcher], tasks=[research_task], verbose=False).kickoff()
                    research_res = research_task.output.raw
                
                with st.expander("🕵️ Research Phase: Data Gathered", expanded=True):
                    st.markdown(research_res)
                    
                st.session_state.debate_history.append({"round": "Research", "agent": "Market Researcher 🕵️", "message": research_res, "sentiment": "neutral"})
                
                last_bull_argument = "No previous arguments."
                last_bear_argument = "No previous arguments."
                
                # --- Step 2: Debate Loop ---
                for round_num in range(1, max_rounds + 1):
                    # Bull Turn
                    bull_desc = f"""
                        Read this background research:
                        {research_res}
                        
                        The Bear analyst recently argued: {last_bear_argument}
                        
                        Construct a highly persuasive 2-3 paragraph argument/rebuttal on why we should BUY or GO LONG on {stock_symbol} today. 
                        Directly attack weaknesses in the bear argument using the technical and news data.
                    """
                    bull_task = Task(description=bull_desc, expected_output="A 2-3 paragraph bullish trading thesis citing data.", agent=bull_agent)
                    
                    with st.spinner(f"🐂 Agent Bull is formulating a case (Round {round_num})..."):
                        Crew(agents=[bull_agent], tasks=[bull_task], verbose=False).kickoff()
                        bull_res = bull_task.output.raw
                    
                    with st.expander(f"🐂 Bullish Analysis - Round {round_num}", expanded=True):
                        st.markdown(f'<div class="base-message bull-message">{bull_res}</div>', unsafe_allow_html=True)
                        
                    st.session_state.debate_history.append({"round": round_num, "agent": "Agent Bull 🐂", "message": bull_res, "sentiment": "bullish"})
                    last_bull_argument = bull_res
                    
                    time.sleep(0.5) # Slight pause for effect
                    
                    # Bear Turn
                    bear_desc = f"""
                        Read this background research:
                        {research_res}
                        
                        The Bull analyst recently argued: {last_bull_argument}
                        
                        Construct a highly persuasive 2-3 paragraph argument/rebuttal on why we should SELL or GO SHORT on {stock_symbol} today. 
                        Directly attack weaknesses in the bull argument using the technical and news data.
                    """
                    bear_task = Task(description=bear_desc, expected_output="A 2-3 paragraph bearish trading thesis citing data.", agent=bear_agent)
                    
                    with st.spinner(f"🐻 Agent Bear is formatting a counter-argument (Round {round_num})..."):
                        Crew(agents=[bear_agent], tasks=[bear_task], verbose=False).kickoff()
                        bear_res = bear_task.output.raw
                        
                    with st.expander(f"🐻 Bearish Counter-Analysis - Round {round_num}", expanded=True):
                        st.markdown(f'<div class="base-message bear-message">{bear_res}</div>', unsafe_allow_html=True)
                        
                    st.session_state.debate_history.append({"round": round_num, "agent": "Agent Bear 🐻", "message": bear_res, "sentiment": "bearish"})
                    last_bear_argument = bear_res
                    
                    time.sleep(0.5)
                
                # --- Step 3: Judge Verdict ---
                judge_desc = f"""
                    Review the full research report for {stock_symbol}:
                    {research_res}
                    
                    The Final Bull argument is: {last_bull_argument}
                    The Final Bear argument is: {last_bear_argument}
                    
                    Deliver a final recommendation formatted strictly in markdown with:
                    - **Decision**: BUY / SELL / HOLD
                    - **Confidence**: 1-10
                    - **Entry / Target / Stop Loss Levels**
                    - **Reasoning**: A concise summary of why you chose this path over the alternatives.
                """
                judge_task = Task(description=judge_desc, expected_output="A structured markdown trading recommendation with clear levels and reasoning.", agent=judge_agent)
                
                with st.spinner("⚖️ Chief Risk Officer is reviewing the debate to render a final verdict..."):
                    Crew(agents=[judge_agent], tasks=[judge_task], verbose=False).kickoff()
                    judge_res = judge_task.output.raw
                    
                st.session_state.final_recommendation_text = judge_res

            st.success("✨ Agentic Workflow Completed Successfully!")

            st.markdown("---")
            st.subheader("🎯 Final Recommendation (Chief Risk Officer)")
            st.markdown(f'''
            <div class="final-recommendation">
                <h3>⚖️ Chief Risk Officer Verdict</h3>
                {st.session_state.final_recommendation_text}
            </div>
            ''', unsafe_allow_html=True)
            
            # Add copy to clipboard button for the recommendation
            if st.session_state.final_recommendation_text:
                if st.button("📋 Copy Recommendation", key="copy_rec"):
                    clean_recommendation = st.session_state.final_recommendation_text.replace('−', '-').replace('*', '')
                    clean_recommendation = ' '.join(clean_recommendation.split())
                    st.code(clean_recommendation, language=None)
                    st.success("💡 Tip: Click the copy icon in the top-right corner of the text box above to copy!")

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
            if stock_data.news_sentiment and stock_data.news_sentiment['summary']:
                st.write(stock_data.news_sentiment['summary'])
                if stock_data.news_sentiment['headlines']:
                    for i, headline in enumerate(stock_data.news_sentiment['headlines']):
                        # If headline is a dict, get 'title' and 'publisher'; if string, use as is
                        if isinstance(headline, dict):
                            title_display = headline.get('title', 'No title available')
                            publisher_display = headline.get('publisher', 'N/A Publisher')
                            sentiment_display = headline.get('sentiment', 'Neutral').capitalize()
                            link_display = headline.get('link', None)
                        else:
                            title_display = headline if headline else 'No title available'
                            publisher_display = 'N/A Publisher'
                            sentiment_display = 'Neutral'
                            link_display = None
                        if link_display:
                            st.markdown(f"- **{sentiment_display}**: [{title_display}]({link_display}) ({publisher_display})")
                        else:
                            st.markdown(f"- **{sentiment_display}**: {title_display} ({publisher_display})")
            else:
                st.write("No news sentiment data available.")

            # ADD THIS NEW SECTION FOR REDDIT ANALYSIS
            st.markdown("---")
            st.subheader("🤖 Reddit Community Sentiment")
            if stock_data.reddit_sentiment and stock_data.reddit_sentiment['summary']:
                st.write(stock_data.reddit_sentiment['summary'])
                
                # Display sentiment breakdown
                sentiment_breakdown = stock_data.reddit_sentiment.get('sentiment_breakdown', {})
                if sentiment_breakdown:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🟢 Bullish", sentiment_breakdown.get('bullish', 0))
                    with col2:
                        st.metric("🔴 Bearish", sentiment_breakdown.get('bearish', 0))
                    with col3:
                        st.metric("⚪ Neutral", sentiment_breakdown.get('neutral', 0))
                
                # Display top discussions
                discussions = stock_data.reddit_sentiment.get('discussions', [])
                if discussions:
                    st.write("**Top Reddit Discussions:**")
                    for i, discussion in enumerate(discussions[:5]):  # Show top 5
                        sentiment_emoji = "🟢" if discussion['sentiment'] == 'bullish' else "🔴" if discussion['sentiment'] == 'bearish' else "⚪"
                        
                        with st.expander(f"{sentiment_emoji} r/{discussion['subreddit']} - {discussion['title'][:60]}..." if len(discussion['title']) > 60 else f"{sentiment_emoji} r/{discussion['subreddit']} - {discussion['title']}"):
                            st.write(f"**Score:** {discussion['score']} upvotes")
                            st.write(f"**Sentiment:** {discussion['sentiment'].capitalize()}")
                            st.write(f"**Preview:** {discussion['content_preview']}")
                            if discussion.get('url'):
                                st.markdown(f"[View on Reddit]({discussion['url']})")
            else:
                st.write("No Reddit sentiment data available.")

            st.markdown("---") # Visual separator
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
                # Replace the copy_to_clipboard_button call with this:
                if st.button("📋 Copy Recommendation", key="copy_rec"):
                    # Clean the text for copying
                    clean_recommendation = st.session_state.final_recommendation_text.replace('−', '-').replace('*', '')
                    clean_recommendation = ' '.join(clean_recommendation.split())  # Remove extra whitespace
                    
                    # Use Streamlit's native approach
                    st.code(clean_recommendation, language=None)
                    st.success("💡 Tip: Click the copy icon in the top-right corner of the text box above to copy!")

    # Footer with disclaimer
    st.divider()
    st.markdown("""
    <div class="footer-disclaimer">
        🤖 Powered by OpenAI GPT-3.5 | 📈 Market data from Yahoo Finance<br>
        <span style='color: red; font-weight: bold;'>Disclaimer: This is for educational and entertainment purposes only and not financial advice. Day trading involves substantial risk.</span>
    </div>
    """, unsafe_allow_html=True)

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()
