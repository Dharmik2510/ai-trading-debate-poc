# ui_utils.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Assuming TradingAgent is imported or defined elsewhere for display_message
# from agents import TradingAgent # Would import if this were a standalone module
# For now, we'll make display_message accept basic agent info

def display_message(agent_info: dict, message: str, sentiment: str = ""):
    """
    Displays an agent's message with appropriate styling and a sentiment icon.
    agent_info should be a dict with 'name', 'avatar', and 'color'.
    """
    css_class = "bull-message" if "Bull" in agent_info['name'] else "bear-message"
    sentiment_icon = ""
    if sentiment == "bullish":
        sentiment_icon = "⬆️"
    elif sentiment == "bearish":
        sentiment_icon = "⬇️"
    else:
        sentiment_icon = "↔️" # Neutral icon
        
    st.markdown(f"""
    <div class="{css_class}">
        <strong>{agent_info['avatar']} {agent_info['name']} {sentiment_icon}</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def create_stock_chart(symbol: str, hist_data: pd.DataFrame, period: str):
    """
    Creates and returns a Plotly candlestick chart with volume, MAs, and Bollinger Bands.
    hist_data is a DataFrame containing the historical data.
    """
    if hist_data.empty:
        st.warning(f"No chart data available for {symbol} for the selected period ({period}).")
        return None
        
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price & Indicators', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Price chart (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name="Price",
            increasing_line_color='#03ad2b', # Green for increasing
            decreasing_line_color='#e60017'  # Red for decreasing
        ),
        row=1, col=1
    )
    
    # Add 20-day Moving Average
    if len(hist_data) >= 20:
        fig.add_trace(
            go.Scatter(x=hist_data.index, y=hist_data['Close'].rolling(window=20).mean(), mode='lines', name='20-MA', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    # Add 50-day Moving Average
    if len(hist_data) >= 50:
        fig.add_trace(
            go.Scatter(x=hist_data.index, y=hist_data['Close'].rolling(window=50).mean(), mode='lines', name='50-MA', line=dict(color='purple', width=1)),
            row=1, col=1
        )

    # Add Bollinger Bands
    if len(hist_data) >= 20:
        prices_for_bb = hist_data['Close']
        window = 20
        num_std_dev = 2
        rolling_mean_bb = prices_for_bb.rolling(window=window).mean()
        rolling_std_bb = prices_for_bb.rolling(window=window).std()
        upper_band_chart = rolling_mean_bb + (rolling_std_bb * num_std_dev)
        lower_band_chart = rolling_mean_bb - (rolling_std_bb * num_std_dev)

        fig.add_trace(
            go.Scatter(x=hist_data.index, y=upper_band_chart, mode='lines', name='BB Upper', line=dict(color='grey', width=1, dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist_data.index, y=lower_band_chart, mode='lines', name='BB Lower', line=dict(color='grey', width=1, dash='dot')),
            row=1, col=1
        )
    
    # Volume chart
    fig.add_trace(
        go.Bar(x=hist_data.index, y=hist_data['Volume'], name="Volume", marker_color='lightblue'),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{symbol} - Last {period.replace('mo', ' Months').replace('y', ' Year')}",
        xaxis_rangeslider_visible=False, # Hide range slider for cleaner look
        height=600, # Increased height for better visibility
        template="plotly_white", # Clean white background theme
        hovermode="x unified" # Shows all trace values at a given x-coordinate on hover
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def copy_to_clipboard_button(text_to_copy):
    """Creates a properly styled copy button with clean text formatting"""
    # Clean the text by removing extra formatting artifacts
    clean_text = text_to_copy.replace('−', '-').replace('*', '').replace('$', '\\$')
    
    # Create a clean, readable version
    clean_text = ' '.join(clean_text.split())  # Remove extra whitespace
    
    # Create the button with proper JavaScript
    button_html = f"""
    <button class="copy-button" onclick="
        const textToCopy = `{clean_text}`;
        navigator.clipboard.writeText(textToCopy).then(function() {{
            // Change button text temporarily to show success
            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '✅ Copied!';
            btn.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
            setTimeout(() => {{
                btn.innerHTML = originalText;
                btn.style.background = 'linear-gradient(135deg, #007bff 0%, #0056b3 100%)';
            }}, 2000);
        }}).catch(function(err) {{
            console.error('Could not copy text: ', err);
            alert('Copy failed. Please select and copy the text manually.');
        }});
    ">
        📋 Copy Recommendation
    </button>
    """
    
    st.markdown(button_html, unsafe_allow_html=True)