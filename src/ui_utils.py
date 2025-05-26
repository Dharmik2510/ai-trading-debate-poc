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
    
    return 
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