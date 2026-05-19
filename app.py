import streamlit as st
from agent.graph import run_agent

# --- Page config ---
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
    layout="centered"
)

# --- Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .stApp {
            background-color: #0f0f0f;
            color: #e0e0e0;
        }

        .stChatMessage {
            background-color: #1a1a1a !important;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
        }

        .stChatInputContainer {
            border-top: 1px solid #2a2a2a;
        }

        h1 {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.4rem;
            color: #ffffff;
            letter-spacing: -0.5px;
        }

        .subtitle {
            font-size: 0.85rem;
            color: #555;
            margin-top: -12px;
            margin-bottom: 24px;
            font-family: 'IBM Plex Mono', monospace;
        }

        .report-saved {
            font-size: 0.75rem;
            color: #444;
            font-family: 'IBM Plex Mono', monospace;
            margin-top: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("# 🔍 research assistant")
st.markdown('<p class="subtitle">web search + local docs → report</p>', unsafe_allow_html=True)

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User input ---
query = st.chat_input("Enter a research topic...")

if query:
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run agent and display response
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            result = run_agent(query)
        st.markdown(result)
        st.markdown('<p class="report-saved">✓ report saved to reports/</p>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": result})