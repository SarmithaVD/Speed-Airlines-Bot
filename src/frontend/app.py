# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 18:58:49 2025

@author: keertanapriya
"""
import streamlit as st
import time
import redis
import uuid
from streamlit_autorefresh import st_autorefresh
import requests
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))  # Add root to path

from extract import AirlineQueryParser

st.set_page_config(page_title="Speed Airline", page_icon="✈️", layout="wide")
st.title("Speed Airlines Chatbot")
st.caption("Chat with support using text.")

MESSAGE_URL = "http://127.0.0.1:8000/chat/"

# ------------------------------ 
# Custom CSS for chat and intro card
# ------------------------------
st.markdown("""
<style>
h1 { margin-bottom: 0rem !important; }
.chat-box { display: flex; flex-direction: column; gap: 4px; margin-top: 1rem; }
.user-bubble { background: linear-gradient(135deg, #1e90ff, #007bff); color: white; border-radius: 12px; padding: 10px 14px; margin-left: auto; max-width: 70%; border: 2px solid #0056b3; text-align: right; font-family: Arial, sans-serif; font-size: 1rem; }
.assistant-bubble { background: #f0f8ff; border: 2px solid #87ceeb; color: #212529; border-radius: 12px; padding: 10px 14px; margin-right: auto; max-width: 70%; text-align: left; font-family: Arial, sans-serif; font-size: 1rem; }
.typing-bubble { background: #e8f4ff; color: #6c757d; border-radius: 12px; padding: 10px 14px; margin-right: auto; max-width: 40%; font-style: italic; }
.intro-text { font-size: 1.1rem; color: #333; margin-top: 0.5rem; margin-bottom: 1rem; }
.category-card { display: flex; justify-content: center; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; }
.card-btn { background: linear-gradient(135deg, #007bff, #1e90ff); color: white; padding: 0.9rem 1.5rem; border-radius: 12px; border: none; cursor: pointer; font-size: 1rem; font-weight: 500; transition: 0.3s ease; }
.card-btn:hover { transform: scale(1.05); background: linear-gradient(135deg, #0056b3, #007bff); }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Redis connection
# ------------------------------
@st.cache_resource
def get_redis_connection(host="localhost", port=6379, db=0):
    r = redis.Redis(host=host, port=port, db=db)
    try:
        r.ping()
        return r
    except redis.exceptions.ConnectionError:
        st.error("❌ Could not connect to Redis. Make sure Redis server is running.")
        return None

r = get_redis_connection()

# ------------------------------
# Session management
# ------------------------------
SESSION_TIMEOUT = 600  # 10 minutes

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def load_session():
    if r:
        data = r.get(f"session:{st.session_state.session_id}")
        if data:
            return eval(data)
    return {"messages": [], "latency_history": [], "last_activity": time.time()}

def save_session(session_data):
    if r:
        r.set(f"session:{st.session_state.session_id}", str(session_data), ex=SESSION_TIMEOUT)

def reset_session():
    session_data = {"messages": [], "latency_history": [], "last_activity": time.time()}
    save_session(session_data)
    return session_data

# Auto-refresh every 30 sec
st_autorefresh(interval=30000, limit=None, key="refresh")

# Load or initialize session
session_data = load_session()

# Check timeout
if time.time() - session_data["last_activity"] > SESSION_TIMEOUT:
    session_data = reset_session()
    st.info("🕒 Session timed out due to inactivity. Chat reset.")

# ------------------------------
# Sidebar Info
# ------------------------------
with st.sidebar:
    st.header("Welcome!!")
    st.info(
        """
        • Ask about flight schedules.  
        • Baggage policies.  
        • Cancellations.  
        • Lookup passenger by ID (type 'Passenger ID: XXXX').  
        """
    )
    if session_data["latency_history"]:
        avg_latency = sum(session_data["latency_history"]) / len(session_data["latency_history"])
        st.metric("Avg. Response Time", f"{avg_latency:.2f} sec")
        st.write(f"Last response: {session_data['latency_history'][-1]:.2f} sec")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        session_data = reset_session()
        st.session_state.first_prompt_shown = False
        st.session_state.category_selected = None
        st.rerun()
    if st.session_state.get("first_prompt_shown", False):
        if st.button("⬅️ Back to Home"):
            st.session_state.first_prompt_shown = False
            st.session_state.category_selected = None
            session_data = reset_session()
            st.rerun()

# ------------------------------
# Display chat messages
# ------------------------------
st.markdown('<div class="chat-box">', unsafe_allow_html=True)
for msg in session_data["messages"]:
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f'<div class="{role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Intro Page
# ------------------------------
if "first_prompt_shown" not in st.session_state:
    st.session_state.first_prompt_shown = False
if "category_selected" not in st.session_state:
    st.session_state.category_selected = None

if not st.session_state.first_prompt_shown and not session_data["messages"]:
    with st.container():
        st.markdown(
            '<p class="intro-text">We’re here to help you with bookings, flights, and travel assistance.<br>'
            'Please choose a category below to get started.</p>',
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💬 Generic", key="generic", use_container_width=True):
                st.session_state.category_selected = "Generic"
        with col2:
            if st.button("📘 Booking & Reservation", key="booking", use_container_width=True):
                st.session_state.category_selected = "Booking & Reservation"
        with col3:
            if st.button("🛫 Pre-Flight & Day-of-Travel", key="preflight", use_container_width=True):
                st.session_state.category_selected = "Pre-Flight & Day-of-Travel"

        if st.session_state.category_selected:
            st.session_state.first_prompt_shown = True
            category = st.session_state.category_selected
            if category == "Generic":
                reply_content = (
                    "Thank you for reaching out. We’ll assist you shortly.\n\n"
                    "**Tip:** You can ask about flight schedules, baggage policies, cancellations."
                )
            elif category == "Booking & Reservation":
                reply_content = "Please provide your booking details: Name, Email, or Booking ID."
            elif category == "Pre-Flight & Day-of-Travel":
                reply_content = "Please provide your Passenger ID (type 'Passenger ID: XXXX')."

            session_data["messages"].append({"role": "assistant", "content": reply_content})
            session_data["last_activity"] = time.time()
            save_session(session_data)
            st.rerun()

# ------------------------------
# Initialize LLM parser
# ------------------------------
if "llm_parser" not in st.session_state:
    st.session_state.llm_parser = AirlineQueryParser(api_key="YOUR_GEMINI_API_KEY")

# ------------------------------
# Handle user input
# ------------------------------
prompt = st.chat_input("Ask me anything.")
if prompt:
    session_data["messages"].append({"role": "user", "content": prompt})
    session_data["last_activity"] = time.time()
    save_session(session_data)
    st.rerun()

# ------------------------------
# Generate assistant response VIA API
# ------------------------------
if session_data["messages"] and session_data["messages"][-1]["role"] == "user":
    user_message = session_data["messages"][-1]["content"]
    start_time = time.time()

    with st.spinner("Thinking..."):
        try:
            payload = {"query": user_message, "session_id": st.session_state.session_id}
            response = requests.post(MESSAGE_URL, json=payload, timeout=60)
            response.raise_for_status()
            api_response_data = response.json()

            # --- THE CRITICAL FIX: Check for common keys ---
            # The Orchestrator generated the answer, so it MUST be in one of the API's JSON keys.
            # Check for 'reply' (default), 'answer', or 'final_answer'.
            reply_content = api_response_data.get("reply")
            if not reply_content:
                reply_content = api_response_data.get("answer")
            if not reply_content:
                reply_content = api_response_data.get("final_answer")
            if not reply_content:
                # Final fallback message if none of the expected keys are found
                reply_content = "Sorry, I couldn't generate a response from the API. The API returned a response, but the final answer field was missing."
            # Note: We can still extract RAG context for debugging, but we don't need the second LLM call.
            # rag_context = ""
            # if api_response_data.get("retrieved_chunks"):
            #     rag_context = "\n".join([chunk.get("content", "") for chunk in api_response_data["retrieved_chunks"]])

            # # REMOVED: Redundant and error-prone local LLM call
            # # final_answer_dict = st.session_state.llm_parser.generate_answer_from_context(...)
            # # reply_content = final_answer_dict.get("reply", "...")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the processing API!: {e}")
            reply_content = "Sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")
            reply_content = "Sorry, something went wrong while processing your request."

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the processing API!: {e}")
            reply_content = "Sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")
            reply_content = "Sorry, something went wrong while processing your request."

    latency = time.time() - start_time
    session_data["messages"].append({"role": "assistant", "content": reply_content})
    session_data["latency_history"].append(latency)
    session_data["last_activity"] = time.time()
    save_session(session_data)
    st.rerun()
