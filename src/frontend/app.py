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

st.set_page_config(page_title="Speed Airline", page_icon="✈️", layout="wide")
st.title("Speed Airlines Chatbot")
st.caption("Chat with support using text.")

MESSAGE_URL = "http://127.0.0.1:8000/process_message/" 
# ------------------------------
# Custom CSS for chat and intro card
# ------------------------------
st.markdown("""
<style>
h1 { margin-bottom: 0rem !important; } /* Remove space below title */
.chat-box {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 1rem;
}
.user-bubble {
    background: linear-gradient(135deg, #1e90ff, #007bff);
    color: white;
    border-radius: 12px;
    padding: 10px 14px;
    margin-left: auto;
    max-width: 70%;
    border: 2px solid #0056b3;
    text-align: right;
    font-family: Arial, sans-serif;
    font-size: 1rem;
}
.assistant-bubble {
    background: #f0f8ff;
    border: 2px solid #87ceeb;
    color: #212529;
    border-radius: 12px;
    padding: 10px 14px;
    margin-right: auto;
    max-width: 70%;
    text-align: left;
    font-family: Arial, sans-serif;
    font-size: 1rem;
}
.typing-bubble {
    background: #e8f4ff;
    color: #6c757d;
    border-radius: 12px;
    padding: 10px 14px;
    margin-right: auto;
    max-width: 40%;
    font-style: italic;
}
.intro-text {
    font-size: 1.1rem;
    color: #333;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
.category-card {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.card-btn {
    background: linear-gradient(135deg, #007bff, #1e90ff);
    color: white;
    padding: 0.9rem 1.5rem;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 500;
    transition: 0.3s ease;
}
.card-btn:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #0056b3, #007bff);
}
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

    # --- Clear chat button
    if st.button("🗑️ Clear Chat"):
        session_data = reset_session()
        st.session_state.first_prompt_shown = False
        st.session_state.category_selected = None
        st.rerun()

    # --- Back button (only show after Continue)
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
# Intro Page (Updated)
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
    start_time = time.time() # Start latency timer

    
    with st.spinner("Thinking..."): 
        try:
            payload = {
                "user_id": "streamlit_user", # Or derive a more persistent user ID if needed
                "session_id": st.session_state.session_id,
                "message": user_message
            }

            # post req to fast api endpoint
            response = requests.post(MESSAGE_URL, json=payload, timeout=60) # Add timeout
            response.raise_for_status() # raise exception for bad codes

            # process the successful response from llm
            api_response_data = response.json()

            # Assuming your API returns the 'RetrievalResponse' model for now:
            # We need to format the chunks for display or eventually pass to an LLM here
            # For now, let's just show a summary of retrieved content
            reply_content = f"Okay, I found the following information related to '{api_response_data['original_query']}':\n\n"
            if api_response_data['retrieved_chunks']:
                for i, chunk in enumerate(api_response_data['retrieved_chunks']):
                    reply_content += f"Chunk {i+1} (from {chunk.get('header_path', 'N/A')}):\n"
                    reply_content += f"{chunk.get('content', 'N/A')[:200]}...\n\n" # Show snippet
            else:
                reply_content += "No specific documents found in the knowledge base."

            # !!! IMPORTANT: LATER, you will replace the above formatting
            # with the actual LLM call using the retrieved chunks to get the final 'answer' !!!

            # final_llm_answer = call_your_llm_function(api_response_data['retrieved_chunks'], user_message)
            # reply_content = final_llm_answer

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the processing API!: {e}")
            reply_content = "Sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")
            reply_content = "Sorry, something went wrong while processing your request."

    # Calculate latency
    latency = time.time() - start_time

    # Update session data in Redis
    session_data["messages"].append({"role": "assistant", "content": reply_content})
    session_data["latency_history"].append(latency)
    session_data["last_activity"] = time.time()
    save_session(session_data)

    # Rerun Streamlit to display the new assistant message
    st.rerun()