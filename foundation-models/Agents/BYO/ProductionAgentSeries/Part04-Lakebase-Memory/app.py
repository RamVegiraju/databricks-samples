"""
app.py - Streamlit chatbot with two-tier Lakebase memory.

Run:  streamlit run app.py
"""

import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from db import Database
import memory

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Claude Chatbot", layout="wide")

# ---------------------------------------------------------------------------
# Cached singletons — created once, reused across reruns
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db() -> Database:
    return Database()

@st.cache_resource
def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["DATABRICKS_TOKEN"],
        base_url=os.environ["DATABRICKS_BASE_URL"],
    )

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "user_id":           None,
        "username":          None,
        "session_id":        None,
        "messages":          [],     # in-memory cache for current session
        "long_term_context": [],     # retrieved once on first message
        "first_turn":        True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------

def call_claude(history: list[dict], long_term: list[str]) -> tuple[str, int, int]:
    client   = get_client()
    messages = [{"role": "system", "content": "You are a helpful, concise assistant."}]

    if long_term:
        memory_block = (
            "What you know about this user from past sessions:\n" +
            "\n".join(f"- {s}" for s in long_term)
        )
        messages.append({"role": "system", "content": memory_block})

    messages.extend(history)

    response = client.chat.completions.create(
        model="databricks-gpt-oss-120b",
        messages=messages,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content
    if isinstance(raw, list):
        msg = "\n".join(block["text"] for block in raw if block.get("type") == "text")
    else:
        msg = raw
    usage = response.usage
    return msg, usage.prompt_tokens, usage.completion_tokens

# ---------------------------------------------------------------------------
# Login screen
# ---------------------------------------------------------------------------

def render_login():
    st.title("Claude Chatbot")
    st.caption("Powered by Databricks Foundation Model APIs + Lakebase memory")
    st.divider()

    username = st.text_input("Username", placeholder="Enter your username")

    if st.button("Login", type="primary", disabled=not username):
        db = get_db()
        user_id    = db.get_or_create_user(username.strip())
        session_id = db.start_session(user_id)

        st.session_state.user_id    = user_id
        st.session_state.username   = username.strip()
        st.session_state.session_id = session_id
        st.session_state.messages   = []
        st.session_state.first_turn = True
        st.rerun()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    db = get_db()

    with st.sidebar:
        st.markdown(f"**{st.session_state.username}**")
        st.caption(f"Session `{st.session_state.session_id[:8]}...`")
        st.divider()

        # Session stats
        if st.session_state.messages:
            stats = db.get_session_stats(st.session_state.session_id)
            st.metric("Turns",  stats["user_turns"])
            st.metric("Tokens", stats["total_tokens"])
            st.divider()

        # Past sessions
        past = db.get_user_sessions(st.session_state.user_id)
        # Exclude the current session
        past = [s for s in past if str(s["session_id"]) != st.session_state.session_id]

        if past:
            st.caption("Past sessions")
            for s in past[:10]:
                status = "closed" if s["is_closed"] else "active"
                label  = s["started_at"].strftime("%b %d %H:%M")
                st.caption(f"{label}  —  {status}")

        st.divider()

        if st.button("Logout", use_container_width=True):
            logout()

# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

def logout():
    db     = get_db()
    client = get_client()

    all_messages = db.get_session_history(st.session_state.session_id)
    if all_messages:
        with st.spinner("Saving session to long-term memory..."):
            summary     = memory.summarize(all_messages, client)
            summary_vec = memory.embed([summary])[0]
            db.save_long_term_memory(
                st.session_state.user_id,
                st.session_state.session_id,
                summary,
                summary_vec,
            )

    db.end_session(st.session_state.session_id)

    # Clear state
    for key in ["user_id", "username", "session_id", "messages",
                "long_term_context", "first_turn"]:
        st.session_state[key] = None if key in ("user_id", "username", "session_id") else \
                                 []   if key in ("messages", "long_term_context") else True
    st.rerun()

# ---------------------------------------------------------------------------
# Chat screen
# ---------------------------------------------------------------------------

def render_chat():
    render_sidebar()

    db = get_db()

    st.title("Claude")

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Message Claude...")
    if not user_input:
        return

    # On first message: retrieve long-term memories
    if st.session_state.first_turn:
        query_vec = memory.embed([user_input])[0]
        st.session_state.long_term_context = db.get_relevant_memories(
            st.session_state.user_id, query_vec
        )
        st.session_state.first_turn = False

    # Persist + display user message
    db.add_message(st.session_state.session_id, st.session_state.user_id, "user", user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call Claude
    with st.chat_message("assistant"):
        with st.spinner(""):
            history = db.get_session_history(st.session_state.session_id)
            reply, prompt_tokens, completion_tokens = call_claude(
                history, st.session_state.long_term_context
            )
        st.markdown(reply)
        st.caption(f"prompt={prompt_tokens}  completion={completion_tokens}")

    # Persist + cache assistant reply
    db.add_message(
        st.session_state.session_id,
        st.session_state.user_id,
        "assistant",
        reply,
        token_count=completion_tokens,
    )
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    init_state()
    if st.session_state.session_id is None:
        render_login()
    else:
        render_chat()

if __name__ == "__main__":
    main()
