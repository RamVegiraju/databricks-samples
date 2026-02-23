"""
test_chat.py - Terminal chat loop with two-tier Lakebase memory.

Short-term:  messages table          every turn, current session only
Long-term:   long_term_memory table  session summaries + pgvector, across sessions

Run:  python3 test_chat.py
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from db import Database
import memory

load_dotenv()

SYSTEM_PROMPT = "You are a helpful, concise assistant."

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=os.environ["DATABRICKS_BASE_URL"],
)


def call_claude(history: list[dict], long_term: list[str]) -> tuple[str, int, int]:
    """
    Build the messages list sent to Claude:
      1. System prompt
      2. Long-term memory block (injected once at session start, if any)
      3. Short-term history (all turns in the current session)
    Returns (reply, prompt_tokens, completion_tokens).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if long_term:
        memory_block = (
            "What you know about this user from past sessions:\n" +
            "\n".join(f"- {s}" for s in long_term)
        )
        messages.append({"role": "system", "content": memory_block})

    messages.extend(history)

    response = client.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=messages,
        max_tokens=1024,
    )
    msg   = response.choices[0].message.content
    usage = response.usage
    return msg, usage.prompt_tokens, usage.completion_tokens


def chat_loop(username: str):
    db = Database()

    # ── Login ─────────────────────────────────────────────────────────────
    user_id    = db.get_or_create_user(username)
    session_id = db.start_session(user_id)
    print(f"\nLogged in  user={username}  session={session_id[:8]}...")
    print("Type 'quit' to exit.\n")

    long_term_context: list[str] = []
    first_turn = True

    # ── Chat loop ──────────────────────────────────────────────────────────
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        # On first message: embed query, retrieve top-5 relevant past memories
        if first_turn:
            query_vec          = memory.embed([user_input])[0]
            long_term_context  = db.get_relevant_memories(user_id, query_vec)
            print(f"[memory] Retrieved {len(long_term_context)} long-term memories")
            first_turn = False

        # Persist user turn to short-term store
        db.add_message(session_id, user_id, "user", user_input)

        # Load short-term history for this session
        history = db.get_session_history(session_id)

        # Call Claude
        reply, prompt_tokens, completion_tokens = call_claude(history, long_term_context)

        # Persist assistant reply
        db.add_message(session_id, user_id, "assistant", reply, token_count=completion_tokens)

        print(f"\nClaude: {reply}")
        print(f"  [prompt={prompt_tokens}  completion={completion_tokens}]\n")

    # ── Logout ─────────────────────────────────────────────────────────────
    all_messages = db.get_session_history(session_id)
    if all_messages:
        print("Summarizing session for long-term memory...")
        summary       = memory.summarize(all_messages, client)
        summary_vec   = memory.embed([summary])[0]
        db.save_long_term_memory(user_id, session_id, summary, summary_vec)
        print(f"Summary: {summary}\n")

    db.end_session(session_id)
    stats = db.get_session_stats(session_id)
    print(f"Session ended - {stats['user_turns']} user turns, {stats['total_tokens']} tokens stored.")
    db.close()


if __name__ == "__main__":
    chat_loop("alice")
