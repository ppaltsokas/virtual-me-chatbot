from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import faiss
import numpy as np
from glob import glob
import sqlite3
import re  # <-- keep re here (used by regex + evaluator)

load_dotenv(override=True)

# =========================
# Config / constants
# =========================
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Phrase matcher for "hello there" with optional punctuation/emojis, any casing
HELLO_THERE_RE = re.compile(r'^\s*[\W_]*hello\s+there[\W_]*\s*$', re.IGNORECASE)

# =========================
# Notifications (Pushover)
# =========================
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Any additional information worth recording"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question you couldn't answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

# =========================
# RAG / FAISS (global client pattern)
# =========================
CLIENT = None  # will be set to the same OpenAI client used by Me()

def set_client(c: OpenAI):
    """Wire the top-level tools to the same OpenAI client as the app."""
    global CLIENT
    CLIENT = c

def embed_texts(texts):
    """Embed a list of texts with the global CLIENT."""
    if CLIENT is None:
        raise RuntimeError("OpenAI client not set. Call set_client(me.openai) at startup.")
    resp = CLIENT.embeddings.create(model=EMBEDDINGS_MODEL, input=texts)
    return [d.embedding for d in resp.data]

FAISS_DIR = "models/faiss"
FAISS_INDEX = os.path.join(FAISS_DIR, "index.faiss")
FAISS_STORE = os.path.join(FAISS_DIR, "store.jsonl")

def _split_md(text: str, max_chars: int = 1200):
    parts, buf, count = [], [], 0
    for line in text.splitlines(keepends=True):
        buf.append(line); count += len(line)
        if line.strip().startswith(("#", "##", "###")) and buf:
            parts.append("".join(buf).strip()); buf, count = [], 0
        elif count >= max_chars:
            parts.append("".join(buf).strip()); buf, count = [], 0
    if buf: parts.append("".join(buf).strip())
    return [p for p in parts if p]

def build_faiss_index():
    """Index all md/txt files under kb/ into FAISS."""
    os.makedirs(FAISS_DIR, exist_ok=True)
    texts, meta = [], []
    for fp in glob("kb/**/*.*", recursive=True):
        if fp.lower().endswith((".md", ".txt")):
            with open(fp, "r", encoding="utf-8") as f:
                raw = f.read()
            for ch in _split_md(raw):
                texts.append(ch)
                meta.append({"source": os.path.relpath(fp, "kb")})
    if not texts:
        return 0
    vecs = embed_texts(texts)
    mat = np.array(vecs, dtype="float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, FAISS_INDEX)
    with open(FAISS_STORE, "w", encoding="utf-8") as f:
        for t, m in zip(texts, meta):
            f.write(json.dumps({"chunk": t, **m}, ensure_ascii=False) + "\n")
    return len(texts)

def _load_index():
    if not (os.path.exists(FAISS_INDEX) and os.path.exists(FAISS_STORE)):
        return None, []
    index = faiss.read_index(FAISS_INDEX)
    meta = [json.loads(line) for line in open(FAISS_STORE, "r", encoding="utf-8")]
    return index, meta

def rag_search(query: str, k: int = 4):
    """Search the personal KB for top-k passages."""
    if CLIENT is None:
        raise RuntimeError("OpenAI client not set. Call set_client(me.openai) at startup.")
    index, meta = _load_index()
    if not index or not meta:
        return "(KB empty)"
    qv = np.array(embed_texts([query])[0], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(qv)
    scores, idxs = index.search(qv, k)
    out = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1: continue
        out.append(f"[{meta[i]['source']}] {meta[i]['chunk']}")
    return "\n\n".join(out) if out else "(no matches)"

# Tool wrapper for RAG
def rag_lookup(query: str, k: int = 4):
    return {"context": rag_search(query, k)}

rag_lookup_json = {
    "name": "rag_lookup",
    "description": "Search Panos's personal knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search in the KB"},
            "k": {"type": "integer", "description": "Top-K passages", "default": 4}
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

# =========================
# SQL Q&A (persistent memory)
# =========================
os.makedirs("data", exist_ok=True)
QADB = "data/qadb.sqlite"

def _qadb_conn():
    con = sqlite3.connect(QADB)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    # main table
    con.execute("""
    CREATE TABLE IF NOT EXISTS qa(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      question TEXT NOT NULL,
      answer TEXT NOT NULL,
      tags TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );""")
    # FTS5 virtual table (contentless index)
    con.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS qa_fts USING fts5(
      question, answer, tags, content='',
      tokenize = 'unicode61 remove_diacritics 2'
    );""")
    return con

def qadb_lookup(question: str, fuzzy: bool = True, limit: int = 5):
    con = _qadb_conn(); cur = con.cursor()
    if fuzzy:
        # bm25 ranking; quotes improve precision, fallback to plain
        try:
            cur.execute("""
              SELECT question, answer, tags
              FROM qa_fts
              WHERE qa_fts MATCH ?
              ORDER BY bm25(qa_fts) ASC
              LIMIT ?;
            """, (f'"{question}"', limit))
        except sqlite3.OperationalError:
            cur.execute("""
              SELECT question, answer, tags
              FROM qa_fts
              WHERE qa_fts MATCH ?
              ORDER BY bm25(qa_fts) ASC
              LIMIT ?;
            """, (question, limit))
    else:
        cur.execute("SELECT question,answer,tags FROM qa WHERE question = ? ORDER BY id DESC LIMIT ?",
                    (question, limit))
    rows = cur.fetchall(); con.close()
    return {"results": [{"question": q, "answer": a, "tags": t} for (q, a, t) in rows]}

def qadb_upsert(question: str, answer: str, tags: str = None):
    con = _qadb_conn(); cur = con.cursor()
    cur.execute("INSERT INTO qa(question,answer,tags) VALUES (?,?,?)", (question, answer, tags))
    # mirror into FTS
    cur.execute("INSERT INTO qa_fts(rowid, question, answer, tags) VALUES (last_insert_rowid(), ?, ?, ?)",
                (question, answer, tags))
    con.commit(); con.close()
    return {"saved": True}


# Tool-callable wrappers (names must match JSON "name")
def qadb_lookup_tool(question: str, fuzzy: bool = True, limit: int = 5):
    return qadb_lookup(question, fuzzy, limit)

def qadb_upsert_tool(question: str, answer: str, tags: str = None):
    return qadb_upsert(question, answer, tags)

# JSON schemas for the tools
qadb_lookup_json = {
    "name": "qadb_lookup",
    "description": "Search the reusable Q&A database for similar questions and answers.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "fuzzy": {"type": "boolean", "default": True},
            "limit": {"type": "integer", "default": 5}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}
qadb_upsert_json = {
    "name": "qadb_upsert",
    "description": "Save a good reusable answer into the Q&A database for future consistency.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "tags": {"type": "string"}
        },
        "required": ["question", "answer"],
        "additionalProperties": False
    }
}
# Ensure JSON "name" matches the actual callable names
qadb_lookup_json["name"] = "qadb_lookup_tool"
qadb_upsert_json["name"] = "qadb_upsert_tool"

# =========================
# Tools registry for OpenAI
# =========================
tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
    {"type": "function", "function": rag_lookup_json},
    {"type": "function", "function": qadb_lookup_json},
    {"type": "function", "function": qadb_upsert_json},
]

# =========================
# Evaluator / Reflector
# =========================
def evaluate_answer(client: OpenAI, user_q: str, context: str, draft: str):
    """Ask the model to score the draft and give targeted feedback."""
    eval_sys = {
        "role": "system",
        "content": (
            "You are an exacting evaluator. Score the assistant's reply on a 1-5 scale: "
            "(1) Helpfulness, (2) Faithfulness to retrieved context, (3) Style alignment with Panos. "
            "Return strict JSON: {\"helpfulness\":int, \"faithfulness\":int, \"style\":int, \"feedback\":string}."
        ),
    }
    msgs = [
        eval_sys,
        {"role": "user", "content": f"USER:\n{user_q}\n\nCONTEXT:\n{context}\n\nDRAFT:\n{draft}"},
    ]
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=msgs)
    text = resp.choices[0].message.content or "{}"
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"helpfulness": 3, "faithfulness": 3, "style": 3, "feedback": "(no-parse)"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"helpfulness": 3, "faithfulness": 3, "style": 3, "feedback": "(bad-json)"}

def reflect_answer(client: OpenAI, user_q: str, context: str, draft: str, feedback: str):
    """Have the model revise the draft once, applying the feedback."""
    refl_sys = {
        "role": "system",
        "content": (
            "You are a senior editor. Revise the answer to address evaluator feedback in one pass. "
            "Keep claims tied to the provided context. Be concise and clear."
        ),
    }
    msgs = [
        refl_sys,
        {"role": "user", "content": f"USER:\n{user_q}\n\nCONTEXT:\n{context}\n\nDRAFT:\n{draft}\n\nFEEDBACK:\n{feedback}"},
    ]
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=msgs)
    return resp.choices[0].message.content

def _assistant_msg_to_dict(msg):
    """Preserve tool_calls so OpenAI accepts following 'tool' messages."""
    out = {"role": msg.role, "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = []
        for tc in msg.tool_calls:
            out["tool_calls"].append({
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
    return out

# =========================
# App
# =========================
class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "Panagiotis Paltsokas"

        # Read LinkedIn PDF as plain text
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        # Read summary file
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        # Base persona
        base = (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. "
            f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            f"If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. "
            f"If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool."
        )
        # Tool policy
        policy = (
            "\n\n# Tool Policy\n"
            "- If the question is about Panos, first call `rag_lookup` to fetch knowledge-base context.\n"
            "- If the question looks reusable (bio blurbs, typical Qs), call `qadb_lookup_tool` to check for prior saved answers.\n"
            "- After you produce a concise, good reusable answer, call `qadb_upsert_tool` to save it (tags: 'virtual-me').\n"
            "- If you still cannot answer, call `record_unknown_question`.\n"
            "- If the user is a potential lead, politely ask for email and call `record_user_details`.\n"
        )
        # Static files content
        summary = f"\n\n## Summary:\n{self.summary}\n"
        linkedin = f"\n## LinkedIn Profile:\n{self.linkedin}\n"
        closing = f"\nWith this context, please chat with the user, always staying in character as {self.name}."
        return base + policy + summary + linkedin + closing

    def chat(self, message, history):
        # Easter egg: "hello there" with optional punctuation/emojis, any casing
        if HELLO_THERE_RE.match(message or ""):
            return "General Kenoooobiiii... I mean... Hi! How are you? 😊"

        # 1) normal tool-using loop to get a draft
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        draft = None

        while not done:
            response = self.openai.chat.completions.create(model=CHAT_MODEL, messages=messages, tools=tools)
            choice = response.choices[0]
            if choice.finish_reason == "tool_calls":
                msg = choice.message
                results = self.handle_tool_call(msg.tool_calls)
                messages.append(_assistant_msg_to_dict(msg))
                messages.extend(results)
            else:
                done = True
                draft = choice.message.content

        # 2) gather any tool outputs to use as "context" for evaluation
        ctx_snippets = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "tool":
                ctx_snippets.append(m.get("content", ""))
        context_for_eval = "\n\n".join(ctx_snippets) if ctx_snippets else "(no ctx)"

        # 3) evaluate the draft and optionally reflect
        ev = evaluate_answer(self.openai, message, context_for_eval, draft)
        print(f"Evaluation: {ev}", flush=True)
        final = draft
        if ev.get("helpfulness", 3) < 4 or ev.get("faithfulness", 3) < 4:
            final = reflect_answer(self.openai, message, context_for_eval, draft, ev.get("feedback", ""))

        # 4) auto-save good reusable answers in QADB
        try:
            fb = (ev.get("feedback", "") or "").lower()
            if len(final) <= 1500 and any(k in fb for k in ["clear", "helpful", "well structured", "faithful"]):
                qadb_upsert_tool(message, final, tags="virtual-me")
        except Exception:
            pass

        return final

# =========================
# Main
# =========================
if __name__ == "__main__":
    me = Me()
    set_client(me.openai)  # wire RAG embeddings to the same OpenAI client

    # Build FAISS KB once if missing
    try:
        if not (os.path.exists(FAISS_INDEX) and os.path.exists(FAISS_STORE)):
            n = build_faiss_index()
            print(f"Built FAISS KB with {n} chunks.")
    except Exception as e:
        print("KB build skipped / failed:", e)

    gr.ChatInterface(me.chat, type="messages").launch()
