from dotenv import load_dotenv
from openai import OpenAI
import json, os, random, requests, sqlite3, re
from pypdf import PdfReader
import gradio as gr
import faiss, numpy as np
from glob import glob
from pathlib import Path
# ---------- FastAPI ----------
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv(override=True)

# --- NEW HELPERS for non-md sources ---------------------------------
from typing import Iterable, Optional, Sequence, Tuple
try:
    import nbformat  # for .ipynb
except Exception:
    nbformat = None

def read_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        chunks = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                chunks.append(t)
        return "\n".join(chunks)
    except Exception as e:
        print(f"[KB] PDF read failed {path}: {e}")
        return ""

def read_ipynb_text(path: str) -> str:
    if nbformat is None:
        print("[KB] nbformat not installed; skipping .ipynb:", path)
        return ""
    try:
        nb = nbformat.read(path, as_version=4)
        parts = []
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                parts.append(cell.source)
            elif cell.cell_type == "code":
                # keep code lightly—useful for RAG but don’t over-index
                parts.append("```code\n" + cell.source + "\n```")
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[KB] ipynb read failed {path}: {e}")
        return ""

def read_plain_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[KB] text read failed {path}: {e}")
        return ""

def iter_kb_files() -> Iterable[str]:
    # extendable place to add patterns
    exts = (".md", ".txt", ".pdf", ".ipynb", ".r", ".rmd", ".py")
    # Try both relative and absolute paths
    kb_patterns = ["kb/**/*.*", os.path.join(os.getcwd(), "kb", "**", "*.*")]
    seen_files = set()  # Track files we've already yielded
    for pattern in kb_patterns:
        for fp in glob(pattern, recursive=True):
            if fp.lower().endswith(exts):
                # Normalize path
                fp = os.path.normpath(fp)
                fp_abs = os.path.abspath(fp)
                if os.path.exists(fp) and fp_abs not in seen_files:
                    seen_files.add(fp_abs)
                    yield fp

ASSIGNMENT_KEYWORDS = ("assignment", "project", "hw", "dama", "paltsokas")
ASSIGNMENT_EXTS = (".pdf", ".md", ".txt", ".ipynb", ".r", ".rmd")

def _looks_like_assignment(path: str) -> bool:
    low = path.lower()
    if not low.endswith(ASSIGNMENT_EXTS):
        return False
    return any(keyword in low for keyword in ASSIGNMENT_KEYWORDS)

def _relative_to_kb(fp: str) -> str:
    return os.path.relpath(fp, KB_DIR).replace("\\", "/")

def _filter_candidates_by_folder(
    candidates: Sequence[str],
    folder: str,
    allowed_exts: Optional[Sequence[str]],
) -> list[str]:
    norm = folder.strip().strip("/").replace("\\", "/")
    exts = tuple(e.lower() for e in allowed_exts) if allowed_exts else None
    out = []
    for fp in candidates:
        rel = _relative_to_kb(fp)
        if norm and not rel.startswith(norm):
            continue
        if exts and not fp.lower().endswith(exts):
            continue
        out.append(fp)
    return out

def select_random_assignment(
    folder_filters: Optional[Sequence[str]] = None,
    allowed_exts: Optional[Sequence[str]] = None,
) -> Optional[Tuple[str, str]]:
    """Pick a random assignment-like file from the KB. Returns (display_name, relative_path)."""
    all_files = list(iter_kb_files())
    if not all_files:
        return None

    candidates: list[str] = []
    if folder_filters:
        for folder in folder_filters:
            candidates = _filter_candidates_by_folder(all_files, folder, allowed_exts)
            if candidates:
                break
    else:
        exts = tuple(e.lower() for e in allowed_exts) if allowed_exts else None
        candidates = [
            fp for fp in all_files
            if _looks_like_assignment(fp)
            and (not exts or fp.lower().endswith(exts))
        ]

    if not candidates:
        return None

    chosen = random.choice(candidates)
    rel_path = _relative_to_kb(chosen)
    raw_title = Path(chosen).stem
    display_name = re.sub(r"[_\\-]+", " ", raw_title).strip()
    if not display_name:
        display_name = rel_path
    return display_name, rel_path

def read_any_to_text(fp: str) -> str:
    low = fp.lower()
    if low.endswith(".pdf"):
        return read_pdf_text(fp)
    if low.endswith(".ipynb"):
        return read_ipynb_text(fp)
    # .md, .txt, .r, .rmd, .py → plain text
    return read_plain_text(fp)

def load_assignment_context(rel_path: str, max_chars: int = 4000) -> Optional[str]:
    abs_path = os.path.join(KB_DIR, rel_path)
    if not os.path.exists(abs_path):
        print(f"[WARNING] Assignment context path not found: {abs_path}", flush=True)
        return None
    raw = read_any_to_text(abs_path)
    if not raw:
        print(f"[WARNING] No extractable text for assignment: {abs_path}", flush=True)
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned

# --------------------------------------------------------------------

# =========================
# Config / constants
# =========================
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# RAG paths (use your repo's ./kb folder)
KB_DIR = os.getenv("KB_DIR", "kb")
KB_GLOB = os.getenv("KB_GLOB", f"{KB_DIR}/**/*.*")

FAISS_DIR = "models/faiss"
FAISS_INDEX = os.path.join(FAISS_DIR, "index.faiss")
FAISS_STORE = os.path.join(FAISS_DIR, "store.jsonl")

HELLO_THERE_RE = re.compile(r'^\s*[\W_]*hello\s+there[\W_]*\s*$', re.IGNORECASE)

# =========================
# Notifications (Pushover)
# =========================
def push(text):
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": os.getenv("PUSHOVER_TOKEN"),
                "user": os.getenv("PUSHOVER_USER"),
                "message": text,
            },
            timeout=8,
        )
    except Exception:
        # non-fatal for local dev
        pass

EMAIL_ADDRESS_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
EMAIL_KEYWORDS = ("email", "e-mail", "mail you", "reach you", "contact you")
POLICY_KEYWORDS = (
    "kill", "bomb", "explosive", "shoot", "weapon",
    "abuse", "hate crime", "self harm", "suicide",
    "explicit sexual", "child abuse", "terrorist",
    "hack", "malware", "ddos"
)

def _needs_pushover_alert(text: str) -> bool:
    low = (text or "").lower()
    if EMAIL_ADDRESS_RE.search(text or ""):
        return True
    if any(keyword in low for keyword in EMAIL_KEYWORDS):
        return True
    if any(keyword in low for keyword in POLICY_KEYWORDS):
        return True
    return False

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    if _needs_pushover_alert(question):
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
    global CLIENT
    CLIENT = c

def embed_texts(texts):
    if CLIENT is None:
        raise RuntimeError("OpenAI client not set. Call set_client(me.openai) at startup.")
    resp = CLIENT.embeddings.create(model=EMBEDDINGS_MODEL, input=texts)
    return [d.embedding for d in resp.data]

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
    """Index md/txt/pdf/ipynb/R files under kb/ into FAISS."""
    os.makedirs(FAISS_DIR, exist_ok=True)
    texts, meta = [], []
    pdf_count = 0
    other_count = 0

    print(f"[KB] Starting to index files from kb/ directory...", flush=True)
    all_files = list(iter_kb_files())
    print(f"[KB] Found {len(all_files)} files to index", flush=True)

    for fp in all_files:
        print(f"[KB] Processing: {fp}", flush=True)
        raw = read_any_to_text(fp)
        if not raw.strip():
            print(f"[KB] Warning: {fp} has no extractable text", flush=True)
            continue
        
        # Count PDFs vs other files
        if fp.lower().endswith('.pdf'):
            pdf_count += 1
        else:
            other_count += 1
        
        chunks = _split_md(raw)
        print(f"[KB] Split {fp} into {len(chunks)} chunks", flush=True)
        
        for ch in chunks:
            texts.append(ch)
            meta.append({"source": os.path.relpath(fp, "kb")})

    if not texts:
        print("[KB] No indexable text found.", flush=True)
        return 0

    print(f"[KB] Indexing {len(texts)} chunks ({pdf_count} PDFs, {other_count} other files)...", flush=True)
    vecs = embed_texts(texts)
    mat = np.array(vecs, dtype="float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    faiss.write_index(index, FAISS_INDEX)
    with open(FAISS_STORE, "w", encoding="utf-8") as f:
        for t, m in zip(texts, meta):
            f.write(json.dumps({"chunk": t, **m}, ensure_ascii=False) + "\n")

    print(f"[KB] Successfully indexed {len(texts)} chunks from assignments ({pdf_count} PDFs, {other_count} other files).", flush=True)
    return len(texts)


def _load_index():
    if not (os.path.exists(FAISS_INDEX) and os.path.exists(FAISS_STORE)):
        return None, []
    index = faiss.read_index(FAISS_INDEX)
    with open(FAISS_STORE, "r", encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]
    return index, meta

def rebuild_if_empty():
    """Rebuild if files exist but there are 0 meta rows."""
    idx, meta = _load_index()
    if idx is None or not meta:
        print("[INFO] FAISS check: index missing or empty — rebuilding…", flush=True)
        n = build_faiss_index()
        if n > 0:
            print(f"[INFO] Successfully built FAISS index with {n} chunks.", flush=True)
        else:
            print("[WARNING] FAISS index build returned 0 chunks. Check KB folder.", flush=True)
        return n
    print(f"[INFO] FAISS check: loaded {len(meta)} chunks.", flush=True)
    return len(meta)

def rag_search(query: str, k: int = 4):
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
        if i == -1: 
            continue
        source = meta[i]['source']
        chunk = meta[i]['chunk']
        # Prioritize PDFs and assignments in output
        if any(keyword in source.lower() for keyword in ['.pdf', 'hw', 'assignment', 'dama']):
            out.insert(0, f"[{source}] {chunk}")  # Put PDFs first
        else:
            out.append(f"[{source}] {chunk}")
    result = "\n\n".join(out) if out else "(no matches)"
    print(f"[DEBUG] rag_search found {len(out)} chunks, sources: {[meta[i]['source'] for i in idxs[0] if i != -1]}", flush=True)
    return result

def rag_lookup(query: str, k: int = 4):
    return {"context": rag_search(query, k)}

rag_lookup_json = {
    "name": "rag_lookup",
    "description": "Search Panos's personal knowledge base containing actual assignment PDFs, notebooks, and project files from the /kb folder. ALWAYS use this for questions about projects, assignments, studies, or coursework. Returns content from real files in the knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search in the KB - use specific terms like 'project', 'assignment', 'homework', 'DAMA', 'clustering', 'R script', etc."},
            "k": {"type": "integer", "description": "Top-K passages to retrieve. Use higher values (8-12) for project/assignment questions to get more comprehensive results.", "default": 8}
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
    con.execute("""
    CREATE TABLE IF NOT EXISTS qa(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      question TEXT NOT NULL,
      answer TEXT NOT NULL,
      tags TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );""")
    con.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS qa_fts USING fts5(
      question, answer, tags, content='',
      tokenize = 'unicode61 remove_diacritics 2'
    );""")
    return con

def qadb_lookup(question: str, fuzzy: bool = True, limit: int = 5):
    con = _qadb_conn(); cur = con.cursor()
    if fuzzy:
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
    cur.execute("INSERT INTO qa_fts(rowid, question, answer, tags) VALUES (last_insert_rowid(), ?, ?, ?)",
                (question, answer, tags))
    con.commit(); con.close()
    return {"saved": True}

def qadb_lookup_tool(question: str, fuzzy: bool = True, limit: int = 5):
    return qadb_lookup(question, fuzzy, limit)

def qadb_upsert_tool(question: str, answer: str, tags: str = None):
    return qadb_upsert(question, answer, tags)

qadb_lookup_json = {
    "name": "qadb_lookup_tool",
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
    "name": "qadb_upsert_tool",
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
        base = (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. "
            f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            f"If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. "
            f"If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool."
        )
        policy = (
            "\n\n# Tool Policy\n"
            "- **CRITICAL**: For ANY question about projects, assignments, studies, coursework, or work done during studies, "
            "you MUST call `rag_lookup` FIRST and use ONLY content from the /kb folder. Do NOT make up projects or use generic knowledge.\n"
            "- **ALWAYS** call `rag_lookup` first for any question about Panos/“I”/“me”, including experience, projects, background, skills, studies, achievements, or career advice.\n"
            "- **PRIORITY**: When asked about projects or assignments, the /kb folder contains actual assignment PDFs and notebooks. "
            "You MUST search the KB and use ONLY information from those files. Do not invent or use general knowledge about projects.\n"
            "- If the question looks reusable (bio blurbs, typical Qs), call `qadb_lookup_tool` next.\n"
            "- After drafting a concise answer, call `qadb_upsert_tool` to save it (tags: 'virtual-me').\n"
            "- If you still cannot answer after searching the KB, call `record_unknown_question`.\n"
            "- If the user is a potential lead, ask for an email and call `record_user_details`.\n"
        )
        summary = f"\n\n## Summary:\n{self.summary}\n"
        linkedin = f"\n## LinkedIn Profile:\n{self.linkedin}\n"
        closing = f"\nWith this context, please chat with the user, always staying in character as {self.name}."
        return base + policy + summary + linkedin + closing


    def chat(self, message, history):
        if HELLO_THERE_RE.match(message or ""):
            return "General Kenoooobiiii... I mean... Hi! How are you? 😊"

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

        # gather tool outputs to use as "context" for evaluation
        ctx_snippets = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "tool":
                ctx_snippets.append(m.get("content", ""))
        context_for_eval = "\n\n".join(ctx_snippets) if ctx_snippets else "(no ctx)"

        ev = evaluate_answer(self.openai, message, context_for_eval, draft)
        print(f"Evaluation: {ev}", flush=True)
        final = draft
        if ev.get("helpfulness", 3) < 4 or ev.get("faithfulness", 3) < 4:
            final = reflect_answer(self.openai, message, context_for_eval, draft, ev.get("feedback", ""))

        # auto-save good reusable answers in QADB
        try:
            fb = (ev.get("feedback", "") or "").lower()
            if len(final) <= 1500 and any(k in fb for k in ["clear", "helpful", "well structured", "faithful"]):
                qadb_upsert_tool(message, final, tags="virtual-me")
        except Exception:
            pass

        return final

# =========================
# Build Gradio app for both local & Spaces
# =========================
def build_demo():
    me = Me()
    set_client(me.openai)

    # Build FAISS KB once (or rebuild if empty)
    try:
        n = rebuild_if_empty()
        if n == 0:
            print("WARNING: KB has 0 chunks. Check files in 'kb/'", flush=True)
    except Exception as e:
        print("KB build skipped / failed:", e, flush=True)

    return gr.ChatInterface(me.chat, type="messages")

demo = build_demo()

# Create ONE shared Me() instance for the /chat endpoint
_shared_me = Me()
set_client(_shared_me.openai)

# Serve ./static (put your index.html here)
root = Path(__file__).resolve().parent
static_dir = root / "static"
static_dir.mkdir(exist_ok=True)

app = FastAPI(title="Panos — Career Conversations")

# Optional: Gradio UI at /gradio
# Note: Mount order matters - mount Gradio before other static routes
gr.mount_gradio_app(app, demo, path="/gradio")

# Serve your single-file website at /
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return (static_dir / "index.html").read_text(encoding="utf-8")

# Minimal schema for your new front-end
@app.post("/chat")
async def chat_api(payload: dict, request: Request):
    """
    Expects: {"message": "...", "history": [...]} (history is optional)
    Returns: {"reply": "..."}
    """
    message = (payload or {}).get("message", "")
    history = (payload or {}).get("history", [])
    if not isinstance(history, list):
        history = []
    
    augmented_message = message
    folder_targets: Optional[Sequence[str]] = None
    allowed_exts: Optional[Sequence[str]] = None
    try:
        normalized = (message or "").strip().lower()
        if "/kb/end_to_end_ml_projects/" in normalized:
            folder_targets = ["End_to_end_ML_projects"]
            allowed_exts = [".pdf", ".ipynb"]
        elif "/kb/ml_theory_practice/" in normalized:
            folder_targets = ["ML_Theory_Practice"]
            allowed_exts = [".pdf", ".ipynb", ".r"]
        elif "/kb/mathematics_for_ml/" in normalized:
            folder_targets = ["Mathematics_For_ML"]
            allowed_exts = [".pdf", ".ipynb"]
        elif "/kb/python_courses_1/" in normalized:
            folder_targets = ["Python_Courses_1", "Python_Courses_2"]
            allowed_exts = [".ipynb"]

        if folder_targets:
            selection = select_random_assignment(folder_targets, allowed_exts)
            if selection:
                display_name, rel_path = selection
                context = load_assignment_context(rel_path)
                repo_instruction = ""
                if rel_path.startswith("Python_Courses_1/"):
                    parts = rel_path.split("/", 1)
                    if len(parts) == 2 and parts[1]:
                        subfolder = parts[1].split("/", 1)[0]
                        repo_url = f"https://github.com/ppaltsokas/{subfolder}"
                        repo_instruction = (
                            f" Also share and highlight the GitHub repository link for this project using a clickable Markdown link: [{subfolder}]({repo_url}). "
                            "Encourage the user to review the code there."
                        )
                        print(f"[DEBUG] Repo link generated for Python project: {repo_url}", flush=True)
                    else:
                        print(f"[WARNING] Unable to derive repo link from path: {rel_path}", flush=True)

                augmented_message = (
                    f"{message}\n\n"
                    f"(Please focus on the assignment stored in `{rel_path}` from the knowledge base. "
                    f"This assignment is titled \"{display_name}\". "
                    f"Use only the provided context and do not invent additional details."
                    f"{repo_instruction})"
                )
                if context:
                    augmented_message += (
                        "\n\n"
                        f"Context from `{rel_path}`:\n"
                        "```"
                        f"\n{context}\n"
                        "```"
                    )
                else:
                    print(f"[WARNING] No context extracted for {rel_path}", flush=True)
                print(f"[DEBUG] Random assignment selected from {folder_targets}: {rel_path}", flush=True)
    except Exception as e:
        print(f"[WARNING] Failed to select random assignment: {e}", flush=True)
        augmented_message = message
    
    try:
        result = _shared_me.chat(augmented_message, history)
        reply = result if isinstance(result, str) else str(result)
    except Exception as e:
        import traceback
        print(f"[ERROR] /chat endpoint error: {e}", flush=True)
        print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
        reply = f"Sorry, something went wrong: {e}"
    
    return JSONResponse({"reply": reply})
