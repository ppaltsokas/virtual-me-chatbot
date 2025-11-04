from dotenv import load_dotenv
from openai import OpenAI
import json, os, requests, sqlite3, re
from pypdf import PdfReader
import gradio as gr
import faiss, numpy as np
from glob import glob
from pathlib import Path
import shutil
from datetime import datetime
# Configure poppler path for pdf2image
POPPLER_PATH = None
# Try to find poppler in common locations
possible_poppler_paths = [
    r"F:\AI-Agents-Course-Ed\agents\1_foundations\career_conversations\poppler-25.07.0\Library\bin",
    os.path.join(os.path.dirname(__file__), "poppler-25.07.0", "Library", "bin"),
    os.path.join(os.getcwd(), "poppler-25.07.0", "Library", "bin"),
    "C:\\poppler\\bin",  # Common Windows installation
    os.path.join(os.environ.get("PROGRAMFILES", ""), "poppler", "bin"),
]

for path in possible_poppler_paths:
    if path and os.path.exists(path):
        POPPLER_PATH = path
        print(f"[INFO] Found poppler at: {POPPLER_PATH}", flush=True)
        break

if POPPLER_PATH:
    # Add poppler to PATH for this session
    os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    print(f"[INFO] pdf2image is available. Poppler path: {POPPLER_PATH}", flush=True)
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("Warning: pdf2image not available. Install with: pip install pdf2image. Also install poppler: https://github.com/oschwartz10612/poppler-windows/releases/")
# ---------- FastAPI ----------
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv(override=True)

# --- NEW HELPERS for non-md sources ---------------------------------
from typing import Iterable
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

def read_any_to_text(fp: str) -> str:
    low = fp.lower()
    if low.endswith(".pdf"):
        return read_pdf_text(fp)
    if low.endswith(".ipynb"):
        return read_ipynb_text(fp)
    # .md, .txt, .r, .rmd, .py → plain text
    return read_plain_text(fp)

# Directory where assignment images will be saved
assignment_images_directory = "screenshots/assignment_images/"

# Ensure the directory exists
os.makedirs(assignment_images_directory, exist_ok=True)

def get_relevant_assignment_files(query: str, max_files: int = 3):
    """Find relevant assignment files from the KB based on the query."""
    # Search the KB for relevant content with more results
    context = rag_search(query, k=12)  # Increased from 8 to 12
    print(f"[DEBUG] RAG context: {context[:300]}...", flush=True)
    
    # Extract file sources from context
    file_sources = set()
    for line in context.split('\n'):
        if line.startswith('[') and ']' in line:
            # Extract file path from [source] format
            source = line.split(']')[0].strip('[')
            file_sources.add(source)
    
    print(f"[DEBUG] Found file sources from RAG: {file_sources}", flush=True)
    
    # Find actual PDF files in kb directory
    pdf_files = []
    kb_base = "kb"  # Base directory for kb folder
    
    for file_source in list(file_sources)[:max_files * 2]:  # Check more sources
        # Try to find the file in kb directory
        possible_paths = [
            os.path.join(kb_base, file_source),
            os.path.join(kb_base, os.path.basename(file_source)),
            file_source,  # Try direct path
        ]
        for path in possible_paths:
            if os.path.exists(path) and path.lower().endswith('.pdf'):
                pdf_files.append(path)
                print(f"[DEBUG] Found PDF from RAG: {path}", flush=True)
                break
    
    # If no specific files found, get ALL assignment PDFs and prioritize them
    if not pdf_files:
        print(f"[DEBUG] No PDFs found from RAG, using fallback - searching all KB files...", flush=True)
        all_pdfs = list([f for f in iter_kb_files() if f.lower().endswith('.pdf')])
        print(f"[DEBUG] Total PDFs found in KB: {len(all_pdfs)}", flush=True)
        
        if all_pdfs:
            # Prioritize assignment/homework PDFs
            assignment_pdfs = [f for f in all_pdfs if any(keyword in f.lower() for keyword in ['hw', 'assignment', 'project', 'paltsokas', 'dama'])]
            if assignment_pdfs:
                pdf_files = assignment_pdfs[:max_files]
                print(f"[DEBUG] Using prioritized assignment PDFs: {pdf_files}", flush=True)
            else:
                # Fallback to any PDFs
                pdf_files = all_pdfs[:max_files]
                print(f"[DEBUG] Using fallback PDFs: {pdf_files}", flush=True)
        else:
            print(f"[WARNING] No PDFs found in KB directory at all!", flush=True)
    
    return pdf_files

def pdf_page_to_image(pdf_path: str, page_num: int = 0, max_pages: int = 3):
    """Convert PDF pages to images. Returns list of image file paths."""
    if not PDF2IMAGE_AVAILABLE:
        print(f"[DEBUG] pdf2image not available. Install it to enable screenshots.", flush=True)
        return []
    
    if not os.path.exists(pdf_path):
        print(f"[DEBUG] PDF file not found: {pdf_path}", flush=True)
        return []
    
    try:
        # Get total pages first
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"[DEBUG] Converting PDF {pdf_path} (total pages: {total_pages})", flush=True)
        
        # Convert first few pages to images
        last_page = min(page_num + max_pages, total_pages)
        
        # Use poppler_path if available
        convert_kwargs = {
            "first_page": page_num + 1,
            "last_page": last_page,
            "dpi": 150  # Good quality but not too large
        }
        
        if POPPLER_PATH:
            convert_kwargs["poppler_path"] = POPPLER_PATH
        
        images = convert_from_path(pdf_path, **convert_kwargs)
        
        image_paths = []
        pdf_name = Path(pdf_path).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for i, image in enumerate(images):
            image_path = os.path.join(
                assignment_images_directory,
                f"{pdf_name}_page_{page_num + i + 1}_{timestamp}.png"
            )
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
            print(f"[DEBUG] Created image: {image_path}", flush=True)
        
        return image_paths
    except Exception as e:
        import traceback
        print(f"[ERROR] Error converting PDF {pdf_path} to images: {e}", flush=True)
        print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
        return []

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
            "\n# Screenshot Capability\n"
            "- **IMPORTANT**: When users ask about projects or assignments, or explicitly request screenshots/images, "
            "the system will automatically provide screenshots from relevant assignment PDFs stored in the /kb folder. "
            "You should acknowledge this capability and let users know that screenshots will be included with your detailed response. "
            "Always be positive and helpful when users ask about screenshots - tell them that you can provide screenshots from your assignments."
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
    
    def chat_with_images(self, message, history):
        """Chat function that can return both text and images for assignment questions."""
        # Check if this is an assignment/project question - make detection more robust
        message_lower = message.lower()
        is_assignment_question = (
            "tell me in detail about a project or assignment" in message_lower or
            "tell me in detail about a project" in message_lower or
            ("project" in message_lower and "assignment" in message_lower) or
            ("assignment" in message_lower and ("detail" in message_lower or "tell me" in message_lower)) or
            "screenshot" in message_lower or
            ("provide" in message_lower and "image" in message_lower)
        )
        
        print(f"[DEBUG] Is assignment question: {is_assignment_question}, message: {message[:100]}...", flush=True)
        
        # Get the text response
        text_response = self.chat(message, history)
        
        # If it's an assignment question or explicitly asking for screenshots, also include images
        if is_assignment_question:
            print(f"[DEBUG] Attempting to get assignment images...", flush=True)
            try:
                # Find relevant assignment files
                relevant_files = get_relevant_assignment_files(message, max_files=2)
                print(f"[DEBUG] Found {len(relevant_files)} relevant PDF files", flush=True)
                
                # Convert PDF pages to images
                all_image_paths = []
                for pdf_file in relevant_files:
                    print(f"[DEBUG] Processing PDF: {pdf_file}", flush=True)
                    image_paths = pdf_page_to_image(pdf_file, page_num=0, max_pages=2)
                    print(f"[DEBUG] Created {len(image_paths)} images from {pdf_file}", flush=True)
                    all_image_paths.extend(image_paths)
                
                print(f"[DEBUG] Total images created: {len(all_image_paths)}", flush=True)
                
                # Return both text and images
                if all_image_paths:
                    print(f"[DEBUG] Returning {len(all_image_paths)} images with response", flush=True)
                    # For Gradio ChatInterface with type="messages", we need to return a message dict
                    # that includes both content and file paths
                    return {
                        "role": "assistant",
                        "content": text_response,
                        "files": all_image_paths
                    }
                else:
                    print(f"[DEBUG] No images were created. Returning text only.", flush=True)
                    # Only add note if images were actually requested but failed
                    if "screenshot" in message_lower or ("provide" in message_lower and "image" in message_lower):
                        # Check if pdf2image is available
                        if not PDF2IMAGE_AVAILABLE:
                            text_response += "\n\nNote: Screenshot functionality requires pdf2image and poppler to be installed."
                        elif not POPPLER_PATH:
                            text_response += "\n\nNote: Screenshot functionality requires poppler to be installed and configured."
                        else:
                            text_response += "\n\nNote: I attempted to provide screenshots but couldn't find relevant assignment files. The images should appear automatically when discussing specific assignments."
                    return text_response
            except Exception as e:
                import traceback
                print(f"[ERROR] Error getting assignment images: {e}", flush=True)
                print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
                return text_response
        
        return text_response

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

# Build (or reuse) your Gradio app
def build_demo_with_images():
    """Build Gradio demo that supports images in chat responses."""
    me = Me()
    set_client(me.openai)

    # Build FAISS KB once (or rebuild if empty)
    try:
        n = rebuild_if_empty()
        if n == 0:
            print("WARNING: KB has 0 chunks. Check files in 'kb/'", flush=True)
    except Exception as e:
        print("KB build skipped / failed:", e, flush=True)

    def chat_wrapper(message, history):
        """Wrapper to handle both text and image responses."""
        print(f"[DEBUG] chat_wrapper called with message: {message[:50]}...", flush=True)
        result = me.chat_with_images(message, history)
        print(f"[DEBUG] chat_wrapper received result type: {type(result)}, is dict: {isinstance(result, dict)}", flush=True)
        
        if isinstance(result, dict):
            print(f"[DEBUG] Result dict keys: {result.keys()}", flush=True)
        
        # If result is a dict with files, format it for Gradio
        if isinstance(result, dict) and "files" in result:
            response_content = result["content"]
            files = result.get("files", [])
            
            # For Gradio ChatInterface, we need to return images differently
            # Gradio ChatInterface with type="messages" supports file paths in the response
            if files:
                # Remove the misleading note about pdf2image/poppler if images were created
                if "Please ensure pdf2image and poppler are installed" in response_content:
                    response_content = response_content.replace(
                        "\n\nNote: I attempted to provide screenshots from your assignments, but encountered an issue. Please ensure pdf2image and poppler are installed.",
                        ""
                    )
                
                # For Gradio, we can return a tuple with (content, files) or embed images
                # Since ChatInterface expects a string, we'll embed them as markdown
                # But first, let's try using Gradio's file component approach
                # Actually, for ChatInterface type="messages", we should return a dict
                
                # Use base64 encoding to embed images directly - this is the most reliable way
                # Gradio ChatInterface supports base64-encoded images in HTML
                print(f"[DEBUG] chat_wrapper: Formatting response with {len(files)} images using base64 encoding", flush=True)
                print(f"[DEBUG] chat_wrapper: Files to encode: {[os.path.basename(f) for f in files[:5]]}", flush=True)
                
                try:
                    from PIL import Image
                    import base64
                    from io import BytesIO
                    
                    image_section = "\n\n## 📎 Relevant Assignment Pages:\n\n"
                    
                    for i, img_path in enumerate(files[:5], 1):  # Limit to 5 images
                        if os.path.exists(img_path):
                            filename = os.path.basename(img_path)
                            print(f"[DEBUG] Encoding image {i}: {filename}", flush=True)
                            
                            try:
                                # Open and encode image as base64
                                with Image.open(img_path) as img:
                                    # Convert to RGB if necessary (handle RGBA, etc.)
                                    if img.mode in ('RGBA', 'LA', 'P'):
                                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                                        if img.mode == 'P':
                                            img = img.convert('RGBA')
                                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                                        img = rgb_img
                                    
                                    # Resize if too large (max 1200px width for better performance)
                                    max_width = 1200
                                    if img.width > max_width:
                                        ratio = max_width / img.width
                                        new_height = int(img.height * ratio)
                                        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                                    
                                    # Convert to base64
                                    buffered = BytesIO()
                                    img.save(buffered, format="PNG")
                                    img_str = base64.b64encode(buffered.getvalue()).decode()
                                    
                                    # Use Markdown image syntax with base64 - Gradio ChatInterface supports this
                                    # Markdown: ![alt](data:image/png;base64,{base64_string})
                                    image_section += f"![Assignment page {i}](data:image/png;base64,{img_str})\n\n"
                                    print(f"[DEBUG] Successfully encoded image {i} using Markdown syntax", flush=True)
                            except Exception as e:
                                print(f"[ERROR] Failed to encode image {img_path}: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                                # Fallback to file path
                                filename = os.path.basename(img_path)
                                image_url = f"/assignment_images/{filename}"
                                image_section += f'<img src="{image_url}" alt="Assignment page {i}" style="max-width: 100%; height: auto; margin: 10px 0;" />\n\n'
                        else:
                            print(f"[WARNING] Image file does not exist: {img_path}", flush=True)
                    
                    print(f"[DEBUG] Returning response with {len(files)} embedded base64 images", flush=True)
                    print(f"[DEBUG] Response length: {len(response_content + image_section)} characters", flush=True)
                    print(f"[DEBUG] Image section contains: {len(image_section)} characters", flush=True)
                    # Verify base64 encoding worked
                    if "data:image/png;base64," in image_section:
                        print(f"[DEBUG] ✓ Base64 images confirmed in response (Markdown format)", flush=True)
                        # Show a sample of the first 100 chars of base64 to verify
                        if "![Assignment page" in image_section:
                            print(f"[DEBUG] ✓ Markdown image syntax confirmed", flush=True)
                    else:
                        print(f"[WARNING] ✗ No base64 images found in response!", flush=True)
                    return response_content + image_section
                    
                except ImportError:
                    print(f"[WARNING] PIL not available, using file paths instead", flush=True)
                    # Fallback to file paths if base64 encoding fails
                    image_section = "\n\n## 📎 Relevant Assignment Pages:\n\n"
                    for i, img_path in enumerate(files[:5], 1):
                        if os.path.exists(img_path):
                            filename = os.path.basename(img_path)
                            image_url = f"/assignment_images/{filename}"
                            image_section += f'<img src="{image_url}" alt="Assignment page {i}" style="max-width: 100%; height: auto; margin: 10px 0;" />\n\n'
                    return response_content + image_section
            else:
                return response_content
        else:
            print(f"[DEBUG] Returning result as-is (not a dict with files)", flush=True)
            return result
    
    # Enable HTML rendering in Gradio ChatInterface
    # Note: Gradio ChatInterface should support HTML by default, but let's make sure
    interface = gr.ChatInterface(chat_wrapper, type="messages")
    print(f"[INFO] Gradio ChatInterface created with chat_wrapper", flush=True)
    return interface

demo = build_demo_with_images()

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

# Serve assignment images so Gradio can display them
# Make sure the directory exists and is accessible
assignment_images_abs = os.path.abspath(assignment_images_directory)
os.makedirs(assignment_images_abs, exist_ok=True)
app.mount("/assignment_images", StaticFiles(directory=assignment_images_abs), name="assignment_images")
print(f"[INFO] Assignment images will be served from: {assignment_images_abs}", flush=True)

@app.get("/", response_class=HTMLResponse)
def index():
    return (static_dir / "index.html").read_text(encoding="utf-8")

# Minimal schema for your new front-end
@app.post("/chat")
async def chat_api(payload: dict, request: Request):
    """
    Expects: {"message": "...", "history": [...]} (history is optional)
    Returns: {"reply": "..."} or {"reply": "...", "images": [...]}
    """
    message = (payload or {}).get("message", "")
    history = (payload or {}).get("history", [])
    if not isinstance(history, list):
        history = []
    
    try:
        # Use chat_with_images to get both text and potentially images
        result = _shared_me.chat_with_images(message, history)
        print(f"[DEBUG] /chat endpoint: result type: {type(result)}, is dict: {isinstance(result, dict)}", flush=True)
        
        # If result is a dict with files, format it like chat_wrapper does
        if isinstance(result, dict) and "files" in result:
            response_content = result["content"]
            files = result.get("files", [])
            
            print(f"[DEBUG] /chat endpoint: Formatting {len(files)} images", flush=True)
            
            # Format images the same way as chat_wrapper
            if files:
                try:
                    from PIL import Image
                    import base64
                    from io import BytesIO
                    
                    image_section = "\n\n## 📎 Relevant Assignment Pages:\n\n"
                    
                    for i, img_path in enumerate(files[:5], 1):
                        if os.path.exists(img_path):
                            filename = os.path.basename(img_path)
                            print(f"[DEBUG] /chat endpoint: Encoding image {i}: {filename}", flush=True)
                            
                            try:
                                with Image.open(img_path) as img:
                                    # Convert to RGB if necessary
                                    if img.mode in ('RGBA', 'LA', 'P'):
                                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                                        if img.mode == 'P':
                                            img = img.convert('RGBA')
                                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                                        img = rgb_img
                                    
                                    # Resize if too large
                                    max_width = 1200
                                    if img.width > max_width:
                                        ratio = max_width / img.width
                                        new_height = int(img.height * ratio)
                                        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                                    
                                    # Convert to base64
                                    buffered = BytesIO()
                                    img.save(buffered, format="PNG")
                                    img_str = base64.b64encode(buffered.getvalue()).decode()
                                    
                                    # Use Markdown image syntax with base64
                                    image_section += f"![Assignment page {i}](data:image/png;base64,{img_str})\n\n"
                                    print(f"[DEBUG] /chat endpoint: Successfully encoded image {i} using Markdown syntax", flush=True)
                            except Exception as e:
                                print(f"[ERROR] /chat endpoint: Failed to encode image {img_path}: {e}", flush=True)
                                # Fallback
                                filename = os.path.basename(img_path)
                                image_url = f"/assignment_images/{filename}"
                                image_section += f'<img src="{image_url}" alt="Assignment page {i}" style="max-width: 100%; height: auto; margin: 10px 0;" />\n\n'
                        else:
                            print(f"[WARNING] /chat endpoint: Image file does not exist: {img_path}", flush=True)
                    
                    reply = response_content + image_section
                    print(f"[DEBUG] /chat endpoint: Returning reply with embedded images", flush=True)
                    return JSONResponse({"reply": reply})
                    
                except ImportError:
                    print(f"[WARNING] /chat endpoint: PIL not available, using file paths", flush=True)
                    # Fallback
                    reply = response_content
                    return JSONResponse({"reply": reply, "images": files[:5]})
            else:
                reply = response_content
        else:
            reply = result if isinstance(result, str) else str(result)
    except Exception as e:
        import traceback
        print(f"[ERROR] /chat endpoint error: {e}", flush=True)
        print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
        reply = f"Sorry, something went wrong: {e}"
    
    return JSONResponse({"reply": reply})
