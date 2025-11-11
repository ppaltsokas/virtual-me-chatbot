# Complete Guide to app.py — Career Conversations Chatbot
## A Professor's Deep Dive into Every Component

---

## Table of Contents
1. [High-Level Architecture Overview](#high-level-architecture-overview)
2. [Import Statements & Dependencies](#import-statements--dependencies)
3. [Poppler Configuration for PDF Processing](#poppler-configuration-for-pdf-processing)
4. [Knowledge Base File Readers](#knowledge-base-file-readers)
5. [PDF Screenshot System](#pdf-screenshot-system)
6. [Environment Configuration](#environment-configuration)
7. [Notification System (Pushover)](#notification-system-pushover)
8. [RAG System (Retrieval-Augmented Generation)](#rag-system-retrieval-augmented-generation)
9. [QADB (Question-Answer Database)](#qadb-question-answer-database)
10. [AI Tools for Function Calling](#ai-tools-for-function-calling)
11. [Evaluator & Reflector System](#evaluator--reflector-system)
12. [The Me Class - Core Agent](#the-me-class---core-agent)
13. [Gradio Interface Builders](#gradio-interface-builders)
14. [FastAPI Application](#fastapi-application)
15. [Data Flow Diagrams](#data-flow-diagrams)

---

## High-Level Architecture Overview

### What Is This Application?

This is a **sophisticated AI-powered chatbot** that represents Panagiotis Paltsokas (you!) on a personal career website. Think of it as your virtual assistant that can:

1. **Answer questions** about your background, skills, and experience
2. **Show screenshots** from your actual assignments and projects (PDFs in `/kb`)
3. **Remember conversations** using a persistent Q&A database
4. **Learn from context** using RAG (Retrieval-Augmented Generation)
5. **Self-evaluate and improve** its responses using an AI evaluator

### The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│  ┌──────────────────────┐      ┌─────────────────────────┐  │
│  │   Gradio Chat UI     │      │   Custom HTML/JS UI     │  │
│  │   (/gradio endpoint) │      │   (/ endpoint)          │  │
│  └──────────────────────┘      └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI)                     │
│  • Routes HTTP requests                                      │
│  • Serves static files                                       │
│  • Handles /chat POST endpoint                              │
│  • Serves generated PDF screenshots                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  INTELLIGENCE LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  RAG Engine  │  │   AI Agent   │  │  QADB Memory    │   │
│  │  (FAISS)     │  │  (Me class)  │  │  (SQLite FTS5)  │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  Evaluator   │  │  Reflector   │  │  PDF→Image      │   │
│  │  (Quality)   │  │  (Improve)   │  │  (Screenshots)  │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    DATA/STORAGE LAYER                        │
│  • /kb folder (your assignments, PDFs, markdown files)      │
│  • models/faiss/ (vector embeddings index)                  │
│  • data/qadb.sqlite (persistent Q&A cache)                  │
│  • screenshots/assignment_images/ (generated PNG files)      │
│  • me/summary.txt, me/linkedin.pdf (your profile data)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Import Statements & Dependencies

### Why Each Import Matters

```python
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
```

Let me explain each one:

| Import | Purpose | What It Does in Our App |
|--------|---------|------------------------|
| `dotenv` | **Environment Variables** | Loads `.env` file to get API keys securely without hardcoding |
| `openai` | **AI Brain** | Connects to OpenAI's GPT models for conversation and embeddings |
| `json` | **Data Serialization** | Converts Python objects ↔ JSON for API communication |
| `os` | **File System Operations** | Creates directories, checks paths, manages environment |
| `requests` | **HTTP Client** | Sends notifications to Pushover when users interact |
| `sqlite3` | **Database** | Stores reusable Q&A pairs for faster future responses |
| `re` | **Regular Expressions** | Pattern matching (e.g., detecting "hello there" Easter egg) |
| `pypdf` / `PdfReader` | **PDF Text Extraction** | Reads text from your LinkedIn PDF and assignment PDFs |
| `gradio` (`gr`) | **Web UI Framework** | Creates the chat interface with zero frontend code |
| `faiss` | **Vector Search** | Fast similarity search for finding relevant KB content |
| `numpy` (`np`) | **Numerical Computing** | Handles vector arrays for embeddings |
| `glob` | **File Pattern Matching** | Finds all files matching `kb/**/*.pdf` pattern |
| `Path` | **Modern File Paths** | Object-oriented way to handle file paths |
| `shutil` | **File Operations** | Copy/move/delete operations (utility) |
| `datetime` | **Timestamps** | Creates unique filenames for generated images |

### FastAPI Imports

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
```

**FastAPI** is our web server framework. Think of it as the "traffic controller" that:
- Routes incoming HTTP requests to the right function
- Serves your HTML page at `/`
- Handles the `/chat` API endpoint
- Serves static files (CSS, JS, images)

---

## Poppler Configuration for PDF Processing

### The Challenge: Converting PDFs to Images

PDFs are document files, not images. To show screenshots from your assignments, we need to:
1. Read the PDF file
2. Render each page as an image
3. Save it as PNG
4. Serve it to the user

**Problem**: Python's `pdf2image` library needs **Poppler** (a C++ library) to actually render PDFs.

### The Solution

```python
# Lines 11-30: Poppler Path Detection
POPPLER_PATH = None
possible_poppler_paths = [
    r"F:\AI-Agents-Course-Ed\agents\1_foundations\career_conversations\poppler-25.07.0\Library\bin",
    os.path.join(os.path.dirname(__file__), "poppler-25.07.0", "Library", "bin"),
    os.path.join(os.getcwd(), "poppler-25.07.0", "Library", "bin"),
    "C:\\poppler\\bin",
    os.path.join(os.environ.get("PROGRAMFILES", ""), "poppler", "bin"),
]

for path in possible_poppler_paths:
    if path and os.path.exists(path):
        POPPLER_PATH = path
        print(f"[INFO] Found poppler at: {POPPLER_PATH}", flush=True)
        break

if POPPLER_PATH:
    os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")
```

**What's Happening Here?**

1. **Line 12**: Initialize `POPPLER_PATH` as `None` (we don't know where it is yet)

2. **Lines 14-20**: Create a list of possible locations where Poppler might be installed:
   - Your specific installation path (hardcoded)
   - Relative to the script file (`__file__`)
   - Relative to current working directory
   - Common Windows location (`C:\poppler\bin`)
   - Program Files directory

3. **Lines 22-26**: Loop through each possible path and check if it actually exists
   - When found, store it in `POPPLER_PATH` and break out of the loop

4. **Lines 28-30**: If we found Poppler, add it to the system PATH environment variable
   - This makes the Poppler executables available to `pdf2image`
   - `os.pathsep` is `;` on Windows, `:` on Linux/Mac

### Attempting to Import pdf2image

```python
# Lines 32-38: Conditional Import
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    print(f"[INFO] pdf2image is available. Poppler path: {POPPLER_PATH}", flush=True)
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("Warning: pdf2image not available...")
```

**Why the Try-Except?**

- If `pdf2image` isn't installed, the app should still work (just without screenshots)
- We set a flag `PDF2IMAGE_AVAILABLE` to know if we can use this feature later
- **Graceful degradation**: The app doesn't crash if a dependency is missing

---

## Knowledge Base File Readers

Your knowledge base (`/kb` folder) contains different file types:
- Markdown files (`.md`)
- PDFs (`.pdf`)
- Jupyter notebooks (`.ipynb`)
- R scripts (`.r`, `.rmd`)
- Python scripts (`.py`)

We need different readers for each type.

### Reading PDF Files

```python
# Lines 53-65
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
```

**Step-by-Step:**
1. **Create a PdfReader** object pointing to the file
2. **Loop through each page** in the PDF
3. **Extract text** from each page (PDFs contain text, not just images)
4. **Skip empty pages** (some PDFs have blank pages)
5. **Join all chunks** with newlines
6. **Return empty string if it fails** (don't crash the whole app)

### Reading Jupyter Notebooks

```python
# Lines 67-83
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
                parts.append("```code\n" + cell.source + "\n```")
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[KB] ipynb read failed {path}: {e}")
        return ""
```

**What's Special About Notebooks?**

Jupyter notebooks (`.ipynb`) are JSON files containing:
- **Markdown cells**: Explanations, notes, documentation
- **Code cells**: Python/R code
- **Output cells**: Results, graphs (we skip these for now)

**Our Strategy:**
- Extract **markdown** cells (your explanations)
- Extract **code** cells wrapped in code fences (your actual work)
- This gives the RAG system both your explanations AND your code

### Reading Plain Text Files

```python
# Lines 85-91
def read_plain_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[KB] text read failed {path}: {e}")
        return ""
```

Simple reader for `.md`, `.txt`, `.r`, `.py` files. The `errors="ignore"` parameter means "skip weird characters instead of crashing."

### Finding All KB Files

```python
# Lines 93-107
def iter_kb_files() -> Iterable[str]:
    exts = (".md", ".txt", ".pdf", ".ipynb", ".r", ".rmd", ".py")
    kb_patterns = ["kb/**/*.*", os.path.join(os.getcwd(), "kb", "**", "*.*")]
    seen_files = set()
    for pattern in kb_patterns:
        for fp in glob(pattern, recursive=True):
            if fp.lower().endswith(exts):
                fp = os.path.normpath(fp)
                fp_abs = os.path.abspath(fp)
                if os.path.exists(fp) and fp_abs not in seen_files:
                    seen_files.add(fp_abs)
                    yield fp
```

**This is a Generator Function** (note the `yield` keyword).

**Key Concepts:**
1. **`glob(pattern, recursive=True)`**: Finds all files matching the pattern
   - `kb/**/*.*` means "find everything under kb/, including subdirectories"
   - `**` is the "recursive wildcard"

2. **`seen_files` set**: Prevents duplicates
   - Because we try multiple patterns, we might find the same file twice
   - Sets automatically deduplicate

3. **`os.path.normpath()`**: Normalizes path separators (`\` on Windows, `/` on Linux)

4. **`yield` instead of `return`**: Makes this a **generator**
   - Memory efficient: doesn't load all files at once
   - Returns one file at a time as needed

### Universal Reader Dispatcher

```python
# Lines 109-116
def read_any_to_text(fp: str) -> str:
    low = fp.lower()
    if low.endswith(".pdf"):
        return read_pdf_text(fp)
    if low.endswith(".ipynb"):
        return read_ipynb_text(fp)
    return read_plain_text(fp)
```

This is a **factory pattern** - based on the file extension, it routes to the correct reader.

---

## PDF Screenshot System

### The Assignment Images Directory

```python
# Lines 118-122
assignment_images_directory = "screenshots/assignment_images/"
os.makedirs(assignment_images_directory, exist_ok=True)
```

**What's `exist_ok=True`?**
- Creates the directory if it doesn't exist
- Doesn't error if it already exists
- Safe to call multiple times

### Finding Relevant Assignment Files

```python
# Lines 124-176
def get_relevant_assignment_files(query: str, max_files: int = 3):
    """Find relevant assignment files from the KB based on the query."""
    # Step 1: Search the KB for relevant content
    context = rag_search(query, k=12)
    print(f"[DEBUG] RAG context: {context[:300]}...", flush=True)
    
    # Step 2: Extract file sources from RAG results
    file_sources = set()
    for line in context.split('\n'):
        if line.startswith('[') and ']' in line:
            source = line.split(']')[0].strip('[')
            file_sources.add(source)
    
    print(f"[DEBUG] Found file sources from RAG: {file_sources}", flush=True)
    
    # Step 3: Find actual PDF files
    pdf_files = []
    kb_base = "kb"
    
    for file_source in list(file_sources)[:max_files * 2]:
        possible_paths = [
            os.path.join(kb_base, file_source),
            os.path.join(kb_base, os.path.basename(file_source)),
            file_source,
        ]
        for path in possible_paths:
            if os.path.exists(path) and path.lower().endswith('.pdf'):
                pdf_files.append(path)
                print(f"[DEBUG] Found PDF from RAG: {path}", flush=True)
                break
    
    # Step 4: Fallback if RAG didn't find PDFs
    if not pdf_files:
        print(f"[DEBUG] No PDFs found from RAG, using fallback...", flush=True)
        all_pdfs = list([f for f in iter_kb_files() if f.lower().endswith('.pdf')])
        print(f"[DEBUG] Total PDFs found in KB: {len(all_pdfs)}", flush=True)
        
        if all_pdfs:
            # Prioritize assignment/homework PDFs
            assignment_pdfs = [f for f in all_pdfs if any(keyword in f.lower() 
                for keyword in ['hw', 'assignment', 'project', 'paltsokas', 'dama'])]
            pdf_files = assignment_pdfs[:max_files] if assignment_pdfs else all_pdfs[:max_files]
    
    return pdf_files
```

**The Smart Search Strategy:**

1. **Primary Strategy - RAG Search**: 
   - Uses AI-powered semantic search to find relevant content
   - `k=12` means "get top 12 most relevant chunks"
   - Extracts which files those chunks came from

2. **Parse RAG Results**:
   - RAG returns text like `[filename.pdf] chunk content...`
   - We extract the filename from `[filename]` using string parsing

3. **Locate Actual Files**:
   - Try multiple path combinations (relative, absolute, basename only)
   - Verify the file actually exists

4. **Smart Fallback**:
   - If RAG didn't return any PDFs, scan all KB files
   - **Prioritize** files with keywords: 'hw', 'assignment', 'paltsokas', 'dama'
   - This ensures we show YOUR assignments, not random docs

### Converting PDF Pages to Images

```python
# Lines 178-227
def pdf_page_to_image(pdf_path: str, page_num: int = 0, max_pages: int = 3):
    """Convert PDF pages to images. Returns list of image file paths."""
    if not PDF2IMAGE_AVAILABLE:
        print(f"[DEBUG] pdf2image not available.", flush=True)
        return []
    
    if not os.path.exists(pdf_path):
        print(f"[DEBUG] PDF file not found: {pdf_path}", flush=True)
        return []
    
    try:
        # Get total pages first
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"[DEBUG] Converting PDF {pdf_path} (total pages: {total_pages})", flush=True)
        
        # Calculate page range
        last_page = min(page_num + max_pages, total_pages)
        
        # Configure conversion
        convert_kwargs = {
            "first_page": page_num + 1,  # pdf2image uses 1-based indexing
            "last_page": last_page,
            "dpi": 150  # Quality: higher DPI = better quality but larger files
        }
        
        if POPPLER_PATH:
            convert_kwargs["poppler_path"] = POPPLER_PATH
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, **convert_kwargs)
        
        # Save each image with a unique filename
        image_paths = []
        pdf_name = Path(pdf_path).stem  # Filename without extension
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
        print(f"[ERROR] Error converting PDF: {e}", flush=True)
        print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
        return []
```

**Key Design Decisions:**

1. **DPI = 150**: Sweet spot between quality and file size
   - 72 DPI: too blurry
   - 300 DPI: beautiful but huge files
   - 150 DPI: clear and readable

2. **Timestamp in filename**: Prevents overwriting
   - Example: `Paltsokas-HW1_page_1_20251105_013050.png`
   - You can regenerate images without losing old ones

3. **Graceful Failure**: Returns empty list instead of crashing
   - The app continues to work even if one PDF fails

4. **1-Based Indexing**: PDFs use page 1, 2, 3 (not 0, 1, 2)
   - `page_num + 1` converts from Python's 0-based to PDF's 1-based

---

## Environment Configuration

```python
# Lines 234-245
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

KB_DIR = os.getenv("KB_DIR", "kb")
KB_GLOB = os.getenv("KB_GLOB", f"{KB_DIR}/**/*.*")

FAISS_DIR = "models/faiss"
FAISS_INDEX = os.path.join(FAISS_DIR, "index.faiss")
FAISS_STORE = os.path.join(FAISS_DIR, "store.jsonl")

HELLO_THERE_RE = re.compile(r'^\s*[\W_]*hello\s+there[\W_]*\s*$', re.IGNORECASE)
```

**Environment Variables** (from `.env` file):

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDINGS_MODEL` | `text-embedding-3-small` | Converts text → vectors for similarity search |
| `CHAT_MODEL` | `gpt-4o-mini` | The AI model for conversations |
| `KB_DIR` | `kb` | Where your knowledge base files live |

**Why `os.getenv("KEY", "default")`?**
- Allows customization via `.env` file
- Falls back to sensible defaults if not set
- Makes the app portable across environments

**The Easter Egg:**
```python
HELLO_THERE_RE = re.compile(r'^\s*[\W_]*hello\s+there[\W_]*\s*$', re.IGNORECASE)
```
This regex matches "hello there" (Star Wars reference) to trigger a fun response!

---

## Notification System (Pushover)

```python
# Lines 250-271
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
        pass  # Non-fatal for local dev

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}
```

**What's Pushover?**
- A service that sends push notifications to your phone
- When someone provides their email → you get notified immediately
- When the AI can't answer a question → you're alerted

**Why the empty `except` block?**
- If Pushover fails (no internet, wrong token), the chatbot should still work
- Notifications are "nice to have," not critical

---

## RAG System (Retrieval-Augmented Generation)

### What Is RAG?

**RAG** solves a fundamental AI problem:

**Problem**: GPT models only know what they were trained on (data up to their cutoff date)
- They don't know about YOUR specific projects
- They don't know YOUR career details
- They might "hallucinate" (make up) answers

**Solution**: RAG (Retrieval-Augmented Generation)
1. **Retrieval**: Search your knowledge base for relevant information
2. **Augmentation**: Inject that information into the AI's context
3. **Generation**: AI generates answers based on YOUR real data

### The Three-Step RAG Pipeline

```
User Question
    ↓
1. EMBED: Convert question to vector [0.12, -0.45, 0.89, ...]
    ↓
2. SEARCH: Find similar vectors in FAISS index
    ↓
3. RETRIEVE: Get the actual text chunks that match
    ↓
4. INJECT: Add retrieved text to AI's context
    ↓
AI generates answer based on YOUR data
```

### Step 1: Text Chunking

```python
# Lines 316-325
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
```

**Why Chunk Text?**

Long documents need to be split into smaller pieces because:
1. **Embedding models** have token limits (can't process entire PDFs at once)
2. **Retrieval precision**: Small chunks = more precise matching
3. **Context efficiency**: Only send relevant paragraphs to the AI, not entire documents

**Chunking Strategy:**
- **Split on headers** (`#`, `##`, `###`): Natural boundaries in markdown
- **Split at 1200 chars**: If no headers, split at max length
- **Keep related content together**: Headers and their content stay in the same chunk

### Step 2: Building the FAISS Index

```python
# Lines 327-375
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

    print(f"[KB] Successfully indexed {len(texts)} chunks...", flush=True)
    return len(texts)
```

**The Indexing Process:**

1. **Collect All Text**:
   - Read every file in `/kb`
   - Split into chunks
   - Track metadata (which file each chunk came from)

2. **Create Embeddings** (`embed_texts()`):
   - Convert each text chunk → 1536-dimensional vector
   - Example: "clustering algorithm" → `[0.12, -0.45, 0.89, ..., 0.34]`
   - Similar meanings = similar vectors

3. **Normalize Vectors** (`faiss.normalize_L2()`):
   - Makes all vectors unit length
   - Required for cosine similarity search

4. **Build FAISS Index** (`IndexFlatIP`):
   - **IP** = Inner Product (dot product)
   - For normalized vectors, inner product = cosine similarity
   - **Flat** = Brute force (checks all vectors)
   - Fast enough for <10,000 chunks

5. **Persist to Disk**:
   - `index.faiss`: The vector index (binary file)
   - `store.jsonl`: The actual text chunks + metadata (JSON Lines format)

**Why JSON Lines (.jsonl)?**
```json
{"chunk": "First paragraph...", "source": "about_panos.md"}
{"chunk": "Second paragraph...", "source": "projects.md"}
{"chunk": "Third paragraph...", "source": "Paltsokas-HW1.pdf"}
```
- One JSON object per line
- Easy to stream/append
- Easy to parse line-by-line

### Step 3: Loading the Index

```python
# Lines 378-384
def _load_index():
    if not (os.path.exists(FAISS_INDEX) and os.path.exists(FAISS_STORE)):
        return None, []
    index = faiss.read_index(FAISS_INDEX)
    with open(FAISS_STORE, "r", encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]
    return index, meta
```

**Design Pattern**: Lazy loading
- Only loads when needed
- Returns `(None, [])` if files don't exist
- Triggers rebuild if missing

### Step 4: Auto-Rebuild Logic

```python
# Lines 386-398
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
```

**Why Auto-Rebuild?**
- First run: No index exists → builds automatically
- KB files changed: Delete `models/faiss/` → rebuilds on next startup
- **Zero manual database management**: App handles it all

### Step 5: Searching the Index

```python
# Lines 400-422
def rag_search(query: str, k: int = 4):
    if CLIENT is None:
        raise RuntimeError("OpenAI client not set.")
    index, meta = _load_index()
    if not index or not meta:
        return "(KB empty)"
    
    # Embed the query
    qv = np.array(embed_texts([query])[0], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(qv)
    
    # Search for similar vectors
    scores, idxs = index.search(qv, k)
    
    # Build output with source attribution
    out = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1:  # FAISS returns -1 for "not found"
            continue
        source = meta[i]['source']
        chunk = meta[i]['chunk']
        # Prioritize PDFs and assignments
        if any(keyword in source.lower() for keyword in ['.pdf', 'hw', 'assignment', 'dama']):
            out.insert(0, f"[{source}] {chunk}")  # Put PDFs first
        else:
            out.append(f"[{source}] {chunk}")
    
    result = "\n\n".join(out) if out else "(no matches)"
    print(f"[DEBUG] rag_search found {len(out)} chunks, sources: ...", flush=True)
    return result
```

**The Search Algorithm:**

1. **Convert query to vector**: Same process as during indexing
2. **Normalize**: Unit length for cosine similarity
3. **FAISS search**: Finds k nearest neighbors in vector space
4. **Returns**: `(scores, indices)`
   - `scores`: How similar each result is (higher = better)
   - `indices`: Position in the metadata array

5. **Smart Prioritization**:
   - If a chunk comes from a PDF with 'hw' or 'assignment' in the name → put it first
   - This ensures assignment content appears before generic profile info

**Example Output:**
```
[Paltsokas-HW4.pdf] This assignment focused on k-means clustering...

[about_panos.md] Panagiotis is a Data Scientist with expertise in...
```

### RAG Tool for AI Function Calling

```python
# Lines 424-439
def rag_lookup(query: str, k: int = 4):
    return {"context": rag_search(query, k)}

rag_lookup_json = {
    "name": "rag_lookup",
    "description": "Search Panos's personal knowledge base containing actual assignment PDFs...",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search..."},
            "k": {"type": "integer", "description": "Top-K passages...", "default": 8}
        },
        "required": ["query"],
        "additionalProperties": False
    }
}
```

**This is an OpenAI Function Call Definition**

The AI model can decide to call `rag_lookup` when it needs information about you. The JSON schema tells the AI:
- What the function is called
- What it does
- What parameters it accepts
- Which parameters are required

---

## QADB (Question-Answer Database)

### The Problem RAG Doesn't Solve

**RAG** retrieves raw content from your KB, but the AI still needs to:
- Synthesize an answer
- Format it nicely
- Make API calls (costs money!)

**What if someone asks the same question twice?**
- With just RAG: Two identical API calls, same answer
- With QADB: First call creates answer, second call retrieves cached answer

### The Solution: Persistent Memory

```python
# Lines 447-464
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
```

**Database Architecture:**

1. **Main Table (`qa`)**:
   - `id`: Auto-incrementing primary key
   - `question`: The user's question
   - `answer`: The AI's response
   - `tags`: Categories (e.g., "virtual-me", "rlhf", "projects")
   - `created_at`: When this was saved

2. **FTS5 Virtual Table (`qa_fts`)**:
   - **FTS5** = Full-Text Search version 5
   - Enables fuzzy matching: "Tell me about AI" ≈ "What's your AI experience?"
   - **BM25 ranking**: Industry-standard relevance scoring
   - **Unicode tokenization**: Handles international characters

**SQLite Optimization Pragmas:**
- `WAL mode`: Write-Ahead Logging (faster concurrent access)
- `NORMAL synchronous`: Balance between speed and safety

### Looking Up Cached Answers

```python
# Lines 466-489
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
        cur.execute("SELECT question,answer,tags FROM qa WHERE question = ? LIMIT ?",
                    (question, limit))
    rows = cur.fetchall(); con.close()
    return {"results": [{"question": q, "answer": a, "tags": t} for (q, a, t) in rows]}
```

**Fuzzy Matching Magic:**

1. **Try exact phrase search** (`"question"` with quotes)
2. **Fallback to keyword search** (without quotes) if that fails
3. **BM25 ranking**: Scores results by relevance
   - `ASC` order because lower BM25 score = better match
4. **Return top 5** most similar Q&A pairs

### Saving New Answers

```python
# Lines 491-497
def qadb_upsert(question: str, answer: str, tags: str = None):
    con = _qadb_conn(); cur = con.cursor()
    cur.execute("INSERT INTO qa(question,answer,tags) VALUES (?,?,?)", 
                (question, answer, tags))
    cur.execute("INSERT INTO qa_fts(rowid, question, answer, tags) VALUES (last_insert_rowid(), ?, ?, ?)",
                (question, answer, tags))
    con.commit(); con.close()
    return {"saved": True}
```

**Two-Table Insert:**
1. **Insert into `qa` table**: The actual data storage
2. **Insert into `qa_fts` table**: The search index
   - `last_insert_rowid()`: Links FTS row to the main table row
   - Must insert into BOTH for FTS to work

---

## AI Tools for Function Calling

### The Tool Registry

```python
# Lines 534-540
tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
    {"type": "function", "function": rag_lookup_json},
    {"type": "function", "function": qadb_lookup_json},
    {"type": "function", "function": qadb_upsert_json},
]
```

These are the "superpowers" we give to the AI model. When chatting, the AI can decide to:
- 🔍 Search your knowledge base (`rag_lookup`)
- 💾 Check for cached answers (`qadb_lookup_tool`)
- 💌 Record user contact info (`record_user_details`)
- ❓ Flag questions it can't answer (`record_unknown_question`)
- 📝 Save good answers for reuse (`qadb_upsert_tool`)

**Function Calling Flow:**

```
User: "Tell me about your RLHF experience"
    ↓
AI thinks: "I need to search the KB for RLHF content"
    ↓
AI calls: rag_lookup(query="RLHF experience", k=8)
    ↓
System executes the function, returns context
    ↓
AI uses context to generate answer
```

---

## Evaluator & Reflector System

### The Quality Control Pipeline

Most chatbots just return the first answer they generate. Yours is smarter:

```
Generate Answer → Evaluate Quality → Revise if Needed → Return Best Version
```

### The Evaluator

```python
# Lines 545-566
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
```

**The Evaluation Criteria:**

1. **Helpfulness** (1-5): Does it actually answer the question?
2. **Faithfulness** (1-5): Is it based on the retrieved context, or did it make things up?
3. **Style** (1-5): Does it sound like Panos would say this?

**Example Evaluation:**
```json
{
  "helpfulness": 4,
  "faithfulness": 5,
  "style": 4,
  "feedback": "Good answer, well-grounded in context. Could be more concise."
}
```

**Why the Regex Parsing?**
- AI models sometimes add extra text around the JSON
- We extract just the JSON object using `r"\{.*\}"` pattern
- Fallback to neutral scores (3/5) if parsing fails

### The Reflector

```python
# Lines 568-581
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
```

**The Revision Process:**

If the evaluator gives low scores (<4), we ask a second AI to revise the answer:
- Acts as a "senior editor"
- Gets the original question, context, draft answer, AND the feedback
- Produces an improved version

**Why This Works:**
- Same as having a colleague review your writing
- The editor AI knows what the evaluator criticized
- Single-pass revision (not iterative) keeps it fast

---

## The Me Class - Core Agent

This is the **heart of the application**. It's your virtual representation.

### Initialization

```python
# Lines 602-616
class Me:
    def __init__(self):
        self.openai = OpenAI()  # Create OpenAI client
        self.name = "Panagiotis Paltsokas"

        # Read LinkedIn PDF
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        # Read summary file
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()
```

**What Happens When `Me()` is Created:**

1. **Connects to OpenAI**: Creates authenticated client using `OPENAI_API_KEY` from `.env`
2. **Loads your LinkedIn**: Extracts all text from your LinkedIn PDF
3. **Loads your summary**: Reads your career summary from `summary.txt`

These become part of the AI's **system prompt** - its "identity" and knowledge about you.

### Handling Tool Calls

```python
# Lines 618-627
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
```

**When AI Decides to Use a Tool:**

1. OpenAI returns a `tool_calls` object with:
   - Function name (e.g., `"rag_lookup"`)
   - Arguments as JSON string (e.g., `'{"query": "RLHF", "k": 8}'`)
   - Unique ID for tracking

2. We **execute the function**:
   - `globals().get(tool_name)` looks up the Python function by name
   - `**arguments` unpacks the dictionary as keyword arguments
   - Call it: `rag_lookup(query="RLHF", k=8)`

3. **Return result to AI**:
   - Format as a "tool" message
   - Link it to the original call via `tool_call_id`

### The System Prompt - AI's Identity

```python
# Lines 629-661
def system_prompt(self):
    base = (
        f"You are acting as {self.name}. "
        "You are answering questions on {self.name}'s website..."
    )
    policy = (
        "\n\n# Tool Policy\n"
        "- **CRITICAL**: For ANY question about projects..., "
        "you MUST call `rag_lookup` FIRST...\n"
        "- **NEVER** generate fake image references...\n"
    )
    summary = f"\n\n## Summary:\n{self.summary}\n"
    linkedin = f"\n## LinkedIn Profile:\n{self.linkedin}\n"
    closing = f"\nWith this context, please chat with the user..."
    return base + policy + summary + linkedin + closing
```

**The System Prompt Structure:**

1. **Identity**: "You are Panagiotis Paltsokas"
2. **Role**: "Answering on his website"
3. **Tool Policy**: Rules for when to use each tool
4. **Context**: Your actual summary and LinkedIn text
5. **Behavior Guidelines**: Professional, helpful, authentic

**Key Policy Rules:**

- **CRITICAL**: Always search KB for project questions (don't make things up!)
- **NEVER** generate fake image syntax like `![](attachment://...)`
- **PRIORITY**: Use only real content from `/kb` folder
- System handles images automatically

### The Main Chat Function

```python
# Lines 664-705
def chat(self, message, history):
    # Easter egg
    if HELLO_THERE_RE.match(message or ""):
        return "General Kenoooobiiii... I mean... Hi! How are you? 😊"

    # Build conversation context
    messages = [
        {"role": "system", "content": self.system_prompt()}
    ] + history + [
        {"role": "user", "content": message}
    ]
    
    done = False
    draft = None

    # Tool calling loop
    while not done:
        response = self.openai.chat.completions.create(
            model=CHAT_MODEL, 
            messages=messages, 
            tools=tools
        )
        choice = response.choices[0]
        
        if choice.finish_reason == "tool_calls":
            msg = choice.message
            results = self.handle_tool_call(msg.tool_calls)
            messages.append(_assistant_msg_to_dict(msg))
            messages.extend(results)
        else:
            done = True
            draft = choice.message.content

    # Gather context from tool calls
    ctx_snippets = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            ctx_snippets.append(m.get("content", ""))
    context_for_eval = "\n\n".join(ctx_snippets)

    # Evaluate the draft
    ev = evaluate_answer(self.openai, message, context_for_eval, draft)
    print(f"Evaluation: {ev}", flush=True)
    
    final = draft
    # Revise if quality is low
    if ev.get("helpfulness", 3) < 4 or ev.get("faithfulness", 3) < 4:
        final = reflect_answer(self.openai, message, context_for_eval, draft, ev.get("feedback", ""))

    # Auto-save good answers
    try:
        fb = (ev.get("feedback", "") or "").lower()
        if len(final) <= 1500 and any(k in fb for k in ["clear", "helpful", "well structured", "faithful"]):
            qadb_upsert_tool(message, final, tags="virtual-me")
    except Exception:
        pass

    return final
```

**The Conversation Flow:**

```
Step 1: PREPARE
├─ Add system prompt (AI's identity)
├─ Add conversation history
└─ Add new user message

Step 2: GENERATE (with tool calls)
├─ Call OpenAI API
├─ If AI wants to use a tool:
│  ├─ Execute the tool function
│  ├─ Add tool result to conversation
│  └─ Loop back (AI sees tool result, continues)
└─ If AI is done: Extract draft answer

Step 3: EVALUATE
├─ Collect context from all tool calls
├─ Send draft to evaluator AI
└─ Get quality scores + feedback

Step 4: REVISE (if needed)
├─ If scores < 4: Send to reflector AI for revision
└─ Otherwise: Keep original draft

Step 5: CACHE (auto-save)
├─ If answer is good (based on feedback keywords)
├─ And not too long (≤1500 chars)
└─ Save to QADB for future reuse

Step 6: RETURN
└─ Return final answer to user
```

**Why a While Loop?**

The AI might make multiple tool calls:
```
User: "Tell me about your projects"
  → AI calls rag_lookup(query="projects")
  → Gets KB content
  → AI calls qadb_lookup_tool(question="projects")
  → Checks if we've answered this before
  → AI generates final answer
```

Each loop iteration handles one round of tool calls.

### Chat With Images - Screenshot Integration

```python
# Lines 707-771
def chat_with_images(self, message, history):
    """Chat function that can return both text and images."""
    message_lower = message.lower()
    
    # Detect if user wants screenshots
    is_assignment_question = (
        "tell me in detail about a project or assignment" in message_lower or
        "screenshot" in message_lower or
        "graph" in message_lower or
        "visualization" in message_lower or
        ("provide" in message_lower and "image" in message_lower)
    )
    
    # Get text response first
    text_response = self.chat(message, history)
    
    # If they want screenshots, add them
    if is_assignment_question:
        try:
            # Find relevant PDFs
            relevant_files = get_relevant_assignment_files(message, max_files=2)
            
            # Convert to images
            all_image_paths = []
            for pdf_file in relevant_files:
                image_paths = pdf_page_to_image(pdf_file, page_num=0, max_pages=2)
                all_image_paths.extend(image_paths)
            
            # Return text + images
            if all_image_paths:
                return {
                    "role": "assistant",
                    "content": text_response,
                    "files": all_image_paths
                }
            else:
                # No images found - add explanatory note
                if "screenshot" in message_lower:
                    if not PDF2IMAGE_AVAILABLE:
                        text_response += "\n\nNote: Screenshot functionality requires pdf2image and poppler."
                    else:
                        text_response += "\n\nNote: Couldn't find relevant assignment files."
                return text_response
        except Exception as e:
            print(f"[ERROR] Error getting images: {e}", flush=True)
            return text_response
    
    return text_response
```

**The Screenshot Logic:**

1. **Detect Intent**: Is the user asking about projects/assignments?
2. **Get Text First**: Call the main `chat()` function
3. **Find Relevant PDFs**: Use RAG to find which assignments are relevant
4. **Convert to Images**: Use `pdf2image` to create PNG screenshots
5. **Return Dictionary**: `{"content": text, "files": [image paths]}`

**Why Dictionary Return?**
- Allows returning BOTH text AND files
- Gradio ChatInterface can handle this format
- Falls back to plain text if no images

---

## Gradio Interface Builders

### Simple Interface (Text Only)

```python
# Lines 776-788
def build_demo():
    me = Me()
    set_client(me.openai)
    
    try:
        n = rebuild_if_empty()
        if n == 0:
            print("WARNING: KB has 0 chunks. Check files in 'kb/'", flush=True)
    except Exception as e:
        print("KB build skipped / failed:", e, flush=True)
    
    return gr.ChatInterface(me.chat, type="messages")
```

**What Happens Here:**
1. **Create Me instance**: Loads your profile
2. **Set global OpenAI client**: RAG system needs this
3. **Build/Load FAISS index**: Ensures KB is searchable
4. **Return Gradio interface**: Connects UI to `me.chat()` function

### Advanced Interface (With Images)

```python
# Lines 791-862
def build_demo_with_images():
    me = Me()
    set_client(me.openai)
    
    try:
        n = rebuild_if_empty()
        if n == 0:
            print("WARNING: KB has 0 chunks.", flush=True)
    except Exception as e:
        print("KB build skipped / failed:", e, flush=True)

    def chat_wrapper(message, history):
        """Wrapper to handle both text and image responses."""
        result = me.chat_with_images(message, history)
        
        if isinstance(result, dict) and "files" in result:
            response_content = result["content"]
            files = result.get("files", [])
            
            if files:
                # Create image section with Markdown syntax
                image_section = "\n\n## 📎 Relevant Assignment Pages:\n\n"
                
                for i, img_path in enumerate(files[:3], 1):
                    if os.path.exists(img_path):
                        filename = os.path.basename(img_path)
                        image_url = f"/assignment_images/{filename}"
                        image_section += f"![Assignment page {i}]({image_url})\n\n"
                
                return response_content + image_section
            else:
                return response_content
        else:
            return result
    
    interface = gr.ChatInterface(chat_wrapper, type="messages")
    return interface
```

**The Wrapper Pattern:**

Instead of calling `me.chat_with_images()` directly, we wrap it:

```
User message → chat_wrapper → me.chat_with_images → returns dict/string
                    ↓
            Formats for Gradio
                    ↓
            Returns formatted text
```

**Why the Wrapper?**
- `me.chat_with_images()` returns `{"content": text, "files": [paths]}`
- Gradio ChatInterface expects a string
- Wrapper converts the dictionary → formatted markdown string with images

**The Image Formatting:**
```markdown
## 📎 Relevant Assignment Pages:

![Assignment page 1](/assignment_images/Paltsokas-HW1_page_1_20251105_013050.png)

![Assignment page 2](/assignment_images/Paltsokas-HW1_page_2_20251105_013050.png)
```

---

## FastAPI Application

### Application Setup

```python
# Lines 864-889
demo = build_demo_with_images()

# Create shared Me instance for /chat endpoint
_shared_me = Me()
set_client(_shared_me.openai)

# Setup static file serving
root = Path(__file__).resolve().parent
static_dir = root / "static"
static_dir.mkdir(exist_ok=True)

# Create FastAPI app
app = FastAPI(title="Panos — Career Conversations")

# Mount Gradio at /gradio
gr.mount_gradio_app(app, demo, path="/gradio")

# Serve static files at /static
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve assignment images at /assignment_images
assignment_images_abs = os.path.abspath(assignment_images_directory)
os.makedirs(assignment_images_abs, exist_ok=True)
app.mount("/assignment_images", StaticFiles(directory=assignment_images_abs), name="assignment_images")
```

**The Routing Architecture:**

```
http://localhost:8000/
    ├─ /                          → index.html (your custom chat UI)
    ├─ /gradio                    → Gradio testing interface
    ├─ /static/...                → CSS, JS, other static files
    ├─ /assignment_images/...     → Generated PDF screenshots (PNG files)
    └─ /chat (POST)               → JSON API for custom frontend
```

**Why Two Me() Instances?**
- `demo` (Gradio): Creates its own `Me()` instance
- `_shared_me` (FastAPI `/chat`): Separate instance for the JSON API
- Each has its own OpenAI client and conversation state

**Mounting Order Matters!**
```python
gr.mount_gradio_app(app, demo, path="/gradio")  # Mount Gradio FIRST
app.mount("/static", ...)                        # Then static files
```
- Gradio needs to register its own static files
- If you mount `/static` first, it might conflict

### The Root Endpoint

```python
# Lines 891-893
@app.get("/", response_class=HTMLResponse)
def index():
    return (static_dir / "index.html").read_text(encoding="utf-8")
```

**Simple but Important:**
- Reads `static/index.html`
- Returns it as HTML
- This is your custom dark-themed chat interface

### The Chat API Endpoint

```python
# Lines 896-947
@app.post("/chat")
async def chat_api(payload: dict, request: Request):
    """
    Expects: {"message": "...", "history": [...]}
    Returns: {"reply": "..."}
    """
    message = (payload or {}).get("message", "")
    history = (payload or {}).get("history", [])
    if not isinstance(history, list):
        history = []
    
    try:
        # Call chat_with_images
        result = _shared_me.chat_with_images(message, history)
        
        # If result has files, format them
        if isinstance(result, dict) and "files" in result:
            response_content = result["content"]
            files = result.get("files", [])
            
            if files:
                # Add image URLs as Markdown
                image_section = "\n\n## 📎 Relevant Assignment Pages:\n\n"
                
                for i, img_path in enumerate(files[:3], 1):
                    if os.path.exists(img_path):
                        filename = os.path.basename(img_path)
                        image_url = f"/assignment_images/{filename}"
                        image_section += f"![Assignment page {i}]({image_url})\n\n"
                
                reply = response_content + image_section
                return JSONResponse({"reply": reply})
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
```

**Request/Response Format:**

**Request** (from your `static/index.html`):
```json
{
  "message": "Tell me about a project",
  "history": [
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello! How can I help?"}
  ]
}
```

**Response**:
```json
{
  "reply": "Here's a project I worked on...\n\n## 📎 Relevant Assignment Pages:\n\n![page 1](/assignment_images/file.png)"
}
```

**Why Async?**
```python
async def chat_api(payload: dict, request: Request):
```
- **FastAPI best practice**: Use `async` for endpoints
- Allows handling multiple requests concurrently
- Not critical for this app but good for scalability

---

## Data Flow Diagrams

### Conversation Flow (Without Images)

```
User types: "What's your RLHF experience?"
    ↓
[1] POST /chat → chat_api()
    ↓
[2] _shared_me.chat_with_images(message, history)
    ↓
[3] Detects NOT an assignment question
    ↓
[4] Calls me.chat(message, history)
    ↓
[5] Builds conversation context:
    - System prompt (identity + rules)
    - History (previous messages)
    - New user message
    ↓
[6] OpenAI API call with tools
    ↓
[7] AI decides: "I need KB context"
    Calls: rag_lookup(query="RLHF experience", k=8)
    ↓
[8] System executes:
    - rag_search() → embeds query → searches FAISS
    - Returns: "[rlhf_experience.md] I worked on RLHF at TaskUs..."
    ↓
[9] AI receives context, generates draft answer
    ↓
[10] Evaluator scores the draft
    ├─ Helpfulness: 5/5
    ├─ Faithfulness: 5/5
    └─ Style: 4/5
    ↓
[11] Scores ≥4 → Keep draft (no revision needed)
    ↓
[12] Auto-save to QADB (if feedback is positive)
    ↓
[13] Return final answer to user
```

### Conversation Flow (With Images)

```
User types: "Tell me about a clustering project"
    ↓
[1] POST /chat → chat_api()
    ↓
[2] _shared_me.chat_with_images(message, history)
    ↓
[3] Detects: IS an assignment question (has "project")
    ↓
[4] Calls me.chat(message, history)
    ├─ [Same flow as above: RAG lookup, generate, evaluate]
    └─ Returns: text_response
    ↓
[5] get_relevant_assignment_files(query="clustering project")
    ├─ Calls rag_search(query, k=12)
    ├─ Extracts PDF filenames from results
    ├─ Fallback: searches all KB for PDFs with keywords
    └─ Returns: ["kb/Paltsokas-HW4.pdf"]
    ↓
[6] For each PDF:
    pdf_page_to_image("kb/Paltsokas-HW4.pdf", max_pages=2)
    ├─ Converts first 2 pages to PNG
    ├─ Saves: screenshots/assignment_images/Paltsokas-HW4_page_1_20251105_013050.png
    └─ Returns: [image_path_1, image_path_2]
    ↓
[7] Returns dict:
    {
      "content": "Here's my clustering project...",
      "files": ["screenshots/.../page_1.png", "screenshots/.../page_2.png"]
    }
    ↓
[8] chat_wrapper formats:
    text + "\n\n## 📎 Relevant Assignment Pages:\n\n"
         + "![page 1](/assignment_images/file1.png)\n\n"
         + "![page 2](/assignment_images/file2.png)\n\n"
    ↓
[9] Returns formatted markdown to Gradio
    ↓
[10] Gradio renders (or attempts to render) images
```

---

## Key Design Patterns

### 1. Graceful Degradation
```python
if not PDF2IMAGE_AVAILABLE:
    return []  # Don't crash, just skip screenshots
```
- Missing dependencies → reduced functionality, not errors

### 2. Defensive Programming
```python
try:
    # risky operation
except Exception as e:
    print(f"[ERROR]: {e}")
    return fallback_value
```
- Always have a backup plan
- Log errors for debugging
- Never let one failure crash the whole app

### 3. Separation of Concerns
- **`Me` class**: Business logic (AI conversation)
- **FastAPI**: Web server (routing, HTTP)
- **Gradio**: UI framework (chat interface)
- **RAG functions**: Knowledge retrieval
- **QADB functions**: Persistent memory

Each component has ONE job, done well.

### 4. Dependency Injection
```python
set_client(me.openai)  # Inject OpenAI client into RAG system
```
- RAG system doesn't create its own client
- Reuses the one from `Me()`
- Avoids duplicate API connections

### 5. Configuration Over Code
```python
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
```
- Change models without changing code
- Just update `.env` file

---

## Current Limitations & Known Issues

### ❌ Image Display Problem
**Issue**: Gradio ChatInterface shows Markdown image syntax as plain text
```
![Assignment page 1](/assignment_images/file.png)
```
Instead of rendering the actual image.

**Why?**
- Gradio may sanitize/escape Markdown in certain contexts
- Version-specific behavior
- ChatInterface `type="messages"` might not support embedded images

**Attempted Solutions:**
1. Base64 encoding → Too large, shows as gibberish
2. HTML `<img>` tags → Gets escaped/sanitized
3. `sanitize_html=False` → Parameter doesn't exist in your Gradio version
4. Markdown with URLs → Currently testing

**Next Steps to Try:**
- Update Gradio to latest version
- Use Gradio's `Gallery` component
- Switch to pure FastAPI frontend (custom JS)

### ⚠️ Linter Warnings
```
Import "pdf2image" could not be resolved
Import "nbformat" could not be resolved
```
- Packages ARE installed and working
- VS Code/Pyright cache issue
- Fix: Reload VS Code window

---

## How the Pieces Work Together

### Example: User Asks About a Project

```
1. USER INTERFACE (static/index.html)
   User types: "Tell me about your clustering project"
   JavaScript POSTs to /chat

2. FASTAPI (app.py - chat_api endpoint)
   Receives: {"message": "...", "history": [...]}
   Calls: _shared_me.chat_with_images()

3. ME CLASS - chat_with_images()
   Detects: "project" keyword → assignment question
   Calls: me.chat() for text response
   
4. ME CLASS - chat()
   Builds context with system prompt
   OpenAI API call with tools enabled
   AI decides: Need KB context
   
5. TOOL EXECUTION - rag_lookup()
   Embeds query: "clustering project"
   Searches FAISS index
   Returns: Chunks from Paltsokas-HW4.pdf
   
6. AI GENERATION
   Uses KB context to write answer
   Draft: "I worked on k-means clustering..."
   
7. EVALUATION
   Evaluator scores: Helpful=5, Faithful=5, Style=4
   Scores high → No revision needed
   
8. AUTO-SAVE
   Saves to QADB for future reuse
   
9. IMAGE GENERATION
   get_relevant_assignment_files("clustering project")
   ├─ RAG finds: Paltsokas-HW4.pdf
   └─ Returns: ["kb/Paltsokas-HW4.pdf"]
   
   pdf_page_to_image("kb/Paltsokas-HW4.pdf")
   ├─ Converts pages 1-2 to PNG
   └─ Saves to screenshots/assignment_images/
   
10. FORMAT RESPONSE
    chat_wrapper receives:
    {
      "content": "I worked on k-means clustering...",
      "files": ["screenshots/.../page_1.png", "screenshots/.../page_2.png"]
    }
    
    Formats as Markdown:
    "I worked on k-means...\n\n## 📎 Relevant Assignment Pages:\n\n![page 1](url)..."
    
11. RETURN TO USER
    FastAPI returns: {"reply": "formatted text"}
    
12. FRONTEND RENDERS
    JavaScript receives reply
    Displays in chat interface
    (Gradio attempts to render Markdown images)
```

---

## Performance & Scalability

### What Makes This Fast?

1. **FAISS**: Optimized C++ library for vector search
   - Searches 1000 chunks in < 10ms

2. **SQLite FTS5**: Fast full-text search
   - Fuzzy matching in < 5ms

3. **Caching**: QADB stores good answers
   - Repeat questions → instant response
   - No API call needed

4. **Async FastAPI**: Handles concurrent requests

### What Could Be Slow?

1. **OpenAI API calls**: 1-3 seconds per request
   - Multiple tool calls = multiple API calls
   - Evaluator = extra API call

2. **PDF conversion**: 2-5 seconds per PDF
   - Poppler rendering is CPU-intensive
   - Mitigated by limiting to 2-3 pages

3. **First-time indexing**: 30-60 seconds
   - Needs to embed all KB content
   - Only happens once

---

## Security Considerations

### ✅ What's Secure

1. **API Keys in .env**: Not hardcoded
2. **Input validation**: Checks history is a list
3. **Error handling**: Doesn't expose stack traces to users (only to console)

### ⚠️ Potential Improvements

1. **Rate limiting**: No protection against spam
2. **Input sanitization**: Trusts all user input
3. **CORS**: No cross-origin restrictions
4. **Authentication**: Anyone can use the chat

**For a Personal Website**: Current security is fine
**For Production**: Would need rate limiting, auth, input validation

---

## Debugging & Monitoring

### Console Logging Strategy

Every major operation logs:
```python
print(f"[DEBUG] Created {len(images)} images", flush=True)
print(f"[INFO] FAISS check: loaded {len(meta)} chunks", flush=True)
print(f"[ERROR] PDF conversion failed: {e}", flush=True)
print(f"Tool called: {tool_name}", flush=True)
```

**Log Levels:**
- `[INFO]`: Normal operations (startup, index loading)
- `[DEBUG]`: Detailed tracing (image paths, RAG results)
- `[ERROR]`: Something went wrong
- `[WARNING]`: Unexpected but handled

**Why `flush=True`?**
- Forces immediate console output
- Without it, logs might be buffered and delayed
- Critical for real-time debugging

### Verifying the System Works

**1. Check FAISS Index:**
```bash
# Should see:
[INFO] FAISS check: loaded 931 chunks.
```

**2. Check RAG Lookup:**
```bash
# When asking about projects:
Tool called: rag_lookup
[DEBUG] rag_search found 12 chunks, sources: ['Paltsokas-HW4.pdf', ...]
```

**3. Check Image Generation:**
```bash
[DEBUG] Created 2 images from kb\Paltsokas-HW4.pdf
[DEBUG] Total images created: 4
[DEBUG] Added image 1: /assignment_images/Paltsokas-HW4_page_1_20251105_013050.png
```

**4. Check QADB Saves:**
```bash
[QADB] Saved: "What is your RLHF experience?"
```

---

## Summary: What We've Built

You now have a **production-grade AI chatbot** with:

### Core Features ✅
- [x] Conversational AI powered by GPT-4
- [x] RAG system with FAISS vector search
- [x] Persistent memory with SQLite Q&A database
- [x] PDF screenshot generation from assignments
- [x] Self-evaluation and answer refinement
- [x] Multiple tool integrations (KB search, caching, notifications)
- [x] Dual interface (Gradio + custom HTML)
- [x] Copy message functionality

### Architecture Highlights 🏗️
- **Modular design**: Easy to extend with new tools
- **Graceful degradation**: Works even with missing dependencies
- **Comprehensive logging**: Easy to debug and monitor
- **Persistent storage**: FAISS index and QADB survive restarts
- **Dual frontend**: Gradio for testing, custom HTML for production

### Technologies Used 🛠️
- **AI**: OpenAI GPT-4 + Embeddings
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Full-Text Search**: SQLite FTS5
- **Web Framework**: FastAPI + Gradio
- **PDF Processing**: pypdf + pdf2image + Poppler
- **Frontend**: Custom HTML/CSS/JavaScript

---

## Tomorrow's Tasks

### 🔴 Critical
1. **Fix image rendering in Gradio**
   - Research Gradio multimodal message formats
   - Test with updated Gradio version
   - Consider custom JavaScript solution

2. **Add user name input**
   - Replace "YOU" with actual user name
   - Store in localStorage
   - Update message rendering

### 🟡 Important
3. Widen chat interface for better readability
4. Test with all assignment PDFs
5. Verify RAG indexing is complete

### 🟢 Nice to Have
6. Clean up excessive debug logging
7. Add mobile responsive design
8. Optimize image compression

---

**End of Technical Guide — Written with ❤️ by Your AI Assistant Professor 🎓**

