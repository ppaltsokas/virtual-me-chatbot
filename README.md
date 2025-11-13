---
title: career_conversations
app_file: app.py
sdk: gradio
sdk_version: 5.34.2
---

# Career Conversations

This little space pairs my career notes with a lightweight Gradio chat so I can rehearse interviews or kick around new project ideas without digging through folders.

## What's inside
- `app.py` spins up the chat interface and wiring to the knowledge base.
- `kb/` carries topic shells (ML theory, maths refreshers, Python refreshers) ready for new notes.
- `static/` holds the minimal front-end assets that give the UI its calm feel.

## Quick start
```bash
pip install -r requirements.txt
python app.py
```
Then open the URL that Gradio prints—no extra wiring needed.

## Project notes
- Prompts stay intentionally short; I prefer to nudge the assistant live instead of stuffing huge system messages.
- The knowledge base is currently empty on purpose; drop curated notes inside the topic folders when you're ready.
- Hugging Face Space metadata lives at the top of this file, so leave that front matter untouched.
