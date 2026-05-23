---
title: Portfolio GPT
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: true
---

# 🤖 Portfolio GPT

> "Instead of reading my resume, just chat with it."

An AI chatbot trained on my resume, projects, and skills using **RAG** (Retrieval-Augmented Generation).

## Features
- Ask anything about my skills, projects, and experience
- Answers are cited with source documents
- Powered by free HuggingFace models — no OpenAI needed

## Tech Stack
| Layer | Tool |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector DB | ChromaDB |
| LLM | Mistral-7B-Instruct (HuggingFace) |
| UI | Streamlit |
| Framework | LangChain |

## How it works
1. Documents (resume, projects, skills) are chunked and embedded
2. User question is embedded using the same model
3. Top matching chunks are retrieved from ChromaDB
4. Mistral-7B generates a cited answer from the context

Built by [Harik](https://github.com/harikrishnan2002)
