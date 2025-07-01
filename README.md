# Samsung Manual RAG Chatbot

A fully-local, CPU-only Retrieval-Augmented Generation (RAG) system that answers natural-language questions against the Samsung Galaxy S24 user manual PDF. Built with FAISS for vector search, TinyLlama-1.1B for local inference, and Streamlit for a clean web UI—no external APIs or cloud services required.

## 🚀 Features

- **PDF Processing & Chunking**  
  Clean extraction and sliding-window chunking of a 150-page manual.  
- **Semantic Search**  
  MiniLM embeddings + FAISS Inner-Product index for fast, accurate retrieval.  
- **Local LLM Inference**  
  4-bit quantized TinyLlama-1.1B loaded via `llama_cpp` for sub-2 s answers on CPU.  
- **Web Interface**  
  Simple Streamlit app—type a question, click “Answer,” and get results instantly.  
- **Zero API Costs**  
  Entirely self-contained; ideal for SMEs and on-prem deployments.
