# Samsung Manual RAG Chatbot

An end-to-end, fully local Retrieval-Augmented Generation (RAG) system for the Samsung Galaxy S24 user manual.  
Builds a searchable knowledge base from a PDF, then answers natural-language questions via a lightweight TinyLlama chat model—no external APIs, cloud services, or GPUs required.

---

## 🧐 Overview

Many organizations—especially in manufacturing, electronics, and technical support—hold vast amounts of unstructured PDF documentation. This project demonstrates how to:

1. **Extract** and **clean** PDF text.
2. **Chunk** and **embed** content for semantic search.
3. **Retrieve** relevant passages with FAISS.
4. **Generate** answers locally using a 4-bit quantized TinyLlama LLM.
5. **Serve** as a user-friendly web app via Streamlit.

---

## ✨ Features

- **PDF Preprocessing**  
  • Automatic extraction with `pdfplumber`  
  • Removal of headers, footers, page numbers, and boilerplate  
  • Overlapping “sliding-window” chunking to preserve context  

- **Semantic Search**  
  • SentenceTransformers MiniLM embeddings  
  • Normalized vectors + FAISS `IndexFlatIP` for fast cosine search  
  • Configurable retrieval size (`k`)  

- **Local LLM Inference**  
  • TinyLlama-1.1B 4-bit quantized model via `llama_cpp`  
  • Sub-2 second response time on a 4-core CPU  
  • Fully offline & zero API cost  

- **Interactive UI**  
  • Streamlit web interface  

- **Extensible & Reusable**  
  • Notebook for rapid experimentation (`notebook.ipynb`)  
  • Easy to swap in new manuals or models  
