
# 📄 Research Paper Summarization & Q&A Assistant

An AI-powered web application that allows users to **upload research papers**, **generate concise summaries**, and **ask context-specific questions** about the content using a Retrieval-Augmented Generation (RAG) pipeline built with open-source LLMs.

---

## 🎯 Features

- 📥 **PDF Upload**: Upload research papers in `.pdf` format
- 🧠 **Summarization**: Generates summaries using Hugging Face’s `distilbart-cnn-12-6` model
- 🔍 **Semantic Chunking**: Uses LangChain’s `RecursiveCharacterTextSplitter` for efficient chunking
- 💬 **Contextual Q&A**: Ask questions using a RAG pipeline with vector embeddings
- 🌐 **Streamlit UI**: Clean, interactive user interface for non-technical users

---

## 🧠 Tech Stack

| Component           | Technology                                      |
|---------------------|--------------------------------------------------|
| Frontend UI         | Streamlit                                        |
| LLM for Summarization | Hugging Face (`distilbart-cnn-12-6`)            |
| RAG Pipeline        | LangChain + ChromaDB + SentenceTransformers     |
| Embeddings          | `all-MiniLM-L6-v2`                               |
| Vector Store        | ChromaDB                                         |
| PDF Parsing         | LangChain’s `PyPDFLoader`                        |

---

## 📊 Impact

- ✅ Used by 10+ testers
- 📄 Summarized over 20 research papers
- ⏱ Reduced manual reading effort by ~60%

---

## 📂 Folder Structure

