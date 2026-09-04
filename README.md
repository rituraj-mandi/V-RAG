# Voice-Enabled RAG

V-RAG (Voice RAG) is a voice-based Retrieval-Augmented Generation assistant that lets you upload documents and talk to an AI assistant about their contents.

It supports PDF, DOCX, PPTX, TXT, PNG, JPG, and JPEG files. Uploaded documents are processed, embedded into a local Qdrant vector database, and retrieved when the assistant needs information from them.

## Features
- 🎙️ **Real-time voice conversation** with Gemini Live
- 📄 **Upload** PDF, DOCX, PPTX, TXT and images
- 🔍 **Hybrid document search** using:
  - Dense semantic embeddings (`BAAI/bge-small-en-v1.5`)
  - Sparse BM25 search (`Qdrant/bm25`)
  - Reciprocal Rank Fusion (RRF)
- 🧠 **Cross-encoder reranking** in the RAG pipeline
- 🖼️ **OCR** for images and images embedded inside documents
- 💾 **Local Qdrant** vector database
- 🌐 **FastAPI backend** + simple web frontend

## How it Works (Pipeline)

The application unifies document ingestion, hybrid search, and live voice AI into a single seamless pipeline:

### 1. Document Ingestion
When you upload a file, the backend instantly parses the text (using `PyMuPDF` or `python-docx`). If it encounters images or scanned pages, it leverages the **Gemini API for Multimodal OCR**. The extracted text is then chunked via a dynamic sliding-window, embedded, and stored locally in **Qdrant**.

### 2. Live Voice Interaction
Clicking "Start Conversation" opens a direct **WebSocket (WSS)** connection between the browser and the **Gemini Multimodal Live API**. Your voice audio is streamed in real-time, bypassing traditional STT/TTS delays for instant, natural conversations.

### 3. Tool-Triggered RAG
During the conversation, if Gemini needs factual context, it triggers a `search_documents` function call over the socket. The backend intercepts this, runs a **Hybrid Search** (Dense + BM25 Sparse + RRF Fusion) across the Qdrant database, passes the top results through a **Cross-Encoder Reranker**, and streams the precise context back into the Gemini Live session so the AI can answer your query out loud!

## Setup

1. **Clone the project**
```bash
git clone <repository-url>
cd V-RAG/backend
```

2. **Create and activate the virtual environment**

*Windows:*
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*Linux / macOS:*
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```
> The project uses FastAPI, Uvicorn, Qdrant, FastEmbed, Sentence Transformers, Google GenAI, and document-processing libraries.

4. **Configure Gemini**
Create a `.env` file in the `backend` directory:
```bash
GEMINI_API_KEY=your_gemini_api_key
```
> The backend uses this key for Gemini Live and Gemini-powered OCR.

5. **Start the server**
```bash
python server.py
```
> Then open your browser and navigate to: http://localhost:8000

## Purpose

Built as a prototype for a **Voice-Enabled RAG system**, combining speech processing, information retrieval, and generative AI into a single seamless application.
