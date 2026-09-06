# Voice-Enabled RAG (V-RAG)

V-RAG is a voice-based Retrieval-Augmented Generation assistant that lets you upload documents and talk to an AI assistant about their contents in real-time.

It supports PDF, DOCX, PPTX, TXT, PNG, JPG, and JPEG files. Uploaded documents are processed, embedded into a local Qdrant vector database, and retrieved when the assistant needs information from them.

---

## ⚡ Prerequisites

> [!IMPORTANT]
> **Python 3.11 ONLY** is required for this project. 
> Python versions below 3.11 lack required async typing features, and versions 3.12+ are incompatible with pre-compiled `paddlepaddle` / `paddleocr` C++ wheels.

* **Python:** `3.11.x`
* **System OCR (Optional for fast local fallback):** [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

---

## Features

* 🎙️ **Real-Time Voice AI:** Low-latency bidirectional voice conversation with Gemini Live (`gemini-3.1-flash-live-preview`).
* 📄 **Multi-Format Support:** Ingestion for `.pdf`, `.docx`, `.pptx`, `.txt`, `.png`, `.jpg`, `.jpeg`.
* 🔍 **Hybrid Vector Search:**
  * Dense semantic embeddings (`BAAI/bge-small-en-v1.5`)
  * Sparse BM25 lexical search (`Qdrant/bm25`)
  * Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`)
* 🖼️ **Multi-Tier Hybrid OCR Engine:**
  * **Tier 1:** Parallel Cloud Gemini 3.6 Flash
  * **Tier 2:** Tesseract Fast CPU OCR with quality heuristic router
  * **Tier 3:** PaddleOCR Heavy CPU fallback with thread locking
* 💾 **Local Vector Storage:** Qdrant embedded database (no external vector database service required).
* 🌐 **Modern UI:** Built-in web dashboard featuring real-time audio visualizers, document viewer links, and document deletion.

---

## How It Works

```
Document Upload ──► PyMuPDF / docx Parser ──► Text Found? ──► YES ──► Qdrant Indexing
                                                   │
                                                  NO
                                                   ▼
                                         Multi-Tier OCR Cascade
                                      ┌─────────────────────────┐
                                      │  Gemini 3.6 Flash (1st) │
                                      └────────────┬────────────┘
                                                   │ (Failed/Quota)
                                                   ▼
                                      ┌─────────────────────────┐
                                      │   Tesseract OCR (2nd)   │
                                      └────────────┬────────────┘
                                                   │ (Low Quality)
                                                   ▼
                                      ┌─────────────────────────┐
                                      │    PaddleOCR (3rd)      │
                                      └─────────────────────────┘
```

1. **Document Ingestion & Chunking:** Native text is parsed instantly. Embedded images or scanned pages are routed through the 3-tier OCR cascade. Text is then semantically chunked and indexed into Qdrant.
2. **Real-time Live Audio:** Clicking "Start Conversation" opens a WebSocket to the Gemini Multimodal Live API (`gemini-3.1-flash-live-preview`). Microphone PCM16 audio is streamed in real-time.
3. **Tool-Triggered RAG:** When queried about document contents, Gemini calls the `search_documents` function. The backend performs a Hybrid Search across Qdrant, reranks the hits, and feeds the context back into the live voice stream so the assistant can answer out loud.

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd V-RAG
```

### 2. Create and activate a Python 3.11 Virtual Environment

*Windows (PowerShell):*
```powershell
python3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

*Linux / macOS:*
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=AIzaSy...
```
*(Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey))*

### 5. Start the Application
```bash
cd backend
python server.py
```
Open your browser and navigate to: **`http://localhost:8000`**

---

## Project Structure

```
V-RAG/
├── backend/
│   ├── server.py             # FastAPI server & WebSocket live voice router
│   ├── rag_pipeline.py       # Hybrid Qdrant retriever & cross-encoder reranker
│   ├── requirements.txt      # Python 3.11 dependency list
│   └── uploads/              # Local storage for uploaded document files
├── frontend/
│   ├── index.html            # Web dashboard & audio controls
│   └── original-*.mp4        # Interactive visual orb video
└── .env                      # API keys configuration
```
