import os
import uuid
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Voice RAG API - Gemini Live")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = None

@app.on_event("startup")
async def startup_event():
    global rag
    print("Initializing RAG Pipeline on worker startup...")
    rag = RAGPipeline()

# Ensure absolute paths for frontend mounting
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(base_dir, "..", "frontend")

import io

from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def perform_ocr_gemini(image_bytes: bytes, mime_type: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "Extract all text from this accurately."
        ]
    )
    return response.text

def extract_pdf_text(file_bytes: bytes) -> str:
    import fitz
    import concurrent.futures
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for page in doc:
            # 1. Get standard text
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
                
            # 2. Extract and OCR any embedded images on the page
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    mime = "image/jpeg" if ext in ("jpeg", "jpg") else "image/png"
                    futures.append(executor.submit(perform_ocr_gemini, image_bytes, mime))
                except Exception as e:
                    print(f"Skipping an embedded image due to error: {e}")
                    
            # 3. Fallback for completely unparseable scanned pages that hide images in weird streams
            if len(text.strip()) < 30 and not image_list:
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                futures.append(executor.submit(perform_ocr_gemini, img_bytes, "image/png"))

        for future in futures:
            try:
                ocr_text = future.result()
                if ocr_text and ocr_text.strip():
                    text_parts.append(f"[Image Content]: {ocr_text.strip()}")
            except Exception:
                pass

    return "\n".join(text_parts)

def extract_docx_text(file_bytes: bytes) -> str:
    from docx import Document
    import concurrent.futures
    doc = Document(io.BytesIO(file_bytes))
    text_parts = []
    
    # 1. Text from paragraphs
    for p in doc.paragraphs:
        if p.text.strip():
            text_parts.append(p.text)
            
    # 2. Text from tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text_parts.append(cell.text.strip())
                    
    # 3. OCR on embedded images
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    image_bytes = rel.target_part.blob
                    mime = getattr(rel.target_part, "content_type", "image/jpeg")
                    futures.append(executor.submit(perform_ocr_gemini, image_bytes, mime))
                except Exception as e:
                    print(f"Skipping docx image due to error: {e}")
                    
        for future in futures:
            try:
                ocr_text = future.result()
                if ocr_text and ocr_text.strip():
                    text_parts.append(f"[Image Content]: {ocr_text.strip()}")
            except Exception:
                pass
                
    return "\n".join(text_parts)

def extract_pptx_text(file_bytes: bytes) -> str:
    from pptx import Presentation
    import concurrent.futures
    prs = Presentation(io.BytesIO(file_bytes))
    text_parts = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for slide in prs.slides:
            for shape in slide.shapes:
                # 1. Text from shapes
                if hasattr(shape, "text") and shape.text.strip():
                    text_parts.append(shape.text.strip())
                    
                # 2. OCR on pictures
                if hasattr(shape, "image"):
                    try:
                        image_bytes = shape.image.blob
                        mime = getattr(shape.image, "content_type", "image/jpeg")
                        futures.append(executor.submit(perform_ocr_gemini, image_bytes, mime))
                    except Exception as e:
                        print(f"Skipping pptx image due to error: {e}")
                        
        for future in futures:
            try:
                ocr_text = future.result()
                if ocr_text and ocr_text.strip():
                    text_parts.append(f"[Image Content]: {ocr_text.strip()}")
            except Exception:
                pass
                    
    return "\n".join(text_parts)

@app.post("/api/upload")
async def upload_doc(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()
    
    text = ""
    try:
        if filename.endswith(".pdf"):
            text = extract_pdf_text(content)
        elif filename.endswith(".docx"):
            text = extract_docx_text(content)
        elif filename.endswith(".pptx"):
            text = extract_pptx_text(content)
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            mime = "image/png" if filename.endswith(".png") else "image/jpeg"
            text = perform_ocr_gemini(content, mime)
        else:
            # Fallback to plain text
            text = content.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return {"status": "error", "message": str(e)}

    doc_id = str(uuid.uuid4())
    rag.add_document(text, doc_id, file.filename)
    return {"status": "success", "doc_id": doc_id, "title": file.filename}

@app.get("/api/documents")
async def get_documents():
    if rag:
        docs = rag.get_all_documents()
        return {"status": "success", "documents": docs}
    return {"status": "error", "message": "RAG pipeline not initialized"}

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        await websocket.close()
        return

    gemini_ws_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={api_key}"
    
    setup_msg = {
        "setup": {
            "model": "models/gemini-3.1-flash-live-preview",
            "generationConfig": {
                "responseModalities": ["AUDIO"]
            },
            "systemInstruction": {
                "parts": [{"text": "You are a helpful voice assistant. You can search documents to answer user queries using the search_documents tool. ALWAYS provide concise responses."}]
            },
            "tools": [{"functionDeclarations": [{
                "name": "search_documents",
                "description": "Search uploaded documents for information.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            }]}]
        }
    }
    
    try:
        async with websockets.connect(gemini_ws_url) as gemini_ws:
            await gemini_ws.send(json.dumps(setup_msg))
            
            # Wait for setup complete
            setup_resp = await gemini_ws.recv()
            print("Gemini Setup Response:", setup_resp)
            
            async def client_to_gemini():
                try:
                    while True:
                        data = await websocket.receive()
                        if 'bytes' in data:
                            msg = {
                                "realtimeInput": {
                                    "audio": {
                                        "mimeType": "audio/pcm;rate=16000", 
                                        "data": base64.b64encode(data['bytes']).decode()
                                    }
                                }
                            }
                            await gemini_ws.send(json.dumps(msg))
                        elif 'text' in data:
                            msg = json.loads(data['text'])
                            if msg.get('type') == 'client_content':
                                content_msg = {
                                    "clientContent": {
                                        "turns": [{
                                            "role": "user",
                                            "parts": [{"text": msg.get('text', '')}]
                                        }],
                                        "turnComplete": True
                                    }
                                }
                                await gemini_ws.send(json.dumps(content_msg))
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    print("Error in client_to_gemini:", e)
                    
            async def gemini_to_client():
                try:
                    while True:
                        resp = await gemini_ws.recv()
                        msg = json.loads(resp)
                        
                        if "serverContent" in msg:
                            model_turn = msg["serverContent"].get("modelTurn", {})
                            for part in model_turn.get("parts", []):
                                if "inlineData" in part:
                                    audio_data = base64.b64decode(part["inlineData"]["data"])
                                    await websocket.send_bytes(audio_data)
                        elif "toolCall" in msg:
                            calls = msg["toolCall"]["functionCalls"]
                            responses = []
                            for call in calls:
                                if call["name"] == "search_documents":
                                    query = call.get("args", {}).get("query", "")
                                    print(f"Tool call: search_documents(query='{query}')")
                                    docs = rag.retrieve(query, top_k=5)
                                    context = "\n".join(docs) if docs else "No documents found."
                                    responses.append({
                                        "id": call["id"],
                                        "name": call["name"],
                                        "response": {"result": context}
                                    })
                            if responses:
                                tool_resp = {
                                    "toolResponse": {
                                        "functionResponses": responses
                                    }
                                }
                                await gemini_ws.send(json.dumps(tool_resp))
                except Exception as e:
                    print("Error in gemini_to_client:", e)

            await asyncio.gather(client_to_gemini(), gemini_to_client())
    except Exception as e:
        print("Gemini WS Error:", e)
    finally:
        try:
            await websocket.close()
        except:
            pass

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/orb.jpg")
async def get_orb():
    return FileResponse(os.path.join(frontend_dir, "orb.jpg"))

@app.get("/")
async def root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
