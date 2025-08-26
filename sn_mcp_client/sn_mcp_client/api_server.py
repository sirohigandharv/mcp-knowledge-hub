from fastapi import FastAPI, Query, BackgroundTasks
from typing import Optional, Any, Dict, List
import asyncio
import json
import re
import os
from fastapi.responses import JSONResponse
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from bs4 import BeautifulSoup
from sn_mcp_client.client import MCPConfig, list_tools, call_tool, tool_to_dict, choose_arg_key
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
import random
import time
from google.cloud import storage

# ----------------------------
# Config & Globals
# ----------------------------

CONFIG_PATH = "config.client.ssc.json"
cfg = MCPConfig.load(CONFIG_PATH)

# Directories for downloads and uploads

DOWNLOAD_DIR = "downloads"
UPLOAD_DIR = "uploads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# NEW: Storage mode and bucket configuration
STORAGE_MODE = os.getenv("storage_mode", "local").lower()
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "")

if STORAGE_MODE == "cloud" and not GCP_BUCKET_NAME:
    raise ValueError("GCP_BUCKET_NAME must be set when storage_mode=cloud")

# If cloud, initialize GCP client
gcp_client = None
bucket = None
if STORAGE_MODE == "cloud":
    gcp_client = storage.Client()
    bucket = gcp_client.bucket(GCP_BUCKET_NAME)

# FAISS index and metadata
INDEX_PATH = os.path.join(UPLOAD_DIR, "vector_index.faiss")
METADATA_PATH = os.path.join(UPLOAD_DIR, "vector_metadata.pkl")

# NEW: If cloud mode, override index/metadata to bucket paths
if STORAGE_MODE == "cloud":
    INDEX_BLOB = "uploads/vector_index.faiss"
    METADATA_BLOB = "uploads/vector_metadata.pkl"

    # Download existing index and metadata if present
    if bucket.blob(INDEX_BLOB).exists():
        tmp_index = "/tmp/vector_index.faiss"
        bucket.blob(INDEX_BLOB).download_to_filename(tmp_index)
        INDEX_PATH = tmp_index

    if bucket.blob(METADATA_BLOB).exists():
        tmp_meta = "/tmp/vector_metadata.pkl"
        bucket.blob(METADATA_BLOB).download_to_filename(tmp_meta)
        METADATA_PATH = tmp_meta

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dimension = 384

if os.path.exists(INDEX_PATH):
    faiss_index = faiss.read_index(INDEX_PATH)
else:
    faiss_index = faiss.IndexFlatL2(embedding_dimension)

if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "rb") as f:
        metadata_store = pickle.load(f)
else:
    metadata_store = []

# Simple progress tracker in-memory
progress = {
    "running": False,
    "kb_total": 0,
    "kb_done": 0,
    "article_total": 0,
    "article_done": 0,
    "errors": 0,
    "last_error": None,
}

 # Simple counter for unique filenames
file_counter = 0

app = FastAPI(title="ServiceNow MCP REST API")

# ----------------------------
# Utilities
# ----------------------------

def pretty(obj: Any):
    """Existing pretty wrapper kept for endpoints that return JSON to client."""
    try:
        if isinstance(obj, dict):
            return pretty_content(obj)
        if isinstance(obj, str) and "text='" in obj:
            return extract_and_format(obj)
        if hasattr(obj, "text"):
            return extract_and_format(str(obj))
        return extract_and_format(str(obj))
    except Exception:
        return str(obj)

def pretty_content(result: Dict[str, Any]):
    return json.dumps(result, indent=4)

def extract_and_format(raw_string: str):
    match = re.search(r"text='(.*?)'(?:,|\))", raw_string, re.S)
    if not match:
        # maybe already JSON?
        try:    
            return json.loads(raw_string)
        except Exception:
            return "No JSON found in text field"
    inner_raw = match.group(1)
    unescaped = inner_raw.encode('utf-8').decode('unicode_escape')
    data = json.loads(unescaped)
    new_json = json.dumps(data, indent=4)
    return json.loads(new_json)

def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[\\/*?:"<>|\r\n\t]+', "_", name)
    sanitized = sanitized.strip().strip(".")
    return sanitized[:100]

def html_to_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n").strip()

def normalize_to_dict(result: Any) -> Dict[str, Any]:
    """Normalize outputs from call_tool into a python dict."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            pass
    # try existing extractor
    parsed = extract_and_format(str(result))
    if isinstance(parsed, str):
        try:
            return json.loads(parsed)
        except Exception:
            return {"raw": parsed}
    return parsed

async def retry_call_tool(tool_name: str, args: Dict[str, Any], *, retries=4, base_delay=0.5, max_delay=4.0) -> Dict[str, Any]:
    attempt = 0
    while True:
        try:
            res = await call_tool(cfg, tool_name, args)
            return normalize_to_dict(res)
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise
            # jittered exponential backoff
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = delay * (0.7 + 0.6 * random.random())
            progress["last_error"] = f"{tool_name}: {e}"
            await asyncio.sleep(delay)

# ----------------------------
# Public endpoints
# ----------------------------

@app.get("/tools")
async def get_tools():
    tools = await list_tools(cfg)
    return [tool_to_dict(t) for t in tools]

@app.get("/discover")
async def discover():
    tools = await list_tools(cfg)
    return {"discovered_tools": [tool_to_dict(t) for t in tools]}

@app.get("/list-kbs")
async def list_kbs():
    result = await retry_call_tool("list_knowledge_bases", {})
    return result


@app.get("/list-articles")
async def list_articles(kb: str = Query(..., description="Knowledge Base sys_id")):
    tools = await list_tools(cfg)
    target = next((t for t in tools if getattr(t, "name", "") == "list_articles"), None)
    key = "kb_id"
    if target:
        key = choose_arg_key(getattr(target, "input_schema", {}) or {},
                             ["kb_sys_id", "kb_id", "knowledge_base_id", "id", "kb"],
                             "kb_id")
    args = {key: kb}
    result = await retry_call_tool("list_articles", args)
    return result

@app.get("/get-article")
async def get_article(article_id: str = Query(..., description="Article sys_id")):
    args = {"article_id": article_id}
    result = await retry_call_tool("get_article", args)
    return result

# ----------------------------
# Optimized download flow
# ----------------------------

@app.get("/download")
async def download_kb_and_articles(background_tasks: BackgroundTasks,
                                   concurrency: int = Query(8, ge=1, le=64)):
    """
    Starts a background job to fetch all KBs, all articles, and generate PDFs.
    Returns immediately with a status pointer.
    """
    if progress["running"]:
        return {"success": False, "message": "A download job is already running."}

    progress.update({
        "running": True,
        "kb_total": 0,
        "kb_done": 0,
        "article_total": 0,
        "article_done": 0,
        "errors": 0,
        "last_error": None,
    })
    
    background_tasks.add_task(download_all_articles_job, concurrency)
    file_counter = 0  # reset counter for this job
    return {"success": True, "message": "Download started", "check_status_at": "/download/status"}

@app.get("/download/status")
async def download_status():
    return progress

async def download_all_articles_job(concurrency: int):
    try:
        # Step 1: KBs
        kb_data = await retry_call_tool("list_knowledge_bases", {})
        kb_ids = [kb["id"] for kb in kb_data.get("knowledge_bases", [])]
        progress["kb_total"] = len(kb_ids)

        sem = asyncio.Semaphore(concurrency)

        # Discover article IDs (sequential per KB to avoid hammering upstream)
        all_article_ids: List[str] = []
        for kb_id in kb_ids:
            try:
                articles_data = await list_articles_raw(kb_id)
                article_ids = [a["id"]["value"] for a in articles_data.get("articles", [])]
                all_article_ids.extend(article_ids)
            except Exception as e:
                progress["errors"] += 1
                progress["last_error"] = f"list_articles({kb_id}): {e}"
            finally:
                progress["kb_done"] += 1

        progress["article_total"] = len(all_article_ids)

        # Step 2: Download + PDF in parallel with cap
        async def worker(article_id: str, index: int):
            async with sem:
                await download_and_save_article(article_id, index)

        # tasks = [asyncio.create_task(worker(aid)) for aid in all_article_ids]
        tasks = [
            asyncio.create_task(worker(aid, i + 1))
            for i, aid in enumerate(all_article_ids)
        ]
        # Use as_completed to update progress promptly
        for fut in asyncio.as_completed(tasks):
            try:
                await fut
            except Exception as e:
                progress["errors"] += 1
                progress["last_error"] = f"article_task: {e}"
            finally:
                progress["article_done"] += 1

    finally:
        progress["running"] = False

async def list_articles_raw(kb_id: str) -> Dict[str, Any]:
    tools = await list_tools(cfg)
    target = next((t for t in tools if getattr(t, "name", "") == "list_articles"), None)
    key = "kb_id"
    if target:
        key = choose_arg_key(getattr(target, "input_schema", {}) or {},
                             ["kb_sys_id", "kb_id", "knowledge_base_id", "id", "kb"],
                             "kb_id")
    args = {key: kb_id}
    return await retry_call_tool("list_articles", args)

async def download_and_save_article(article_id: str, index: int):
    # Fetch article via MCP
    data = await retry_call_tool("get_article", {"article_id": article_id})
    article = data.get("article", {}) if isinstance(data, dict) else {}
    title = article.get("title", "Untitled Article")
    html_text = article.get("text", "")
    plain_text = html_to_text(html_text)

    safe_title = sanitize_filename(title)[:50]
    pdf_path = os.path.join(DOWNLOAD_DIR, f"{index}_{article_id}_{safe_title}.pdf")

    if os.path.exists(pdf_path):
        return

    # Offload CPU/file I/O to thread
    await asyncio.to_thread(write_pdf, pdf_path, title, plain_text)

def write_pdf(pdf_path: str, title: str, plain_text: str):
    c = canvas.Canvas(pdf_path, pagesize=LETTER)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, title)
    c.setFont("Helvetica", 12)
    y = 720
    for line in plain_text.splitlines():
        c.drawString(50, y, line[:90])
        y -= 20
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 750
    c.save()

    # NEW: Upload to GCP bucket if in cloud mode
    if STORAGE_MODE == "cloud":
        blob_name = os.path.basename(pdf_path)
        blob = bucket.blob(f"downloads/{blob_name}")
        blob.upload_from_filename(pdf_path)
        os.remove(pdf_path)  # Remove local copy after upload

# ----------------------------
# Upload / Index / Search (unchanged API, minor robustness)
# ----------------------------

@app.get("/upload")
async def upload_files_from_downloads():
    print("Starting upload and indexing of PDF files from downloads...")
    stored_files = []
    chunk_size = 500

     # NEW: Decide source based on storage mode
    if STORAGE_MODE == "cloud":
        # List files from GCP bucket under 'downloads/'
        blobs = list(bucket.list_blobs(prefix="downloads/"))
        pdf_files = [b.name for b in blobs if b.name.lower().endswith(".pdf")]
    else:
        pdf_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        return JSONResponse(content={
            "success": False,
            "message": "No PDF files found in the downloads folder."
        })

    total_chunks = 0
    all_embeddings: List[np.ndarray] = []
    new_metadata: List[Dict[str, str]] = []

    for filename in pdf_files:
        if STORAGE_MODE == "cloud":
            # Download from GCP to memory
            blob = bucket.blob(filename)
            tmp_path = os.path.join("/tmp", os.path.basename(filename))
            blob.download_to_filename(tmp_path)
            file_path = tmp_path
            stored_files.append(os.path.basename(filename))
        else:
            file_path = os.path.join(DOWNLOAD_DIR, filename)
            stored_files.append(filename)

        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue

        if not text.strip():
            print(f"Skipping empty PDF: {filename}")
            continue

        # Chunk
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        total_chunks += len(chunks)

        # Embeddings (batch)
        embeddings = embedding_model.encode(chunks)
        embeddings = np.array(embeddings).astype('float32')

        all_embeddings.append(embeddings)
        for chunk in chunks:
            new_metadata.append({"filename": os.path.basename(filename), "chunk": chunk})

    if new_metadata:
        # Add to FAISS in one go
        concat_embeddings = np.concatenate(all_embeddings, axis=0)
        faiss_index.add(concat_embeddings)
        metadata_store.extend(new_metadata)

        # Persist locally
        faiss.write_index(faiss_index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(metadata_store, f)

        # NEW: Upload to bucket if in cloud mode
        if STORAGE_MODE == "cloud":
            bucket.blob(INDEX_BLOB).upload_from_filename(INDEX_PATH)
            bucket.blob(METADATA_BLOB).upload_from_filename(METADATA_PATH)


    return JSONResponse(content={
        "success": True,
        "message": f"Processed and indexed {len(stored_files)} PDF(s).",
        "files": stored_files,
        "total_vectors": faiss_index.ntotal,
        "chunks_added": total_chunks
    })

@app.get("/search")
async def search(query: str = Query(..., description="Search query"),
                 top_k: int = Query(5, description="Number of top results to return")):
    if faiss_index.ntotal == 0:
        return JSONResponse(content={
            "success": False,
            "message": "No data found in the FAISS index. Please upload files first."
        })

    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        meta = metadata_store[idx]
        results.append({
            "filename": meta["filename"],
            "chunk": meta["chunk"],
            "distance": float(dist)
        })

    return JSONResponse(content={
        "success": True,
        "query": query,
        "top_k": len(results),
        "results": results
    })
