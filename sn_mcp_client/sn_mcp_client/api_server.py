from fastapi import FastAPI, Query, BackgroundTasks
from typing import Optional, Any, Dict, List, Tuple
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
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
#from nltk import pos_tag
from itertools import combinations
import nltk
from openai import OpenAI

# Download required NLTK resources (only run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')



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


# ---- Embeddings New ----
from sentence_transformers import SentenceTransformer
import faiss, os, pickle
from typing import List, Dict, Any, Optional

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = embedding_model.get_sentence_embedding_dimension()

# ---- FAISS + metadata paths ----
INDEX_PATH_NEW = os.path.join(UPLOAD_DIR, "kb.index")
META_PATH_NEW  = os.path.join(UPLOAD_DIR, "kb_meta.pkl")

# For cloud mode, define blob names
if STORAGE_MODE == "cloud":
    INDEX_BLOB_NEW = "uploads/kb.index"
    META_BLOB_NEW  = "uploads/kb_meta.pkl"

    # Download existing index & metadata if present in GCP bucket
    if bucket.blob(INDEX_BLOB_NEW).exists():
        tmp_index = "/tmp/kb.index"
        bucket.blob(INDEX_BLOB_NEW).download_to_filename(tmp_index)
        INDEX_PATH_NEW = tmp_index

    if bucket.blob(META_BLOB_NEW).exists():
        tmp_meta = "/tmp/kb_meta.pkl"
        bucket.blob(META_BLOB_NEW).download_to_filename(tmp_meta)
        META_PATH_NEW = tmp_meta

# ---- FAISS index initialization ----
index: Optional[faiss.Index] = None
metadata_store: List[Dict[str, Any]] = []

def load_or_init_index():
    """Load existing FAISS + metadata or create a new one."""
    global index, metadata_store
    if os.path.exists(INDEX_PATH_NEW) and os.path.exists(META_PATH_NEW):
        index = faiss.read_index(INDEX_PATH_NEW)
        with open(META_PATH_NEW, "rb") as f:
            metadata_store[:] = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)  # cosine similarity via inner product

def persist_index():
    """Persist FAISS + metadata locally and upload to GCP if in cloud mode."""
    faiss.write_index(index, INDEX_PATH_NEW)
    with open(META_PATH_NEW, "wb") as f:
        pickle.dump(metadata_store, f)

    if STORAGE_MODE == "cloud":
        bucket.blob(INDEX_BLOB_NEW).upload_from_filename(INDEX_PATH_NEW)
        bucket.blob(META_BLOB_NEW).upload_from_filename(META_PATH_NEW)

load_or_init_index()

# ---- LLM (optional, used in /searchquery) ----
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    openai_client = OpenAI()
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")



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



@app.get("/onlinesearch")
async def online_search(
    query: str = Query(..., description="Search phrase for articles"),
    limit: int = Query(50, description="Maximum number of articles to return"),
    knowledge_base: Optional[str] = Query(None, description="Optional Knowledge Base filter")
):
    """
    Perform an online semantic search for knowledge articles using the list_articles tool.
    - First tries the original query directly.
    - If no results, expands the query into multiple related searches and combines the results.
    """
    try:
        # Step 1: Try the original query first
        args = {
            "limit": limit,
            "query": query,
            "knowledge_base": knowledge_base
        }

        try:
            original_data = await retry_call_tool("list_articles", args)
        except Exception as e:
            print(f"Error while querying original phrase '{query}': {e}")
            original_data = {"articles": []}

        initial_results = []
        seen_ids: Set[str] = set()

        for art in original_data.get("articles", []):
            article_id = (
                art.get("id", {}).get("value")
                if isinstance(art.get("id"), dict)
                else art.get("id")
            )
            if article_id not in seen_ids:
                seen_ids.add(article_id)
                initial_results.append({
                    "article_id": article_id,
                    "title": art.get("title", ""),
                    "short_description": art.get("short_description", ""),
                    "knowledge_base": (
                        art.get("knowledge_base", {}).get("value")
                        if isinstance(art.get("knowledge_base"), dict)
                        else art.get("knowledge_base")
                    )
                })

        # If we found results for the original query, return them immediately
        if initial_results:
            return {
                "success": True,
                "original_query": query,
                "expanded_queries": [],
                "total_matches": len(initial_results),
                "results": initial_results
            }

        # Step 2: Fallback – Expand query only if no results found
        query_variants = generate_query_variants(query)
        print(f"Generated variants: {query_variants}")

        all_results = []
        for q in query_variants:
            args = {
                "limit": limit,
                "query": q,
                "knowledge_base": knowledge_base
            }

            try:
                articles_data = await retry_call_tool("list_articles", args)
            except Exception as e:
                print(f"Error while querying variant '{q}': {e}")
                continue  # Skip this variant and move to the next

            for art in articles_data.get("articles", []):
                article_id = (
                    art.get("id", {}).get("value")
                    if isinstance(art.get("id"), dict)
                    else art.get("id")
                )

                if article_id not in seen_ids:
                    seen_ids.add(article_id)
                    all_results.append({
                        "article_id": article_id,
                        "title": art.get("title", ""),
                        "short_description": art.get("short_description", ""),
                        "knowledge_base": (
                            art.get("knowledge_base", {}).get("value")
                            if isinstance(art.get("knowledge_base"), dict)
                            else art.get("knowledge_base")
                        )
                    })

        return {
            "success": True,
            "original_query": query,
            "expanded_queries": query_variants,
            "total_matches": len(all_results),
            "results": all_results
        }

    except Exception as e:
        return {"success": False, "error": str(e)}




# ----------------------------
# Optimized download flow
# ----------------------------

@app.get("/download")
async def download_kb_and_articles(concurrency: int = Query(2, ge=1, le=64)):
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
    
    #background_tasks.add_task(download_all_articles_job, concurrency)
    asyncio.create_task(download_all_articles_job(concurrency))  # Fire-and-forget
    # file_counter = 0  # reset counter for this job
    return {"success": True, "message": "Download started", "check_status_at": "/download/status"}

@app.get("/download/status")
async def download_status():
    return progress

async def download_all_articles_job(concurrency: int):
    try:
        kb_data = await retry_call_tool("list_knowledge_bases", {})
        kb_ids = [kb["id"] for kb in kb_data.get("knowledge_bases", [])]
        progress["kb_total"] = len(kb_ids)

        sem = asyncio.Semaphore(concurrency)
        all_article_ids: List[str] = []

        for kb_id in kb_ids:
            try:
                articles_data = await list_articles_raw(kb_id)
                article_ids = [a["id"]["value"] for a in articles_data.get("articles", [])]
                all_article_ids.extend(article_ids)
            except Exception as e:
                progress["errors"] += 1
                progress["last_error"] = f"list_articles({kb_id}): {str(e)}"
            finally:
                progress["kb_done"] += 1

        progress["article_total"] = len(all_article_ids)
        progress["failed_articles"] = []

        async def worker(article_id: str, index: int):
            async with sem:
                try:
                    await download_and_save_article(article_id, index)
                except Exception as inner_e:
                    progress["errors"] += 1
                    progress["failed_articles"].append(article_id)
                    if not progress.get("last_error"):
                        progress["last_error"] = f"Error for article {article_id}: {repr(inner_e)}"
                finally:
                    progress["article_done"] += 1

        tasks = [asyncio.create_task(worker(aid, i + 1)) for i, aid in enumerate(all_article_ids)]
        await asyncio.gather(*tasks)  # gather ensures all tasks finish even if some fail

        # Step 3: Retry failed ones
        if progress["failed_articles"]:
            retry_list = progress["failed_articles"].copy()
            progress["failed_articles"] = []
            for i, aid in enumerate(retry_list):
                try:
                    await download_and_save_article(aid, i + 1)
                except Exception as e:
                    progress["errors"] += 1
                    progress["failed_articles"].append(aid)
                    if not progress.get("last_error"):
                        progress["last_error"] = f"Retry failed for article {aid}: {repr(e)}"

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
        # Extract KB ID and Article from filename
        filename = meta["filename"].replace(".pdf", "")
        parts = filename.split("_", 2)  # split only first two underscores
        # parts[0] = '2', parts[1] = '021e2cc4dbde578073cb78b5ae961917', parts[2] = 'Prevent further recurrence...'
        article_id = parts[1] if len(parts) > 1 else ""
        article = parts[2] if len(parts) > 2 else ""
        
        results.append({
            "Article Title": article,
            "Article ID": article_id,
            "chunk": meta["chunk"],
            "distance": float(dist)
        })


    return JSONResponse(content={
        "success": True,
        "query": query,
        "top_k": len(results),
        "results": results
    })


'''

def generate_query_variants(query: str, max_combo_length: int = 3) -> list[str]:
    """
    Generate semantic query variants from a sentence:
    - Removes unnecessary stopwords but keeps meaningful modifiers (POS-based)
    - Extracts individual keywords
    - Creates combinations of keywords
    - Adds synonyms for each keyword
    """
    stop_words = set(stopwords.words('english'))
    # Tokenize and keep only alphanumeric tokens
    tokens = [w.lower() for w in word_tokenize(query) if w.isalnum()]
    
    # Part-of-speech tagging
    pos_tags = pos_tag(tokens)
    
    # Keep nouns (NN), verbs (VB), adjectives (JJ), adverbs (RB)
    keywords = [
        word for word, pos in pos_tags
        if pos.startswith(('N', 'V', 'J', 'R')) and word not in stop_words
    ]
    
    variants = set()
    
    # Add cleaned base phrase
    if keywords:
        variants.add(" ".join(keywords))
    
    # Add individual keywords
    for kw in keywords:
        variants.add(kw)
    
    # Add combinations (pairs, triples)
    for r in range(2, min(len(keywords), max_combo_length) + 1):
        for combo in combinations(keywords, r):
            variants.add(" ".join(combo))
    
    # Add synonyms (only 1–2 per word for relevance)
    for kw in keywords:
        for syn in wordnet.synsets(kw):
            for lemma in syn.lemmas()[:2]:  # limit synonyms per word
                synonym = lemma.name().replace("_", " ").lower()
                if synonym != kw and synonym.isalnum():
                    variants.add(synonym)
    
    return sorted(list(variants))

'''

def generate_query_variants(query: str, max_combo_length: int = 3) -> list[str]:
    """
    Generate query variants using only the actual words present in the input string.
    - Removes stopwords
    - Keeps only meaningful words
    - Generates combinations (pairs, triples)
    """
    stop_words = set(stopwords.words('english'))
    
    # Tokenize and keep only alphanumeric tokens
    tokens = [w.lower() for w in word_tokenize(query) if w.isalnum()]
    
    # Remove stopwords
    keywords = [w for w in tokens if w not in stop_words]

    variants = set()

    # Add cleaned base phrase (if more than one word remains)
    if keywords:
        variants.add(" ".join(keywords))

    # Add individual keywords
    for kw in keywords:
        variants.add(kw)

    # Add combinations (pairs, triples)
    for r in range(2, min(len(keywords), max_combo_length) + 1):
        for combo in combinations(keywords, r):
            variants.add(" ".join(combo))

    return sorted(list(variants))



# New Embeddings - Index and Search

def extract_article_text(article: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Normalize article payload into a single text blob and structured metadata.
    Expect fields like: title, body_html/body_md/body_text, url/permalink, kb_id, article_id
    """
    title = article.get("title") or article.get("name") or ""
    url   = article.get("url") or article.get("permalink") or ""
    kb_id = article.get("kb_id") or article.get("knowledge_base_id")
    aid   = article.get("id") or article.get("article_id")

    body_candidates = [
        article.get("body_html"),
        article.get("body_md"),
        article.get("body"),
        article.get("resolution_steps"),
        article.get("content"),
        article.get("description"),
    ]
    bodies = [b for b in body_candidates if b]
    merged = "\n\n".join(bodies)

    # Try to convert HTML/MD→text but keep raw if already plain.
    text = html_to_text(merged)
    if len(text.strip()) < 10 and isinstance(merged, str):
        text = merged.strip()

    meta = {
        "kb_id": kb_id,
        "article_id": aid if isinstance(aid, str) else (aid or {}).get("value", aid),
        "title": title,
        "url": url,
    }
    return text, meta

# ---- Chunking ----
def chunk_text(s: str, max_chars: int = 1200, overlap: int = 150) -> List[Tuple[str, int, int]]:
    """
    Simple char-based recursive splitter with overlap.
    Returns list of (chunk, start_idx, end_idx).
    """
    s = s.strip()
    if not s:
        return []

    chunks = []
    i = 0
    n = len(s)
    while i < n:
        end = min(i + max_chars, n)
        # try to cut on whitespace if possible
        cut = s.rfind("\n", i, end)
        if cut == -1:
            cut = s.rfind(" ", i, end)
        if cut == -1 or cut <= i + 200:  # don't cut too early
            cut = end
        segment = s[i:cut].strip()
        if segment:
            chunks.append((segment, i, cut))
        i = max(cut - overlap, cut)  # step with overlap but never backward
    return chunks

# ---- Embedding helpers ----
def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        e = embedding_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        vecs.append(e)
    return np.vstack(vecs) if vecs else np.zeros((0, EMBED_DIM), dtype="float32")


# ---- Ingest (single op) ----
@app.post("/ingest")
async def ingest_all(concurrency: int = Query(4, ge=1, le=64)):
    """
    Single operation to build/refresh the RAG index:
    - List KBs
    - List articles per KB
    - Fetch each article payload using existing MCP methods
    - Extract text (from `text` field), chunk, embed
    - Add to FAISS with metadata
    """
    load_or_init_index()

    # Use your existing KB listing method
    kbs = await list_kbs()
    kb_list = kbs.get("knowledge_bases", []) if isinstance(kbs, dict) else kbs
    kb_ids = [kb.get("id") for kb in kb_list]
    if not kb_ids:
        return JSONResponse({"success": False, "message": "No knowledge bases returned by MCP."})

    sem = asyncio.Semaphore(concurrency)
    new_chunks: List[str] = []
    new_chunk_meta: List[Dict[str, Any]] = []

    async def process_article(article_entry: Dict[str, Any], kb_id: str):
        # Extract article ID (works with both string or dict)
        art_id = (
            article_entry.get("id")
            if isinstance(article_entry.get("id"), str)
            else (article_entry.get("id") or {}).get("value")
        )

        async with sem:
            # Fetch full article using your existing helper
            article_data = await get_article(art_id)
            article = article_data.get("article", {})

            # Use your known schema: "text" contains HTML
            raw_html = article.get("text") or ""
            text = html_to_text(raw_html).strip()

            if not text or len(text) < 10:
                print(f"Skipping empty article: {art_id}")
                return

            # Chunk & store
            for chunk, start, end in chunk_text(text):
                new_chunks.append(chunk)
                new_chunk_meta.append({
                    "kb_id": kb_id,
                    "article_id": art_id,
                    "title": article.get("title"),
                    "url": article.get("url") or "",
                    "char_start": start,
                    "char_end": end,
                    "preview": chunk[:240],
                    "content": chunk
                })

    # Gather all articles across KBs
    articles: List[Tuple[str, Dict[str, Any]]] = []
    for kb_id in kb_ids:
        try:
            # Use your existing list_articles_raw() function
            arr = await list_articles_raw(kb_id)
            # Your existing response is a dict with "articles"
            article_items = arr.get("articles", arr) if isinstance(arr, dict) else arr
            for a in article_items:
                articles.append((kb_id, a))
        except Exception as e:
            print(f"list_articles failed for KB {kb_id}: {e}")

    # Process concurrently
    tasks = [asyncio.create_task(process_article(a, kb_id)) for kb_id, a in articles]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Embed & persist
    vecs = embed_texts(new_chunks)
    added = 0
    if vecs.shape[0] > 0:
        index.add(vecs)
        metadata_store.extend(new_chunk_meta)
        persist_index()
        added = vecs.shape[0]

    return JSONResponse({
        "success": True,
        "message": "Ingest complete",
        "kb_count": len(kb_ids),
        "article_count": len(articles),
        "chunks_added": added,
        "total_vectors": int(index.ntotal),
    })


# ---- Search + generate with citations ----
def retrieve(query: str, k: int = 6) -> List[Tuple[int, float]]:
    if index.ntotal == 0:
        return []
    qv = embed_texts([query])  # already normalized
    scores, idxs = index.search(qv, k)
    pairs = []
    for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
        if i == -1:
            continue
        pairs.append((i, float(s)))
    return pairs

def build_citations(hits: List[Tuple[int, float]], min_score: float = 0.25) -> List[Dict[str, Any]]:
    cites = []
    for i, score in hits:
        if score < min_score:
            continue
        m = metadata_store[i]
        cites.append({
            "rank": len(cites) + 1,
            "score": score,
            "kb_id": m.get("kb_id"),
            "article_id": m.get("article_id"),
            "title": m.get("title"),
            "url": m.get("url"),
            "char_start": m.get("char_start"),
            "char_end": m.get("char_end"),
            "preview": m.get("preview"),
            "index": i,
        })
    return cites
'''
def build_llm_context(citations: List[Dict[str, Any]]) -> str:
    """
    Small, clean context block for the LLM. Each source is tagged [S1], [S2], ...
    """
    lines = []
    for c in citations:
        tag = f"[S{c['rank']}]"
        meta = f"{tag} {c.get('title') or 'Untitled'} (KB: {c.get('kb_id')}, Article: {c.get('article_id')})"
        url = c.get("url") or ""
        if url:
            meta += f" — {url}"
        lines.append(meta)
        # Use full chunk instead of just preview
        lines.append(c.get("content") or c.get("preview") or "")
        lines.append("")  # blank line
    return "\n".join(lines)
'''

# --- Helper: deduplicate and build LLM context using full chunk content ---
def dedupe_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for c in citations:
        key = (c.get("article_id"), c.get("char_start"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique

def build_llm_context(citations: List[Dict[str, Any]], max_sources: int = 5) -> str:
    """
    Use full 'content' if available, otherwise use preview.
    Deduplicates on (article_id, char_start) to avoid repeating same chunk.
    """
    citations = dedupe_citations(citations)
    lines = []
    for i, c in enumerate(citations[:max_sources], start=1):
        tag = f"[S{i}]"
        meta = f"{tag} {c.get('title') or 'Untitled'} (KB: {c.get('kb_id')}, Article: {c.get('article_id')})"
        lines.append(meta)
        # prefer full chunk content saved as 'content'
        lines.append(c.get("content") or c.get("chunk") or c.get("preview") or "")
        lines.append("")  # blank line
    return "\n".join(lines)

SYSTEM_INSTRUCTIONS = """\
You are a practical technical support copilot for L2/L3 service engineers.
Use ONLY the provided sources to answer. If the sources contain the answer,
extract and summarize the actionable steps or troubleshooting guidance in
concise bullet points. Include inline citations like [S1], [S2] next to each
step that comes from a source. If the sources are incomplete, state what's
missing and provide the best possible guidance strictly grounded in the sources.
Do NOT hallucinate or introduce facts not present in the sources.
If there are no relevant sources, reply exactly:
"I was not able to find any solution from our knowledge base right now. You can try rephrasing or escalate."
"""

USER_PROMPT_TEMPLATE = """\
Question:
{question}

Sources:
{context}

Instructions:
- Use the sources to answer. If the source contains explicit steps, list them as bullets and add the source tag(s) after each bullet (e.g. [S1]).
- If the sources only partially answer, say what is missing and provide the best guidance you can, citing the sources used.
- If you cannot find any relevant information, reply exactly with:
"I was not able to find any solution from our knowledge base right now. You can try rephrasing or escalate."
- Keep answers concise and practical (4-8 bullets if applicable).
"""

'''
@app.post("/searchquery")
async def searchquery(payload: Dict[str, Any]):
    """
    Body: { "q": "...", "k": 6, "min_score": 0.25 }
    Returns: { "answer": "...", "citations": [...], "used_k": n }
    """
    print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
    print("USE_OPENAI =", USE_OPENAI)
    
    q = (payload.get("q") or "").strip()
    k = int(payload.get("k") or 6)
    min_score = float(payload.get("min_score") or 0.25)

    if not q:
        return JSONResponse({"success": False, "message": "Missing 'q'."}, status_code=400)

    hits = retrieve(q, k=k)
    citations = build_citations(hits, min_score=min_score)

    # Graceful fallback if nothing good enough
    if not citations:
        graceful = "I was not able to find any solution from our knowledge base right now. You can try rephrasing or escalate."
        return JSONResponse({
            "success": True,
            "answer": graceful,
            "citations": [],
            "used_k": 0
        })

    context = build_llm_context(citations)
    print("LLM context:\n", context)
    user_prompt = USER_PROMPT_TEMPLATE.format(question=q, context=context)
    print("User prompt:\n", user_prompt)

    # Synthesize with LLM if available; otherwise return extractive summary with citations only.
    if USE_OPENAI:
        completion = openai_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": user_prompt},
            ]
        )
        print("LLM raw response:", completion)
        answer = completion.choices[0].message.content.strip()
        print("LLM answer:", answer)
    else:
        # Zero-LLM fallback: stitch the most relevant previews with source tags.
        parts = [f"- {c['preview']} [S{c['rank']}]" for c in citations[:3] if c.get("preview")]
        answer = "\n".join(parts) if parts else \
            "I was not able to find any solution from our knowledge base right now. You can try rephrasing or escalate."

    return JSONResponse({
        "success": True,
        "answer": answer,
        "citations": citations,
        "used_k": len(citations),
    })

'''

@app.post("/searchquery")
async def searchquery(payload: Dict[str, Any]):
    q = (payload.get("q") or "").strip()
    k = int(payload.get("k") or 6)
    min_score = float(payload.get("min_score") or 0.25)

    if not q:
        return JSONResponse({"success": False, "message": "Missing 'q'."}, status_code=400)

    hits = retrieve(q, k=k)
    citations = build_citations(hits, min_score=min_score)

    if not citations:
        return JSONResponse({
            "success": True,
            "answer": "I was not able to find any solution from our knowledge base right now. You can try rephrasing or escalate.",
            "citations": [],
            "used_k": 0
        })

    # dedupe and build context (full content)
    citations = dedupe_citations(citations)
    context = build_llm_context(citations, max_sources=min(len(citations), 6))
    user_prompt = USER_PROMPT_TEMPLATE.format(question=q, context=context)

    # Try LLM (if configured), with robust fallback to extractive
    fallback_text = "I was not able to find any solution from our knowledge base right now. You can try rephrasing or escalate."
    answer = None

    if USE_OPENAI:
        try:
            print("LLM context:\n", context)  # debug
            print("User prompt:\n", user_prompt[:2000])  # avoid extremely long logs
            completion = openai_client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=512,
            )
            # Extract assistant text safely
            answer = completion.choices[0].message.content.strip() if completion.choices else None
            # If model returned exact refusal or empty, treat as failure
            if not answer or answer.strip() == fallback_text:
                answer = None
        except Exception as e:
            print("LLM call failed, falling back to extractive. Error:", e)
            answer = None

    # If no LLM answer, build an extractive, cited reply from top chunks
    if not answer:
        parts = []
        for i, c in enumerate(citations[:3], start=1):
            text = (c.get("content") or c.get("chunk") or c.get("preview") or "").strip()
            if text:
                parts.append(f"[S{i}] {text}")
        answer = "\n\n".join(parts) if parts else fallback_text

    return JSONResponse({
        "success": True,
        "answer": answer,
        "citations": citations,
        "used_k": len(citations),
    })