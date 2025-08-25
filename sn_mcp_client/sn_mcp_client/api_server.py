from fastapi import FastAPI, Query, BackgroundTasks, UploadFile, File, Query
from fastapi.testclient import TestClient  # For internal API call
from typing import Optional
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

# Path to your existing config
CONFIG_PATH = "config.client.ssc.json"

# Load configuration once
cfg = MCPConfig.load(CONFIG_PATH)


# Directories
DOWNLOAD_DIR = "downloads"  # Source folder with PDF files
UPLOAD_DIR = "uploads"      # Destination folder for processed outputs
os.makedirs(UPLOAD_DIR, exist_ok=True)

# FAISS & metadata storage paths (inside uploads folder)
INDEX_PATH = os.path.join(UPLOAD_DIR, "vector_index.faiss")
METADATA_PATH = os.path.join(UPLOAD_DIR, "vector_metadata.pkl")

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dimension = 384  # for all-MiniLM-L6-v2

# Initialize FAISS index
if os.path.exists(INDEX_PATH):
    faiss_index = faiss.read_index(INDEX_PATH)
else:
    faiss_index = faiss.IndexFlatL2(embedding_dimension)

# Load metadata if exists
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "rb") as f:
        metadata_store = pickle.load(f)
else:
    metadata_store = []  # List of dicts: [{filename, chunk_text}, ...]


app = FastAPI(title="ServiceNow MCP REST API")

def pretty(obj):
    try:
        #return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
        
        # Case 1: If already a dict, pretty-print it
        if isinstance(obj, dict):
            return pretty_content(obj)

        # Case 2: If it looks like the messy TextContent(...) string
        if isinstance(obj, str) and "text='" in obj:
            return extract_and_format(obj)

        # Case 3: If it's an object but has a `text` attribute (like TextContent)
        if hasattr(obj, "text"):
            return extract_and_format(str(obj))

        # Case 4: Fallback for other objects
        return extract_and_format(str(obj))
        
        '''
        #json = json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
        #return pretty_content(json)
        # Directly handle dicts or lists
        if isinstance(obj, dict):
            return pretty_content(obj)
        if hasattr(obj, "__dict__"):  # Convert custom objects
            return pretty_content(obj.__dict__)
        return pretty_content(obj)
        '''
    except Exception:
        return str(obj)

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
    result = await call_tool(cfg, "list_knowledge_bases", {})
    return pretty(result)

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
    result = await call_tool(cfg, "list_articles", args)
    return pretty(result)

'''
@app.get("/get-article")
async def get_article(article_id: str = Query(..., description="Article sys_id")):
    tools = await list_tools(cfg)
    target = next((t for t in tools if getattr(t, "name", "") == "get_article"), None)
    key = "id"
    if target:
        key = choose_arg_key(getattr(target, "input_schema", {}) or {},
                             ["sys_id", "id", "article_id", "record_id"],
                             "id")
    args = {key: article_id}
    result = await call_tool(cfg, "get_article", args)
    return pretty(result)
'''

@app.get("/get-article")
async def get_article(article_id: str = Query(..., description="Article sys_id")):
    args = {"article_id": article_id}
    result = await call_tool(cfg, "get_article", args)
    return pretty(result)


@app.get("/download")
async def download_kb_and_articles():
    print("Starting download of all articles in each KB...")
    DOWNLOAD_DIR = "downloads"
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    client = TestClient(app)
    timeout = 300  # 5 minutes

    # Step 1: Get KB IDs
    response = client.get("/list-kbs", timeout=timeout)
    if response.status_code != 200:
        return {"success": False, "message": "Failed to fetch knowledge bases"}

    kb_data = response.json()
    kb_ids = [kb["id"] for kb in kb_data.get("knowledge_bases", [])]

    # Step 2 & 3: Fetch article IDs & download articles
    article_ids_map = {}
    total_articles_downloaded = 0

    for kb_id in kb_ids:
        articles_resp = client.get(f"/list-articles?kb={kb_id}", timeout=timeout)
        if articles_resp.status_code != 200:
            print(f"Failed to fetch articles for KB {kb_id}")
            continue

        articles_data = articles_resp.json()
        article_ids = [a["id"]["value"] for a in articles_data.get("articles", [])]
        article_ids_map[kb_id] = article_ids

        for article_id in article_ids:
            print(f"Downloading article {article_id} from KB {kb_id}...")
            article_resp = client.get(f"/get-article?article_id={article_id}", timeout=timeout)
            if article_resp.status_code == 200:
                article_data = article_resp.json().get("article", {})
                title = article_data.get("title", "Untitled Article")
                html_text = article_data.get("text", "")
                plain_text = html_to_text(html_text)

                safe_title = sanitize_filename(title)[:50]  # Limit filename length
                pdf_path = os.path.join(DOWNLOAD_DIR, f"{total_articles_downloaded}_{article_id}_{safe_title}.pdf")

                # Generate PDF for each article
                c = canvas.Canvas(pdf_path, pagesize=LETTER)
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, 750, title)
                c.setFont("Helvetica", 12)
                y = 720
                for line in plain_text.splitlines():
                    c.drawString(50, y, line[:90])  # wrap long lines
                    y -= 20
                    if y < 50:
                        c.showPage()
                        c.setFont("Helvetica", 12)
                        y = 750
                c.save()

                total_articles_downloaded += 1
                print(f"Saved article {article_id} as PDF: {pdf_path}")
                print(f"Downloaded {total_articles_downloaded} articles so far...")
            else:
                print(f"Failed to download article {article_id}")

    print("Download completed.")
    return JSONResponse(content={
        "success": True,
        "KBs": {
            "kb_ids": kb_ids,
            "count": len(kb_ids)
        },
        "article_ids": {
            **article_ids_map,
            "count": sum(len(v) for v in article_ids_map.values())
        },
        "articles": {
            "downloaded": True,
            "count": total_articles_downloaded
        }
    })  


@app.get("/upload")
async def upload_files_from_downloads():
    print("Starting upload and indexing of PDF files from downloads...")
    count_of_files = 0
    DOWNLOAD_DIR = "downloads"
    stored_files = []
    chunk_size = 500

    # Get all PDF files from downloads folder
    pdf_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        return JSONResponse(content={
            "success": False,
            "message": "No PDF files found in the downloads folder."
        })

    for filename in pdf_files:
        file_path = os.path.join(DOWNLOAD_DIR, filename)
        stored_files.append(filename)

        # Step 1: Read PDF content
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

        # Step 2: Chunk content
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Step 3: Generate embeddings
        embeddings = embedding_model.encode(chunks)
        embeddings = np.array(embeddings).astype('float32')

        # Step 4: Add to FAISS index
        faiss_index.add(embeddings)

        # Step 5: Store metadata for each chunk
        for chunk in chunks:
            metadata_store.append({
                "filename": filename,
                "chunk": chunk
            })
        count_of_files += 1
        print(f"Processed {filename}, total files processed: {count_of_files}")
    print("Upload and indexing completed. Processed {count_of_files} files.")


    # Persist FAISS index & metadata
    faiss.write_index(faiss_index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    return JSONResponse(content={
        "success": True,
        "message": f"Processed and indexed {len(stored_files)} PDF(s) from downloads.",
        "files": stored_files,
        "total_vectors": faiss_index.ntotal
    })


@app.get("/search")
async def search(query: str = Query(..., description="Search query"),
                 top_k: int = Query(5, description="Number of top results to return")):
    if faiss_index.ntotal == 0:
        return JSONResponse(content={
            "success": False,
            "message": "No data found in the FAISS index. Please upload files first."
        })

    # Step 1: Create embedding for the query
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    # Step 2: Perform FAISS search
    distances, indices = faiss_index.search(query_embedding, top_k)

    # Step 3: Retrieve metadata
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




#Utility Methods

def pretty_content(result):
    #print("called pretty_content with:", result)

    # Here, you can directly format if result contains the JSON dict
    return json.dumps(result, indent=4)

    '''
    # If result is already a dict, don't json.loads() again
    if isinstance(result, dict):
        outer_data = result
    else:
        # Try to parse string as JSON
        outer_data = json.loads(result)

    # Extract inner JSON string
    inner_json_str = outer_data["content"][0]["text"]

    # Parse inner JSON
    inner_data = json.loads(inner_json_str)

    # Pretty-print
    formatted_json = json.dumps(inner_data, indent=4)
    print(formatted_json)
    return formatted_json
    '''
    
    '''return {
        "success": not result.get("isError", False),
        "data": result.get("content"),
        "raw": result
    }'''


def extract_and_format(raw_string):

    #print("called extract_and_format with:", raw_string)

    # Step 1: Extract the text='...' value using regex
    #match = re.search(r"text='(.*?)', annotations", raw_string)
    match = re.search(r"text='(.*?)'(?:,|\))", raw_string, re.S)  # allows ) or , after the string
    
    if not match:
        return "No JSON found in text field"
    
    inner_raw = match.group(1)  # The JSON string with escapes

    # Step 2: Unescape the string (convert \\n -> \n and \\\" -> \")
    unescaped = inner_raw.encode('utf-8').decode('unicode_escape')

    # Step 3: Parse JSON
    data = json.loads(unescaped)

    # Step 4: Pretty print JSON
    # return json.dumps(data, indent=4)
    new_json = json.dumps(data, indent=4)
    return json.loads(new_json)

# Utility: Sanitize filename
def sanitize_filename(name: str) -> str:
    # Replace forbidden characters and control chars
    sanitized = re.sub(r'[\\/*?:"<>|\r\n\t]+', "_", name)
    sanitized = sanitized.strip().strip(".")  # remove trailing dots/spaces
    return sanitized[:100]  # Limit to 100 chars for safety

# Utility: Convert HTML to plain text
def html_to_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n").strip()
