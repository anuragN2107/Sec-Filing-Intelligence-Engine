import os
import re
import glob
import time
import json
import pickle
import streamlit as st
import numpy as np
from bs4 import BeautifulSoup
import html2text
import requests
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- STORAGE CONFIGURATION ROOTS ---
STORAGE_DIR = "local_vault"
CHUNKS_VAULT = os.path.join(STORAGE_DIR, "text_chunks")
VECTORS_VAULT = os.path.join(STORAGE_DIR, "vector_indices")

# Ensure permanent storage infrastructure directories exist locally
os.makedirs(CHUNKS_VAULT, exist_ok=True)
os.makedirs(VECTORS_VAULT, exist_ok=True)


# --- UTILITY CLEANING & PARSING ENGINE ---
def is_meaningless_metadata(line: str) -> bool:
    line_strip = line.strip()
    if not line_strip: return True
    if "http://" in line_strip or "https://" in line_strip or "fasb.org" in line_strip: return True
    if "us-gaap:" in line_strip or "xbrli:" in line_strip or "aapl:" in line_strip: return True
    if re.match(r'^[0-9\-]{10,}$', line_strip): return True
    return False

def token_count_estimate(text: str) -> int:
    return int(len(text.split()) * 1.3)

def download_and_chunk_filing(target_url: str, ticker: str, year: str, email: str, max_tokens: int = 600) -> list:
    """Downloads an SEC document, strips XBRL metadata, and chunks it locally with context attributes."""
    headers = {"User-Agent": f"ResearchAgent {email}", "Accept-Encoding": "gzip, deflate"}
    
    response = requests.get(target_url, headers=headers)
    if response.status_code != 200:
        st.error(f"SEC servers rejected download for {ticker} ({year}). Code: {response.status_code}")
        return []
        
    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
        
    converter = html2text.HTML2Text()
    converter.bypass_tables = False
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0
    markdown_text = converter.handle(str(soup))
    
    raw_sections = markdown_text.split("* * *")
    chunks_built = []
    
    for sec_idx, section in enumerate(raw_sections):
        lines = section.split("\n")
        cleaned_lines = [line for line in lines if not is_meaningless_metadata(line)]
        section_cleaned = "\n".join(cleaned_lines).strip()
        
        if not section_cleaned or len(section_cleaned.split()) < 5:
            continue
            
        if token_count_estimate(section_cleaned) <= max_tokens:
            chunks_built.append({
                "ticker": ticker,
                "year": year,
                "text": section_cleaned
            })
        else:
            paragraphs = section_cleaned.split("\n\n")
            current_buffer = []
            current_tokens = 0
            for para in paragraphs:
                para_strip = para.strip()
                if not para_strip: continue
                para_tokens = token_count_estimate(para_strip)
                
                if current_tokens + para_tokens > max_tokens and current_buffer:
                    chunks_built.append({
                        "ticker": ticker,
                        "year": year,
                        "text": "\n\n".join(current_buffer)
                    })
                    current_buffer = [para_strip]
                    current_tokens = para_tokens
                else:
                    current_buffer.append(para_strip)
                    current_tokens += para_tokens
            if current_buffer:
                chunks_built.append({
                    "ticker": ticker,
                    "year": year,
                    "text": "\n\n".join(current_buffer)
                })
                
    return chunks_built


# --- STREAMLIT UI LAYOUT DESIGN ---
st.set_page_config(page_title="SEC Filing Intelligence Engine", layout="wide")
st.title("📊 SEC Filing Intelligence Engine")
st.caption("A multi-filing semantic RAG system with Local Disk Persistence and automated risk scoring.")

# Fallback checking ensures keys remain hidden from source code repositories
if "GEMINI_API_KEY" not in os.environ:
    # If running locally without terminal setups, you can place your key inside the quotes below temporarily.
    # CRITICAL: Revert to empty quotes BEFORE pushing changes to public web instances.
    os.environ["GEMINI_API_KEY"] = ""

# Sidebar controls for Batch Data Ingestion
with st.sidebar:
    st.header("1. Data Ingestion Matrix")
    user_email = st.text_input("SEC Compliance Email", value="analyst@firm.com")
    
    st.subheader("Select Scope to Index")
    target_ticker = st.selectbox("Company Target Ticker", ["AAPL", "MSFT"])
    target_years = st.multiselect("Historical Fiscal Years", ["2023", "2024", "2025"], default=["2023", "2024"])
    
    url_database = {
        "AAPL": {
            "2023": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
            "2024": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
            "2025": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000115/aapl-20250927.htm"
        },
        "MSFT": {
            "2023": "https://www.sec.gov/Archives/edgar/data/789019/000078901923000058/msft-20230630.htm",
            "2024": "https://www.sec.gov/Archives/edgar/data/789019/000078901924000068/msft-20240630.htm",
            "2025": "https://www.sec.gov/Archives/edgar/data/789019/000078901925000072/msft-20250630.htm"
        }
    }
    
    trigger_ingest = st.button("Build / Re-Index Vector Matrices", use_container_width=True)

# Cache compilation optimization layer
def compile_search_matrices(all_chunks: list):
    """Compiles combined vector and keyword layers for active data fragments."""
    if not all_chunks:
        return None
    texts = [c["text"] for c in all_chunks]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return model, embeddings, bm25, all_chunks


# --- ORCHESTRATION PIPELINE WITH DISK CACHING ---
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if trigger_ingest:
    master_chunks = []
    progress_bar = st.sidebar.progress(0)
    
    for idx, yr in enumerate(target_years):
        chunk_file_path = os.path.join(CHUNKS_VAULT, f"{target_ticker}_{yr}.json")
        
        # PERSISTENCE CHECK: Read from local storage if available
        if os.path.exists(chunk_file_path):
            st.sidebar.info(f"💾 Loading cached chunks for {target_ticker} ({yr})...")
            with open(chunk_file_path, "r", encoding="utf-8") as f:
                retrieved_chunks = json.load(f)
        else:
            # Otherwise, pull data down online
            st.sidebar.text(f"🌐 Fetching fresh SEC data: {target_ticker} ({yr})...")
            if target_ticker in url_database and yr in url_database[target_ticker]:
                url = url_database[target_ticker][yr]
                retrieved_chunks = download_and_chunk_filing(url, target_ticker, yr, user_email)
                
                # Save to disk permanently immediately
                if retrieved_chunks:
                    with open(chunk_file_path, "w", encoding="utf-8") as f:
                        json.dump(retrieved_chunks, f, ensure_ascii=False, indent=2)
            else:
                retrieved_chunks = []
                
        master_chunks.extend(retrieved_chunks)
        progress_bar.progress(int(((idx + 1) / len(target_years)) * 100))
        time.sleep(0.2)
        
    if master_chunks:
        # Create unique signature hash for the current selection query room
        selection_id = f"{target_ticker}_" + "_".join(sorted(target_years))
        vector_cache_path = os.path.join(VECTORS_VAULT, f"{selection_id}.pkl")
        
        # PERSISTENCE CHECK: Read compiled vectors from disk if available
        if os.path.exists(vector_cache_path):
            st.sidebar.info("💾 Loading pre-compiled mathematical vector matrices...")
            with open(vector_cache_path, "rb") as f:
                st.session_state.pipeline = pickle.load(f)
        else:
            # Compile new matrix spaces
            st.sidebar.text("🧮 Compiling vector maps...")
            compiled_pipeline = compile_search_matrices(master_chunks)
            if compiled_pipeline:
                st.session_state.pipeline = compiled_pipeline
                # Save matrices to disk permanently
                with open(vector_cache_path, "wb") as f:
                    pickle.dump(compiled_pipeline, f)
                    
        st.sidebar.success(f"Workspace Compiled! Loaded {len(master_chunks)} fragments.")
    else:
        st.sidebar.error("Ingestion failed. No valid chunks found.")

# --- ANALYST QUERY CHAT ENVIRONMENT ---
st.header("Natural-Language Intelligence Framework")

if st.session_state.pipeline is None:
    st.info("👈 Please select your target variables and hit 'Build / Re-Index Vector Matrices' in the sidebar to open the data room.")
else:
    model, embeddings, bm25, chunks = st.session_state.pipeline
    
    user_query = st.text_input("Enter your core macro query or cross-year evaluation target:", 
                              placeholder="Type your question here...")
    
    if user_query:
        with st.spinner("Executing hybrid lookup..."):
            query_vec = model.encode([user_query], convert_to_numpy=True)[0]
            dot_products = np.dot(embeddings, query_vec)
            norm_embeddings = np.linalg.norm(embeddings, axis=1)
            norm_query = np.linalg.norm(query_vec)
            norm_embeddings[norm_embeddings == 0] = 1.0
            semantic_scores = dot_products / (norm_embeddings * norm_query)
            
            tokenized_q = user_query.lower().split()
            keyword_scores = np.array(bm25.get_scores(tokenized_q))
            if np.max(keyword_scores) > 0:
                keyword_scores = keyword_scores / np.max(keyword_scores)
                
            hybrid_scores = (0.5 * semantic_scores) + (0.5 * keyword_scores)
            top_indices = np.argsort(hybrid_scores)[::-1][:4]
            
            context_payload = ""
            for rank, idx in enumerate(top_indices):
                item = chunks[idx]
                context_payload += f"\n--- EVIDENCE BLOCK {rank+1} (Company: {item['ticker']} | Fiscal Year: {item['year']}) ---\n{item['text']}\n"
                
            system_instruction = (
                "You are an Elite Wall Street Financial Analyst. Answer the user question relying STRICTLY and exclusively on the provided Evidence Blocks.\n"
                "Explicitly point out historical changes, trends, or comparisons if relevant.\n\n"
                "Format rules:\n"
                "Use clean markdown headers.\n"
                "Provide a 'Risk Score' from 1 to 10 with a short narrative justification."
            )
            
            try:
                client = genai.Client()
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=f"Context Metadata Room:\n{context_payload}\n\nUser Question: {user_query}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.15
                    )
                )
                
                col_left, col_right = st.columns([3, 2])
                with col_left:
                    st.subheader("Engine Intelligence Analysis")
                    st.markdown(response.text)
                    
                with col_right:
                    st.subheader("Extracted Ground-Truth Evidence")
                    for rank, idx in enumerate(top_indices):
                        item = chunks[idx]
                        with st.expander(f"Evidence {rank+1}: {item['ticker']} (FY {item['year']})"):
                            st.code(item['text'], language="text")
                            
            except APIError as e:
                st.error(f"Google Endpoint restriction: {e.message}")
            except Exception as e:
                st.error(f"System error: {e}")