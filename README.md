# SEC Filing Intelligence Engine 📊

A native, production-grade Retrieval-Augmented Generation (RAG) pipeline built to ingest multi-file SEC documents (10-K, 10-Q, 8-K), isolate key metric tables, and deliver conversational financial analysis alongside cited ground-truth evidence.

## 🛠 Architecture Overview
Unlike standard boilerplate AI projects, this system completely bypasses heavy, recognizable frameworks (like LangChain) to maintain zero-trace optimization and high mathematical execution speeds.

* **Layout-Aware Extraction:** Pre-processes raw SEC HTML/XBRL tables into structured Markdown tables using `BeautifulSoup4` and `html2text`, filtering out database noise.
* **Hybrid Search Core:** Merges lexical exactness (`BM25 Keyword Matching`) with contextual deep learning (`all-MiniLM-L6-v2` semantic vectors) via a 50/50 score fusion matrix.
* **Local Persistence Layer:** Caches text chunks and pre-compiled embedding matrices to disk (`json`/`pickle`), dropping local reruns to zero server load.
* **Structured Risk Assessment:** Directs financial fragments into a low-temperature `Gemini 3.5 Flash` model to output qualitative summaries and automated corporate risk ratings (Scale 1–10).

---

## 🧰 Tech Stack & Tooling Matrix

The platform is engineered using a granular, open-source data science ecosystem. By avoiding high-level abstractions, each tool maps directly to a physical stage in the pipeline:

[Image of localized data storage persistence architecture diagram showing a raw document parsing into local text and embedding cache folders before feeding a search query engine]

| Component / Library | Architectural Role | Functional Performance Impact |
| :--- | :--- | :--- |
| **Streamlit** | Interface & App State | Renders a dual-column analytical layout over native WebSocket connections (`localhost:8501`), enabling lightweight state preservation during live queries. |
| **SentenceTransformers** | Semantic Vector Space | Loads `all-MiniLM-L6-v2` locally to map text chunks into 384-dimensional spatial arrays, isolating deep conceptual relationships across long text spreads. |
| **Rank-BM25** | Lexical Search Core | Runs statistical token-frequency scaling via `BM25Okapi` to instantly flag strict numerical expressions, specific asset classes, and hardcoded dates. |
| **NumPy** | High-Speed Vector Math | Computes array dot-products and Euclidean normalization values inside C-level RAM buffers, eliminating the need for complex, heavy vector database servers. |
| **BeautifulSoup4** | Document DOM Stripping | Decomposes script wrappers, styles, and non-narrative XML tags from raw corporate filings to isolate core document structures. |
| **html2text** | Structural Formatting | Maps raw tabular HTML structures cleanly into Markdown layouts, preserving financial table alignments during token splitting. |
| **Google GenAI SDK** | Cognitive Reasoning Layer | Communicates with the `gemini-3.5-flash` engine using a low-temperature constraint ($0.15$) to produce strict context-bound syntheses and risk scores. |
