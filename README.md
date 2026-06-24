# RAG Evaluation Pipeline

A document Q&A system that pairs **hybrid retrieval** (BM25 + vector search) with a **built-in, automated evaluation harness** (RAGAS), so changes to retrieval strategy, chunking, or prompting can be measured against a fixed benchmark instead of eyeballed by hand.

Built with ChromaDB, sentence-transformers, Groq (Llama 3.1 8B for generation, GPT-OSS 120B as RAGAS judge), RAGAS, and Gradio.

## What it does

- **Ingests** PDF, TXT, or MD documents and splits them into overlapping chunks (500 chars, 50 overlap).
- **Embeds** chunks locally with `BAAI/bge-base-en-v1.5` (no API cost) and stores them in a persistent ChromaDB collection.
- **Retrieves** with either pure vector search or a hybrid mode that blends vector similarity with BM25 keyword scoring (50/50 weighted fusion) — hybrid catches exact-match terms that dense embeddings alone can miss.
- **Generates** grounded answers via Groq's `llama-3.1-8b-instant`, with a system prompt that explicitly tells the model to only use the retrieved context and admit when it doesn't know.
- **Evaluates** the whole pipeline automatically: a fixed "golden dataset" of question/ground-truth pairs is run through retrieval + generation, then scored on **faithfulness**, **answer relevancy**, and **context precision** using RAGAS with `openai/gpt-oss-120b` as an LLM judge (see CASE_STUDY.md for why this choice matters).
- **Exposes** all of this through a two-tab Gradio app — a Chat tab for live Q&A, and an Evaluation Dashboard tab for running and comparing benchmark passes.

## Architecture

| Component | Implementation | Why |
|---|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` (local, CPU) | Free, no API dependency, strong MTEB performance for its size |
| Vector store | ChromaDB (persistent, local) | Zero infrastructure cost |
| Keyword retrieval | `rank_bm25` (BM25Okapi) | Catches exact-match queries dense vectors underweight |
| Generation | `llama-3.1-8b-instant` via Groq | Free, fast, on a separate token quota from the judge model |
| Evaluation | RAGAS, `openai/gpt-oss-120b` as judge | Stronger structured-output reliability than smaller free models (see CASE_STUDY.md) |

**Document path:** upload → chunk & embed → store in Chroma + BM25 index.
**Query path:** ask a question → retrieve chunks (hybrid or vector) → generate a grounded answer via Groq.
**Evaluation path:** golden Q&A set → run through both paths above, in both retrieval modes → RAGAS scores each answer → dashboard shows per-question and averaged results.

## Project structure

```
rag-eval-pipeline/
├── app.py              # Gradio UI: Chat tab + Evaluation Dashboard tab
├── ingestion.py         # Document loading + chunking
├── vector_store.py     # Embedding, ChromaDB, BM25, hybrid retrieval
├── generation.py       # Grounded answer generation via Groq
├── evaluation.py       # RAGAS-based evaluation harness
├── golden_dataset.json  # Question / ground-truth pairs used for evaluation
├── requirements.txt
├── chroma_db/           # Created automatically — persistent vector store
└── results/             # Created automatically — timestamped CSV of each eval run
```

## Setup

1. **Clone and create a virtual environment**
   ```bash
   git clone https://github.com/somesh05/rag-eval-pipeline.git
   cd rag-eval-pipeline
   python -m venv env
   # Windows: env\Scripts\activate
   # macOS/Linux: source env/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Groq API key.** Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_key_here
   ```

4. **Run it**
   ```bash
   python app.py
   ```
   Open the printed local URL (typically `http://127.0.0.1:7860`).

## Usage

**Chat tab:** upload one or more PDF/TXT/MD files, click *Process Documents*, pick a retrieval mode (`hybrid` or `vector`), then ask questions. Each answer shows its source documents and basic latency/token usage.

**Evaluation Dashboard tab:** pick a retrieval mode and click *Run Evaluation Suite*. This runs every question in `golden_dataset.json` through the full pipeline, scores each answer with RAGAS, and shows averaged Faithfulness / Answer Relevancy / Context Precision plus a per-question results table. Each run is also saved to `results/` as a timestamped CSV, so you can compare runs later.

To get meaningful results, replace the placeholder entries in `golden_dataset.json` with real questions and ground-truth answers about your own sample documents.


## Evaluation methodology

A golden dataset of question/ground-truth pairs is run through the pipeline twice — once per retrieval mode — with chunk size, overlap, `k`, and the generation model held constant, isolating retrieval strategy as the only variable. The full write-up, including real results once both passes have been run, lives in [`CASE_STUDY.md`](./CASE_STUDY.md).

## Limitations

- The golden dataset is small by design (portfolio-scale); a production evaluation would want 50+ questions for tighter confidence intervals.
- RAGAS's LLM-as-judge approach is sensitive to the judge model's ability to follow structured output instructions, and this varies by metric. See CASE_STUDY.md section 4.1 for a documented case where two of three metrics remained unreliable even after switching to a stronger judge model.
- The hybrid fusion weighting (0.5 vector / 0.5 BM25) is fixed, not tuned or learned.

## Future work

- Tune or learn the hybrid fusion weight instead of using a fixed 50/50 split.
- Expand the golden dataset for tighter statistical confidence.
- Add a re-ranking stage after initial retrieval.
- Test retrieval performance as the document collection scales beyond a handful of files.

**Live Demo Link:**
