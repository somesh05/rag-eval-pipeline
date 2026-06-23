# rag-eval-pipeline

-----

   
**Description** 

A document Q&A system that pairs hybrid retrieval (BM25 + vector search) with a built-in, automated evaluation harness (RAGAS), so changes to retrieval strategy, chunking, or prompting can be measured against a fixed benchmark instead of eyeballed by hand.

Built with ChromaDB, sentence-transformers, Groq (Llama 3.3 70B), RAGAS, and Gradio. 

**Features**

-Ingests PDF, TXT, or MD documents and splits them into overlapping chunks (500 chars, 50 overlap).

-Embeds chunks locally with BAAI/bge-base-en-v1.5 (no API cost) and stores them in a persistent ChromaDB collection.

-Retrieves with either pure vector search or a hybrid mode that blends vector similarity with BM25 keyword scoring (50/50 weighted fusion) — hybrid catches exact-match terms that dense embeddings alone can miss.

-Generates grounded answers via Groq's Llama 3.3 70B, with a system prompt that explicitly tells the model to only use the retrieved context and admit when it doesn't know.

-Evaluates the whole pipeline automatically: a fixed "golden dataset" of question/ground-truth pairs is run through retrieval + generation, then scored on faithfulness, answer relevancy, and context precision using RAGAS with Llama 3.1 8B as an LLM judge.

-Exposes all of this through a two-tab Gradio app — a Chat tab for live Q&A, and an Evaluation Dashboard tab for running and comparing benchmark passes.

**Tech Stack**

Embeddings                            BAAI/bge-base-en-v1.5 (local, CPU)
Keyword retrieval                         rank_bm25 (BM25Okapi)
Vector store                             ChromaDB (persistent, local)
Generation                                    Llama 3.3 70B via Groq
Evaluation                             RAGAS, Llama 3.1 8B as judge
Interface                                          Gradio
Secrets management                              python-dotenv

**Document path:** 

-**upload** → chunk & embed → store in Chroma + BM25 index.

-**Query path**: ask a question → retrieve chunks (hybrid or vector) → generate a grounded answer via Groq.

-**Evaluation path**: golden Q&A set → run through both paths above, in both retrieval modes → RAGAS scores each answer → dashboard shows per-question and averaged results.

**How to Run Locally**

#### 1. Clone the repository
git clone https://github.com/yourusername/rag-eval-pipeline.git
cd rag-eval-pipeline
####  2. Create and activate a virtual environment
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
####  3. Install dependencies
 pip install -r requirements.txt

####  4. Add your API key
GROQ_API_KEY=your_key_here

####  5. Launch the app
python app.py

The app will start a local Gradio server (typically at a temporary public link to the console if http://127.0.0.1:7860 ) and print a shareable share=True is set in demo.launch().

####  6. Evaluation methodology

A golden dataset of question/ground-truth pairs is run through the pipeline twice — once per retrieval mode — with chunk size, overlap, k, and the generation model held constant, isolating retrieval strategy as the only variable or comparing both modes at once. The full write-up, including real results once both passes have been run, lives in **CASE_STUDY.md.**

####  7. Limitations

-The golden dataset is small by design (portfolio-scale); a production evaluation would want 50+ questions for tighter confidence intervals.

-RAGAS's LLM-as-judge approach (Llama 3.1 8B here) introduces its own scoring variance, since the judge is itself an LLM rather than a ground-truth oracle.

-The hybrid fusion weighting (0.5 vector / 0.5 BM25) is fixed, not tuned or learned.

**Live Demo link** : 
