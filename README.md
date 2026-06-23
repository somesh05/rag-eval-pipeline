# rag-eval-pipeline

-----

   
**Description** 

An AI-powered assistant that lets healthcare professionals upload clinical documents and ask plain-English questions, returning answers grounded strictly in the source material with page-level citations

**Domain:**

Healthcare / Medical Technology (**relevant to NHS Digital, BUPA, Babylon Health, Philips Healthcare, and other healthtech startups**)

**Problem solved:** 

Clinicians and healthcare staff routinely need to search long documents — discharge summaries, NICE guidelines, lab reports, drug interaction sheets — for specific facts under time pressure. Manual searching is slow and error-prone, while free-form AI chatbots risk hallucinating medical facts. This tool retrieves the exact relevant passage from the uploaded document before generating an answer, so every response is traceable to its source. 

**Features**

- Upload one or more clinical PDFs (discharge summaries, guidelines, lab reports) directly through the app
- Ask natural-language clinical questions and receive answers grounded only in the uploaded document — no fabricated facts
- Every answer includes an expandable Sources panel showing the exact page number and passage used
- Built-in safety disclaimer reinforcing the tool's role as a research aid, not a clinical decision-maker
- Fast semantic retrieval over document chunks using FAISS vector search, so answers return in seconds even on long PDFs

**Tech Stack**

Layer                                           Technology
LLM                                   Groq API — llama-3.3-70b-versatile
Orchestration                         LangChain (RetrievalQA pipeline)
Vector search                                   FAISS (CPU)
Embeddings                            sentence-transformers — all-MiniLM-L6-v2
PDF parsing                                       PyMuPDF
Interface                                          Gradio
Secrets management                              python-dotenv

**How to Run Locally**

#### 1. Clone the repository
git clone https://github.com/yourusername/medical-chatbot-assistant.git
cd medical-chatbot-assistant
####  2. Create and activate a virtual environment
python -m venv medenv
source medenv/bin/activate        # Windows: medenv\Scripts\activate
####  3. Install dependencies
pip install langchain langchain-community faiss-cpu PyMuPDF \
            sentence-transformers groq langchain-groq gradio python-dotenv

####  4. Add your API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

####  5. Launch the app
python app.py

The app will start a local Gradio server (typically at a temporary public link to the console if http://127.0.0.1:7860 ) and print a shareable share=True is set in demo.launch().

**Live Demo link** : https://huggingface.co/spaces/somesh05/medical-chatbot-assistant
