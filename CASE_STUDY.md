# Case Study: Building a RAG System That Can Prove It Works

*A hybrid retrieval & automated evaluation case study for [rag-eval-pipeline](./README.md).*


## Abstract

Retrieval-Augmented Generation (RAG) systems are widely deployed but rarely measured rigorously. This case study documents the design and evaluation of a document question-answering system that combines hybrid (BM25 + vector) retrieval with an automated RAGAS-based evaluation harness. While building the evaluation harness, a reliability problem emerged in the measurement process itself: the small, free LLM initially used as RAGAS's judge model frequently failed to produce the structured output RAGAS requires to score a response, causing repeated evaluation runs against identical data to produce inconsistent results with sample sizes too small to trust. Switching to a larger, separately-quota'd judge model resolved this for Context Precision, which showed a validated **26.4% improvement under hybrid retrieval** (n=11/19 and 12/19 questions, both above the reliability threshold) — consistent with BM25's keyword matching helping surface chunks containing exact terms that dense embeddings under-rank. Faithfulness and Answer Relevancy remained unreliably sampled even after the fix, and are reported as an open limitation rather than a conclusive finding. The broader lesson holds regardless: automated RAG evaluation is only as trustworthy as the judge doing the scoring, and that needs to be verified per-metric, not assumed.

## 1. Problem statement

Most RAG implementations are evaluated informally: a developer asks a handful of questions, reads the answers, and decides subjectively whether the system "seems good." This has two failure modes. First, it doesn't scale — as a knowledge base grows or a prompt is tweaked, re-checking by hand becomes impractical. Second, it isn't reproducible — two people reviewing the same outputs can reach different conclusions about quality.

This project addresses both failure modes by building an automated evaluation harness alongside the RAG system itself, so any change to retrieval strategy, chunking parameters, or prompting can be measured against a fixed, repeatable benchmark.

## 2. System design

### 2.1 Architecture

The system has four main components: an ingestion pipeline that chunks uploaded documents, a hybrid retrieval layer combining dense vector search with BM25 keyword search, a generation layer using a grounded prompting strategy, and an evaluation layer built on RAGAS.

| Component | Implementation choice | Rationale |
|---|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` (local, CPU) | Free, no API dependency, strong performance on retrieval benchmarks (MTEB) for its size |
| Vector store | ChromaDB (persistent, local) | Zero infrastructure cost, sufficient for portfolio-scale document collections |
| Keyword retrieval | BM25Okapi (`rank_bm25`) | Classic, well-understood algorithm; captures exact-match queries vector search alone misses |
| Generation model | `llama-3.1-8b-instant` via Groq | Free, fast, and on a separate token quota from the judge model — avoids the two competing for the same daily budget |
| Evaluation judge | `openai/gpt-oss-120b` via Groq | Stronger structured-output reliability than the smaller model originally tried (see section 4.1), and a separate quota pool from the generation model |

### 2.2 Why hybrid retrieval

Pure dense (vector) retrieval embeds both the query and document chunks into a shared semantic space and ranks by similarity. This captures paraphrase and conceptual similarity well, but can underperform on queries containing exact identifiers — product codes, proper nouns, or specific numbers — where lexical overlap matters more than semantic similarity. BM25 addresses this gap by scoring chunks on term frequency and inverse document frequency, which rewards exact keyword matches regardless of semantic framing.

The hybrid approach in this system retrieves the top `2k` candidates from each method independently, then merges them using a simple weighted score (0.5 × normalized vector similarity + 0.5 × normalized BM25 score), returning the top `k` overall. This is a deliberately simple fusion method — reciprocal rank fusion is a reasonable alternative — chosen because its behavior is easy to explain and debug.

## 3. Evaluation methodology

### 3.1 Golden dataset construction

A golden dataset of **[19]** question/ground-truth pairs was constructed manually from the sample documents, covering a mix of: direct factual lookups, questions requiring synthesis across multiple chunks, and at least **[3]** questions designed to be unanswerable from the provided documents, to test whether the system correctly declines rather than hallucinating.

### 3.2 Metrics

- **Faithfulness** — measures whether the generated answer's claims are actually supported by the retrieved context, addressing hallucination directly.
- **Answer relevancy** — measures whether the answer actually addresses the question asked, independent of factual grounding.
- **Context precision** — measures whether the retrieved chunks are relevant to the question, evaluating the retrieval step in isolation from generation.

### 3.3 Experimental procedure

The golden dataset was run through the full pipeline (retrieve, generate, evaluate) twice: once with `retrieval_mode="vector"` (pure dense retrieval) and once with `retrieval_mode="hybrid"`. All other parameters (chunk size 500, overlap 50, k=4, generation model and temperature) were held constant between runs, isolating retrieval strategy as the only variable.

## 4. Results

### 4.1 An unexpected finding: judge model reliability

Before any vector-vs-hybrid comparison could be trusted, a more fundamental problem surfaced during evaluation: the RAGAS judge model (`llama-3.1-8b-instant`, chosen initially for being fast and free) frequently failed to produce output in the structured format RAGAS requires to compute each metric. Rather than raising a visible error, RAGAS silently records a failed computation as a missing value, so the failure was invisible until the per-question results were inspected directly.

Two independent runs against the same 19-question golden dataset, same documents, and same retrieval code produced inconsistent results:

| Metric | Run 1: Vector → Hybrid | Run 2: Vector → Hybrid |
|---|---|---|
| Faithfulness | 0.000 (n=1/19) → 1.000 (n=1/19) | 0.667 (n=4/19) → N/A (n=0/19) |
| Answer Relevancy | 0.540 (n=7/19) → 0.703 (n=8/19) | 0.904 (n=4/19) → 0.699 (n=8/19) |
| Context Precision | 0.375 (n=4/19) → 0.618 (n=4/19) | 0.667 (n=3/19) → 0.535 (n=4/19) |

Sample sizes were small in every case (at best 8 of 19 questions producing a usable score), and the apparent "winner" between retrieval modes flipped between runs — strong evidence that the variation was coming from judge model output instability rather than from any real difference between vector and hybrid retrieval. Faithfulness was affected most severely, consistent with it being the most structurally demanding metric to compute (RAGAS decomposes the answer into discrete claims, then verifies each one against context — two chained steps that both require strict output formatting from the judge, versus Answer Relevancy's simpler embedding-similarity comparison).

**Resolution attempted:** two changes were made — the chat-generation model was moved to `llama-3.1-8b-instant` (separating it from the judge model's token budget, since both had been competing for the same Groq free-tier daily quota and triggering rate-limit errors), and the RAGAS judge was switched to `openai/gpt-oss-120b`, a larger, separately-quota'd model chosen for its stronger structured-output track record.

A third run after this change showed partial improvement:

| Metric | Run 3: Vector → Hybrid | Sample size |
|---|---|---|
| Faithfulness | 0.400 → 0.350 | vector 4/19, hybrid 5/19 ⚠️ |
| Answer Relevancy | 0.254 → 0.786 | vector 6/19, hybrid 5/19 ⚠️ |
| Context Precision | 0.553 → 0.699 | vector 11/19, hybrid 12/19 ✓ |

Switching judge models fixed the problem for Context Precision specifically — both modes finally cleared half the dataset for the first time across three runs. **Faithfulness and Answer Relevancy did not recover to a trustworthy sample size even with a stronger judge model**, suggesting the issue isn't purely "judge model capability" — it may be specific to how `ragas`'s multi-step prompting for those two metrics interacts with Groq-hosted open models through the `LangchainLLMWrapper` integration in this version of the library, which is a narrower and more specific hypothesis than the original one. This is documented as an open limitation (section 5) rather than something resolved within this project's scope.

### 4.2 Vector vs. hybrid retrieval comparison

Given the sample-size finding above, only **Context Precision** is reported as a validated comparison; Faithfulness and Answer Relevancy numbers are shown for completeness but should not be read as conclusive.

| Metric | Vector-only | Hybrid | Change | Valid sample |
|---|---|---|---|---|
| Context Precision | 0.553 | 0.699 | **+0.146 (+26.4%)** | 11/19, 12/19 — both reliable |
| Faithfulness | 0.400 | 0.350 | −0.050 (−12.5%) | 4/19, 5/19 — unreliable, do not interpret |
| Answer Relevancy | 0.254 | 0.786 | +0.532 (+209.4%) | 6/19, 5/19 — unreliable, do not interpret |

Hybrid retrieval produced a real, well-sampled **26.4% improvement in Context Precision** — the metric that most directly measures whether retrieval surfaced relevant chunks, independent of how the generator used them. This is consistent with the underlying hypothesis for using hybrid retrieval at all: BM25's keyword matching helps surface chunks containing exact terms (model names, percentages, acronyms like "RA-RAG" or "κ-RRSS") that dense embedding similarity alone can under-rank, even when it correctly identifies the general topic.

The Faithfulness and Answer Relevancy numbers above are not asserted as real findings. In particular, the apparent +209% jump in Answer Relevancy is built from only 5-6 questions per mode out of 19, and is at least partly an artifact of *which* questions happened to produce a parseable judge response in this particular run rather than a real measurement of typical relevancy — exactly the failure mode documented in section 4.1.

### 4.3 Failure case analysis

The clearest, most directly verifiable finding doesn't require RAGAS at all: **all three deliberately unanswerable questions** (the GPT-5 context-window question, the AWS hosting-cost question, and the author's institutional affiliation question) were correctly declined — "I don't have enough information in the provided documents to answer that" — in **both** retrieval modes, in this run. The system did not hallucinate an answer to any of them, which is the specific behavior the golden dataset's unanswerable questions were designed to test.

A clearer generation-level failure showed up on the question *"Compare the main advantage and limitation of fine-tuning versus RAG."* In **vector mode**, the model conflated the two methods — it attributed "high accuracy for specialized domains" to RAG, when the retrieved context actually lists that as fine-tuning's advantage, not RAG's. In **hybrid mode**, it failed differently: it stated "no specific advantages are mentioned for RAG" despite RAG's actual advantages (real-time retrieval, better scalability across domains) being present in the retrieved context chunks for that question. Both are synthesis failures rather than retrieval failures — the relevant text was retrieved in both cases, but the generation step didn't organize a two-method comparison cleanly. This suggests a next area to investigate would be the generation prompt's structure for comparison-style questions, separate from retrieval quality entirely.

## 5. Limitations

- The golden dataset size (19 questions) is small enough that individual question results carry meaningful variance; a production evaluation would benefit from 50+ questions for tighter confidence intervals.
- **RAGAS's LLM-as-judge approach is highly sensitive to the judge model's ability to follow structured-output instructions, and this sensitivity varies by metric, not just by model.** Across three evaluation runs, switching the judge model from `llama-3.1-8b-instant` to the larger `openai/gpt-oss-120b` fixed Context Precision's sample reliability (going from never exceeding 4/19 valid scores to consistently exceeding 11/19), but Faithfulness and Answer Relevancy remained stuck at 4-6/19 valid scores even with the stronger model. This points to something more specific than "small models are unreliable" — most likely an interaction between RAGAS's particular multi-step prompting for those two metrics and how `ChatGroq`/`LangchainLLMWrapper` handles that prompting in this library version. This was not root-caused further within this project's scope, and is flagged as the most valuable next debugging step.
- Two of the three metrics (Faithfulness, Answer Relevancy) should be considered **unvalidated** in this case study's current results, despite numbers being reported for completeness in section 4.2. Only the Context Precision comparison should be treated as a real finding.
- The hybrid fusion weighting (0.5/0.5) was not tuned systematically; a learned or validated weighting could plausibly outperform this fixed split.

## 6. Conclusion & future work

This project set out to compare vector-only against hybrid retrieval on three RAGAS metrics, and ended up surfacing a more fundamental problem first: the evaluation harness itself was not reliable until its judge model choice was scrutinized per-metric. Once that was accounted for, the one metric with a trustworthy sample size — Context Precision — showed a clear, real **26.4% improvement under hybrid retrieval** (0.553 → 0.699), supporting the original hypothesis that combining BM25 keyword matching with dense vector search surfaces more relevant context than vector search alone, particularly for questions containing specific named entities, model names, and statistics. Faithfulness and Answer Relevancy remain open questions rather than resolved findings.

Future work, in priority order: (1) root-cause why Faithfulness and Answer Relevancy specifically continue to fail to parse even with a stronger judge model — likely starting with inspecting `ragas`'s raw judge prompts and responses directly rather than only the final scores; (2) add an automated check to the evaluation harness that blocks reporting a metric's average whenever its valid sample size falls below a set threshold, rather than relying on manual inspection to catch it, as happened here; (3) expand the golden dataset beyond 19 questions; (4) investigate the comparison-question generation failure noted in section 4.3, independent of retrieval mode; (5) test a learned hybrid fusion weight instead of the fixed 0.5/0.5 split.

## References

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401.
- Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217.
- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
- BAAI (2023). *BGE: BAAI General Embedding.* Hugging Face Model Card, huggingface.co/BAAI/bge-base-en-v1.5.
