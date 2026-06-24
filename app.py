import os
import gradio as gr
import pandas as pd

from ingestion import process_uploaded_files
from vector_store import RAGVectorStore
from generation import generate_answer
from evaluation import run_evaluation, summarize_results, save_results

store = RAGVectorStore(persist_directory="./chroma_db")

def handle_upload(files):
    if not files:
        return "No files uploaded."
    file_paths = [f.name for f in files]
    chunks = process_uploaded_files(file_paths)
    store.add_documents(chunks)
    return f"Processed {len(file_paths)} file(s) into {len(chunks)} chunks. Ready to answer questions."

def handle_clear_documents():
    store.clear()
    return "All documents cleared. Upload new documents to start fresh.", []

def handle_chat(message, history, retrieval_mode):
    retrieved = store.retrieve(message, k=4, mode=retrieval_mode)

    if not retrieved:
        answer = "I don't have any documents loaded yet — please upload some first."
        sources_display = ""
    else:
        result = generate_answer(message, retrieved)
        answer = result["answer"]
        sources_display = "\n\n**Sources used:** " + ", ".join(result["sources"])
        sources_display += f"\n\n*({result['latency_seconds']}s, {result['input_tokens']}+{result['output_tokens']} tokens)*"

    full_response = answer + sources_display
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": full_response},
    ]
    return history, ""

def _format_metric_line(summary, key, label):
    score = summary.get(key)
    n_valid = summary.get(f"{key}_n_valid", 0)
    n_total = summary.get(f"{key}_n_total", 0)
    low_sample = summary.get(f"{key}_low_sample", False)
    score_str = f"{score:.3f}" if score is not None else "N/A"
    line = f"**Average {label}:** {score_str}  *(based on {n_valid}/{n_total} questions)*"
    if low_sample:
        line += "  ⚠️ fewer than half the questions produced a usable score — treat this number with caution"
    return line

def handle_run_evaluation(retrieval_mode):
    results_df = run_evaluation(store, retrieval_mode=retrieval_mode)
    summary = summarize_results(results_df)
    save_results(results_df, label=f"eval_{retrieval_mode}")

    summary_text = "\n\n".join([
        _format_metric_line(summary, "faithfulness", "Faithfulness"),
        _format_metric_line(summary, "answer_relevancy", "Answer Relevancy"),
        _format_metric_line(summary, "context_precision", "Context Precision"),
    ])
    return summary_text, results_df

def _format_comparison_markdown(vector_summary, hybrid_summary):
    metrics = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("context_precision", "Context Precision"),
    ]
    lines = [
        "**Vector-only vs Hybrid comparison**",
        "",
        "| Metric | Vector-only | Hybrid | Change | Sample size (valid/total) |",
        "|---|---|---|---|---|",
    ]
    any_low_sample = False
    for key, label in metrics:
        v = vector_summary.get(key)
        h = hybrid_summary.get(key)
        v_n, v_total = vector_summary.get(f"{key}_n_valid", 0), vector_summary.get(f"{key}_n_total", 0)
        h_n, h_total = hybrid_summary.get(f"{key}_n_valid", 0), hybrid_summary.get(f"{key}_n_total", 0)
        sample_str = f"vector {v_n}/{v_total}, hybrid {h_n}/{h_total}"
        if vector_summary.get(f"{key}_low_sample") or hybrid_summary.get(f"{key}_low_sample"):
            sample_str += " ⚠️"
            any_low_sample = True
        if v is None or h is None:
            lines.append(f"| {label} | {v if v is not None else 'N/A'} | {h if h is not None else 'N/A'} | N/A | {sample_str} |")
            continue
        change = h - v
        pct = f" ({change / v * 100:+.1f}%)" if v else ""
        lines.append(f"| {label} | {v:.3f} | {h:.3f} | {change:+.3f}{pct} | {sample_str} |")
    if any_low_sample:
        lines.append("")
        lines.append("⚠️ At least one metric above is based on fewer than half the golden-dataset questions producing a usable score (RAGAS failed to parse the judge model's output for the rest). Treat flagged numbers as unreliable until investigated.")
    return "\n".join(lines)

def handle_run_both_evaluations():
    vector_df = run_evaluation(store, retrieval_mode="vector")
    vector_summary = summarize_results(vector_df)
    save_results(vector_df, label="eval_vector")

    hybrid_df = run_evaluation(store, retrieval_mode="hybrid")
    hybrid_summary = summarize_results(hybrid_df)
    save_results(hybrid_df, label="eval_hybrid")

    comparison_text = _format_comparison_markdown(vector_summary, hybrid_summary)

    vector_df = vector_df.copy()
    vector_df.insert(0, "mode", "vector")
    hybrid_df = hybrid_df.copy()
    hybrid_df.insert(0, "mode", "hybrid")
    combined_df = pd.concat([vector_df, hybrid_df], ignore_index=True)

    return comparison_text, combined_df

with gr.Blocks(title="RAG Evaluation Pipeline") as demo:
    gr.Markdown("# RAG Evaluation Pipeline\nA document Q&A system with a built-in, automated evaluation harness.")

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=1):
                file_upload = gr.File(file_count="multiple", label="Upload PDF or TXT documents")
                upload_button = gr.Button("Process Documents")
                clear_button = gr.Button("Clear All Documents", variant="stop")
                upload_status = gr.Textbox(label="Status", interactive=False)
                retrieval_mode_dropdown = gr.Dropdown(
                    choices=["hybrid", "vector"],
                    value="hybrid",
                    label="Retrieval Mode",
                )
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=450)
                msg_box = gr.Textbox(label="Ask a question about your documents")

        upload_button.click(handle_upload, inputs=[file_upload], outputs=[upload_status])
        clear_button.click(handle_clear_documents, inputs=[], outputs=[upload_status, chatbot])
        msg_box.submit(
            fn=handle_chat,
            inputs=[msg_box, chatbot, retrieval_mode_dropdown],
            outputs=[chatbot, msg_box],
        )

    with gr.Tab("Evaluation Dashboard"):
        gr.Markdown("Run the golden-dataset evaluation suite against the current retrieval mode and see how well the pipeline is performing.")
        eval_mode_dropdown = gr.Dropdown(choices=["hybrid", "vector"], value="hybrid", label="Retrieval Mode To Evaluate")
        run_eval_button = gr.Button("Run Evaluation Suite", variant="primary")
        eval_summary = gr.Markdown()
        eval_table = gr.Dataframe(label="Per-Question Results")

        run_eval_button.click(
            handle_run_evaluation,
            inputs=[eval_mode_dropdown],
            outputs=[eval_summary, eval_table],
        )

        gr.Markdown("---\nOr run both modes back-to-back and compare them directly (this takes roughly twice as long, since it evaluates the full golden dataset once per mode).")
        compare_button = gr.Button("Run Both Modes (Compare)", variant="secondary")
        comparison_summary = gr.Markdown()
        comparison_table = gr.Dataframe(label="Per-Question Results — Both Modes")

        compare_button.click(
            handle_run_both_evaluations,
            inputs=[],
            outputs=[comparison_summary, comparison_table],
        )

if __name__ == "__main__":
    demo.launch()
