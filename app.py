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

def handle_run_evaluation(retrieval_mode):
    results_df = run_evaluation(store, retrieval_mode=retrieval_mode)
    summary = summarize_results(results_df)
    save_results(results_df, label=f"eval_{retrieval_mode}")

    summary_text = (
        f"**Average Faithfulness:** {summary.get('faithfulness', 'N/A')}\n\n"
        f"**Average Answer Relevancy:** {summary.get('answer_relevancy', 'N/A')}\n\n"
        f"**Average Context Precision:** {summary.get('context_precision', 'N/A')}"
    )
    return summary_text, results_df


with gr.Blocks(title="RAG Evaluation Pipeline") as demo:
    gr.Markdown("# RAG Evaluation Pipeline\nA document Q&A system with a built-in, automated evaluation harness.")

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=1):
                file_upload = gr.File(file_count="multiple", label="Upload PDF or TXT documents")
                upload_button = gr.Button("Process Documents")
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

if __name__ == "__main__":
    demo.launch()