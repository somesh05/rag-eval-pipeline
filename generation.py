
import os
import time
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
 
load_dotenv()
 
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

 
ANSWER_MODEL = "llama-3.3-70b-versatile"
 
SYSTEM_PROMPT = """You are a careful, precise assistant that answers questions \
using ONLY the provided context. Follow these rules strictly:
 
1. Only use information that appears in the context below.
2. If the context does not contain enough information to answer the \
question, say "I don't have enough information in the provided documents \
to answer that" instead of guessing.
3. When you do answer, be concise and directly address the question.
4. Do not mention "the context" explicitly in your answer — just answer \
naturally, as if you know the information.
"""

def build_context_string(chunks: List[Dict]) -> str:
    parts= []
    for i, chunk in enumerate(chunks):
        source= chunk.get("source", "unknown")
        parts.append (f" [Source {i +1}: {source}]\n {chunk ['text']}")
    return "\n\n".join(parts)


def generate_answer(question:str, rertieved_chunks: List [Dict]) -> Dict:

    context_str= build_context_string(rertieved_chunks)
    user_message= f""" Context: \n{context_str}\n\nQuestion: {question}"""

    start_time= time.time()
    response= client.chat.completions.create(
        model=ANSWER_MODEL,
        messages= [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature= 0.1,
    )

    latency= time.time()- start_time

    answer_text= response.choices[0]. message.content
    sources_used= sorted (set (c.get ("source", "unknown") for c in rertieved_chunks))

    return {
        "answer": answer_text,
        "sources": sources_used,
        "latency_seconds": round(latency, 2),
        "input_tokens": response. usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }