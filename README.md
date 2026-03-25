# RAG PDF Chatbot with TinyLlama

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that allows users to ask questions about a PDF document using a fully local language model and an interactive chat interface.

Compared to earlier versions of this project, this implementation focuses on:
- faster inference
- improved user interface
- better transparency through source display

---

## Key Improvements Over Basic RAG Versions

This version introduces several enhancements:

- Faster model: TinyLlama instead of larger LLMs
- Modern chat interface using Streamlit chat components
- Source document visualization for transparency
- Response time measurement for performance analysis
- Chat history management with session state

---

## System Architecture

```
PDF → Chunking → Embeddings → FAISS → Retriever → TinyLlama → Chat UI
```

The language model generates answers using only retrieved document context to reduce hallucinations.

---

## Tech Stack

- Python
- LangChain
- FAISS vector database
- Hugging Face Transformers
- Streamlit chat UI

Models and embeddings:
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

---

## ⚙️ How to run the project

### 1. Create environment

```
conda create -n rag_tinyllama python=3.10
conda activate rag_tinyllama
```

### 2. Install dependencies

```
Install requirements.txt
```

---

## Preparing the PDF

Place your document in:

```
data/
```

You can replace this file with any PDF you want to analyze.

---

## Creating the Vector Index

Run:

```
python ingest.py
```

This step:
- extracts text from the PDF
- splits it into overlapping chunks
- creates embeddings
- stores them in a FAISS index

---

## Running the Chatbot

```
In the terminal type: streamlit run app.py
```

Then open:
```
Usually it automatically opens up in browser or click:http://localhost:.....
```

---

## Chat Features

The Streamlit interface supports:

- persistent chat history
- clear chat button
- source document expansion panels
- response time measurement

Example user questions:
- "What skills are listed in the CV?"
- "What is the document about?"
- "What work experience does the candidate have?"
<img width="1897" height="880" alt="Rag futtatási eredméyn tiny llama új felület 1 " src="https://github.com/user-attachments/assets/598392ac-c6e9-49eb-9eba-459f77837064" />

<img width="1902" height="906" alt="Rag futtatási eredméyn tiny llama új felület 2 " src="https://github.com/user-attachments/assets/621d0664-4a79-40b0-b0c4-a170b97b970c" />

<img width="1882" height="871" alt="Rag futtatási eredméyn tiny llama új felület 3 " src="https://github.com/user-attachments/assets/5b75b880-dbce-4052-8bc0-0ab05ad22b6e" />

---

## Source Transparency

The chatbot displays the exact document chunks used to generate the answer.  
This helps verify correctness and improves trust in the system.

```
Sources used
Chunk 1: ...
Chunk 2: ...
```

This functionality is enabled using:

```
return_source_documents=True
```

---

## Why did I choose TinyLlama?

TinyLlama was chosen because:
- it runs efficiently on CPU
- lower latency compared to larger models
- suitable for local deployment and real-time chat

This makes the system usable on standard consumer hardware.

---

## Prompt Strategy

The model is constrained using a context-only prompt:

```
Answer the question using only the context.
```

This reduces hallucinations and ensures document-grounded answers.

---

## Example Use Cases

- CV analysis
- academic paper Q&A
- internal company document assistant
- offline knowledge base chatbot

---

## Fully Local Execution

This system runs completely offline:
- no API keys
- no external calls
- suitable for private or sensitive documents

---

## Possible Future Improvements

- reranking retrieved chunks
- hybrid search (BM25 + embeddings)
- streaming token responses
- multi-document support

---

## Author

- LinkedIn: www.linkedin.com/in/bence-kupecz-119701305
- GitHub: https://github.com/kupeczbence
