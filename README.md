# Retrieval-Augmented Generation (RAG) App using Hugging Face + Gemma:2b

An intelligent RAG system that combines Hugging Face sentence embeddings with Groq’s blazing-fast `gemma:2b` LLM to answer user questions based on uploaded documents. Built for speed, simplicity, and high accuracy using Python and Streamlit.

---

## Table of Contents

1. Features  
2. Dataset  
3. Approach  
4. Query Flow  
5. Model Architecture  
6. Accuracy  
7. Technologies Used  
8. How to Run  
9. Future Improvements  

---

## Features

1. Upload PDF or text files directly from the web interface  
2. Automatic document chunking and vector embedding using Hugging Face  
3. FAISS-powered fast semantic search  
4. Integration with `gemma:2b` model for accurate question answering  
5. Real-time Q&A interface with source chunk references  
6. Lightweight, fast, and deployable locally or in the cloud  

---

## Dataset

- Type: Custom documents (uploaded by user)  
- Formats Supported: `.pdf`, `.txt`  
- Chunk Size: ~300-500 tokens per chunk  
- Embeddings: `all-MiniLM-L6-v2` from Hugging Face  
- No fixed training dataset; works on dynamic user uploads  

---

## Approach

### Document Chunking:
- Documents are split into overlapping chunks (stride=100)  
- Ensures semantic continuity for long paragraphs  

### Embedding:
- Each chunk is embedded using Hugging Face MiniLM (`sentence-transformers`)  
- Embeddings are stored in an in-memory FAISS index  

### Retrieval:
- When a query is asked, top-k (usually 3–5) most relevant chunks are retrieved  
- Uses cosine similarity from FAISS  

### Answer Generation:
- Query + Retrieved Chunks → Prompt for Groq’s `gemma:2b` model  
- Generated answer is returned with relevant sources  

---

## Query Flow

1. **User Input:** Query entered via Streamlit  
2. **Retrieve:** Find top relevant chunks using vector similarity  
3. **Generate:** Combine query + context → send to `gemma:2b`  
4. **Display:** Show final answer along with context chunks  

---

## Model Architecture

```python
# Embedding Pipeline
document_chunks = split_text(doc)
embeddings = model.encode(document_chunks)
faiss_index.add(embeddings)

# Query Processing
query_embedding = model.encode(query)
top_k_chunks = faiss_index.search(query_embedding)

# Prompt Creation
prompt = f"Context: {top_k_chunks}\n\nQuestion: {query}\n\nAnswer:"

# LLM Call
response = groq_client.chat(prompt=prompt, model="gemma-2b")
