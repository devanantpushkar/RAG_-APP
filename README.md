# Retrieval-Augmented Generation (RAG) App using Hugging Face + Gemma:2b

An intelligent RAG system that combines Hugging Face sentence embeddings with Groq’s blazing-fast `gemma:2b` LLM to answer user questions based on uploaded documents. Built for speed, simplicity, and high accuracy using Python, FAISS, and Streamlit.

---

## 📑 Table of Contents
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

## 🚀 Features
- Upload PDFs or plain text documents  
- Automatic chunking, embedding, and vector storage  
- Fast semantic retrieval using FAISS  
- Accurate answer generation via `gemma:2b` from Groq  
- Real-time Q&A interface with document-grounded responses  
- Optional caching and logging (via Redis/FastAPI backend)  

---

## 📂 Dataset
- **Type:** Dynamic (uploaded by user)  
- **Formats Supported:** `.pdf`, `.txt`  
- **Chunking Strategy:** Overlapping windows (~300–500 tokens)  
- **Embeddings:** `all-MiniLM-L6-v2` (`sentence-transformers`)  
- **Storage:** FAISS in-memory vector store  

---

## 🔍 Approach

### 1. Document Chunking
- Input files split into token-based chunks using a sliding window  
- Maintains semantic context across document sections  

### 2. Embedding
- Each chunk vectorized using Hugging Face `all-MiniLM-L6-v2`  
- Stored in FAISS index for fast approximate nearest neighbor search  

### 3. Retrieval
- User query embedded, then top-k similar chunks retrieved via cosine similarity  

### 4. Generation
- Retrieved context combined with query into a prompt  
- Sent to `gemma:2b` model via Groq API for response  

---

## Query Flow

```text
[User Query] → [Embed Query] → [Search FAISS Index] 
→ [Select Top-k Chunks] → [Format Prompt] → [LLM (Gemma)] 
→ [Return Answer + Source Context]

 ## Model Architecture

# Document Embedding Pipeline
chunks = split_document(doc)
embeddings = embed_model.encode(chunks)
faiss_index.add(embeddings)

# Retrieval + Generation
query_vec = embed_model.encode([query])
top_k_chunks = faiss_index.search(query_vec)

prompt = f"Context:\n{top_k_chunks}\n\nQuestion:\n{query}\n\nAnswer:"
response = groq.chat(model="gemma-2b", prompt=prompt)
