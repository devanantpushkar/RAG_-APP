import streamlit as st
import os
os.environ["USER_AGENT"] = "dev-anant-rag-app/0.1"
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["USER_AGENT"] = os.getenv("USER_AGENT")


# Set Streamlit title
st.title("🧠 RAG App with HuggingFace + gemma:2b (Groq)")

# Initialize stateful variables only once
if "vectors" not in st.session_state:
    # Load and split documents
    st.session_state.loader = PyPDFLoader(r"C:\Users\DELL\Desktop\genai\Groq\Ch 5 System Modelling.pdf")
   
    st.session_state.docs = st.session_state.loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(st.session_state.docs[:50])

    # HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS vector store
    st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)

# Create LLM from Groq
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"

)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the following context only.
    If the answer isn't in the context, say you don't know.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Document chain and retrieval
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Input prompt
user_input = st.text_input("🔍 Ask a question")

if user_input:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_input})
    st.write("⏱️ Response time:", round(time.process_time() - start, 2), "sec")
    st.write("🤖", response["answer"])

    # Show context
    with st.expander("📄 Documents Used"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("------")
