import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="Research Paper Summarizer", layout="wide")
st.title("📄 Research Paper Summarizer + Q&A Assistant")

file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

# ----------------------------- Handle PDF -----------------------------
if file:
    with st.spinner("Reading PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        full_text = "\n".join([doc.page_content for doc in documents])
        st.subheader("📖 Extracted Content Preview")
        st.text_area("Text (First 3000 chars)", full_text[:3000] + "..." if len(full_text) > 3000 else full_text, height=250)

    # ----------------------------- Text Split -----------------------------
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents([Document(page_content=full_text)])

    # ----------------------------- Summarization -----------------------------
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",truncation=True)
    llm_sum = HuggingFacePipeline(pipeline=summarizer)

    if st.button("🔍 Summarize Paper"):
        with st.spinner("Summarizing... Please wait..."):
            
            # Remove empty chunks (important)
            split_docs = [doc for doc in split_docs if doc.page_content.strip()]
            
            if not split_docs:
                st.error("❌ No content to summarize. Please upload a valid PDF.")
            else:
                sum_chain = load_summarize_chain(llm_sum, chain_type="map_reduce")
                summary = sum_chain.invoke({"input_documents": split_docs})
                st.subheader("📝 Summary")
                st.success(summary)

    # ----------------------------- RAG Q&A -----------------------------
    st.markdown("## ❓ Ask Questions About the Paper")
    user_question = st.text_input("Type your question here:")

    if user_question:
        with st.spinner("Finding answer..."):
            # Create or load vector DB
            persist_dir = "chroma_db"
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=persist_dir)

            # Create retriever and chain
            retriever = vectordb.as_retriever()
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm_sum,
                retriever=retriever,
                chain_type="stuff"
            )

            response = rag_chain.run(user_question)
            st.subheader("📌 Answer")
            st.write(response)
