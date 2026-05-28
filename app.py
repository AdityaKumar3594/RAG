import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

DB_DIR = "chroma_db"

# -----------------------------
# Create Vector Database
# -----------------------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        with st.spinner("Processing PDF..."):

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001"
            )

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=DB_DIR
            )

        os.remove(file_path)

        st.success("Vector Database Created!")

# -----------------------------
# Load Existing Vector DB
# -----------------------------

if os.path.exists(DB_DIR):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If answer is not available in context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}
"""
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    st.divider()
    st.subheader("Ask Questions")

    query = st.text_input("Enter your question")

    if query:

        with st.spinner("Searching document..."):

            docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            response = chain.invoke({
                "context": context,
                "question": query
            })

        st.write("### AI Answer")
        st.write(response)

        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(docs, start=1):
                st.write(f"### Chunk {i}")
                st.write(doc.page_content)
                st.divider()
