from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

load_dotenv()

# Create documents
docs = [
    Document(
        page_content="What is RAG?",
        metadata={"source": "notes.txt"}
    ),
    Document(
        page_content=(
            "RAG stands for Retrieval Augmented Generation. "
            "It is a technique that combines retrieval-based "
            "methods with generative models to improve the "
            "quality and relevance of generated content."
        ),
        metadata={"source": "notes.txt"}
    )
]

# Mistral Embeddings Model
embeddings_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# Create Chroma Vector Store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    collection_name="rag_notes"
)

# Test retrieval
retriever = vectorstore.as_retriever()

results = retriever.invoke("What is RAG?")

for doc in results:
    print(doc.page_content)