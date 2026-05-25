from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("D:\\Resume\\RAG project\\documents\\N-Gram_Language_Models.pdf").load()

print(data[0].page_content)