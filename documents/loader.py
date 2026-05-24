from langchain_community.document_loaders import TextLoader

data=TextLoader(r"D:\Resume\RAG project\documents\notes.txt").load()
print(data)

