from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)

loader = TextLoader("documents/notes.txt")
documents = loader.load()

template =ChatPromptTemplate.from_messages([
    ("system", "You are a summary helpful assistant."),
    ("human","{documents}")
])

prompt = template.format_messages(documents=documents[0].page_content)  

response = model.invoke(prompt)

print(response.content) 