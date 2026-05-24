from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)

loader = TextLoader("documents/notes.txt")
documents = loader.load()

response = model.invoke("What is the capital of France?")

print(response.content) 