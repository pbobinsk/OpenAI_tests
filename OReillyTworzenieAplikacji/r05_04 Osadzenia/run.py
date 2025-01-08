from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("Minecraft.pdf")
pages = loader.load_and_split()

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain.vectorstores import FAISS
db = FAISS.from_documents(pages, embeddings)

q = "Co jest potrzebne do zbudowania drzwi?"
response = db.similarity_search(q)[0]
print(response)

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
llm = OpenAI()
chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever())

q = "Co jest potrzebne do zbudowania drzwi?"
response = chain(q, return_only_outputs=True)
print(response)