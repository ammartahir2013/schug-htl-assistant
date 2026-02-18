import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.title("SCHUG HTL Assistent")

st.markdown("Hinweis: Dieses System ersetzt keine rechtliche Beratung.")

openai_api_key = st.secrets["OPENAI_API_KEY"]

loader = PyPDFLoader("schug.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

query = st.text_input("Frage zum Schulunterrichtsgesetz")

if query:
    result = qa(query)
    st.write("### Antwort")
    st.write(result["result"])

    st.write("### Quelle")
    for doc in result["source_documents"][:1]:
        st.write(doc.page_content)
