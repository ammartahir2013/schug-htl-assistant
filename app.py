import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

st.title("SCHUG HTL Assistent")
st.markdown("Hinweis: Dieses System ersetzt keine rechtliche Beratung.")

openai_api_key = st.secrets["OPENAI_API_KEY"]

# PDF laden
loader = PyPDFLoader("schug.pdf")
documents = loader.load()

# Text aufteilen
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# Embeddings + Vektor DB
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0
)

query = st.text_input("Frage zum Schulunterrichtsgesetz")

if query:
    relevant_docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])

    prompt = f"""
Beantworte die folgende Frage ausschließlich auf Basis des bereitgestellten Kontextes.
Keine Spekulation. Wenn nicht enthalten, schreibe: "Im Dokument nicht gefunden."
Zitiere Paragraph (§) und Absatz wörtlich.

Kontext:
{context}

Frage:
{query}
"""

    response = llm.invoke(prompt)

    st.write("### Antwort")
    st.write(response.content)

    st.write("### Quelle")
    for doc in relevant_docs[:1]:
        st.write(doc.page_content)
