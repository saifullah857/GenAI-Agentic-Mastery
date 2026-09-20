import streamlit as st
import os
import import_ipynb

from langchain_groq import ChatGroq
from RAG.code import (
    EmbedingManager,
    VectorStoreManager,
    RAGRetriever,
    generate_output
)

st.title("📚 RAG Chatbot")

embedding_manager = EmbedingManager()

vector_store = VectorStoreManager()

rag_retriever = RAGRetriever(
    embedding_manager,
    vector_store
)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3-32b",
    temperature=0.1
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask anything...")

if query:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = generate_output(
                query,
                rag_retriever,
                llm
            )

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )