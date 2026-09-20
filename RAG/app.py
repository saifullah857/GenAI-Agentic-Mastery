import streamlit as st
import os
import nbimporter  # 👈 IMPORTANT FIX for .ipynb import

from code import (
    EmbedingManager,
    VectorStoreManager,
    RAGRetriever,
    generate_output
)

from langchain_groq import ChatGroq

st.set_page_config(page_title="RAG Chatbot", page_icon="📚")

st.title("📚 RAG Chatbot")

# ----------------------------
# ENV SETUP
# ----------------------------
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="qwen/qwen3-32b",
    temperature=0.1
)

# ----------------------------
# INIT RAG COMPONENTS
# ----------------------------
embedding_manager = EmbedingManager()
vector_store = VectorStoreManager()
rag_retriever = RAGRetriever(embedding_manager, vector_store)

# ----------------------------
# SESSION STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# USER INPUT
# ----------------------------
query = st.chat_input("Ask anything...")

if query:
    # store user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            try:
                answer = generate_output(query, rag_retriever, llm)
            except Exception as e:
                answer = f"❌ Error: {str(e)}"

            st.markdown(answer)

    # store assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})