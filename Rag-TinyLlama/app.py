#ALAP VERZÓ KÓD:
# import streamlit as st
# from src.chatbot import load_qa_chain

# st.title("RAG PDF Chatbot")

# st.write("Ask questions about your PDF document.")

# @st.cache_resource
# def get_chain():
#     return load_qa_chain()

# qa = get_chain()

# user_question = st.text_input("Your question:")

# if user_question:
#     with st.spinner("Thinking..."):
#         result = qa.invoke({"query": user_question})
#         st.write(result["result"])

#ÚJ VERZIÓ KÓD:
import streamlit as st
import time
from src.chatbot import load_qa_chain

# Oldal beállítások 
# böngésző fül neve, ikon és széles layout
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG PDF Chatbot")
st.write("Ask questions about your PDF document.")

# Sidebar 
# Ide kerülnek a beállítások és információk
with st.sidebar:
    st.header("⚙️ Settings")

    # Chat törlés gomb
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.header("📄 Document info")
    st.write("Embedding model: all-MiniLM-L6-v2")
    st.write("LLM: TinyLlama")

# RAG lánc cache-elése 
# megakadályozza, hogy minden kérdésnél újratöltse a modellt
@st.cache_resource
def get_chain():
    return load_qa_chain()

qa = get_chain()

# Session state a chat előzményekhez 
# Ez tárolja a beszélgetést újratöltés után is
if "messages" not in st.session_state:
    st.session_state.messages = []

# Korábbi üzenetek megjelenítése 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#  Chat input mező 
user_input = st.chat_input("Type your question here...")

#  Ha a user küld üzenetet 
if user_input:

    # User üzenet mentése a session-be
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # User üzenet megjelenítése
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant válasz generálása
    with st.chat_message("assistant"):

        # Spinner animáció amíg az LLM dolgozik
        with st.spinner("🔍 Searching document and generating answer..."):

            start = time.time()

            # RAG lekérdezés
            result = qa.invoke({"query": user_input})

            end = time.time()
            latency = end - start

            # A modell válasza
            answer = result["result"]

            st.markdown(answer)

            # Válaszidő megjelenítése
            st.caption(f"⏱️ Response time: {latency:.2f} seconds")

            #  Forrás dokumentumok megjelenítése 
            # Ehhez a chatbot.py-ban return_source_documents=True kell
            if "source_documents" in result:
                with st.expander("📚 Sources used"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(doc.page_content)
                        st.markdown("---")

    # Assistant válasz mentése a chat előzményekbe
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )