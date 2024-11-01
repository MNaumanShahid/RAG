import streamlit as st
import requests
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI



data_dir = "data"
persist_directory = 'db'
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=persist_directory)


    file_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".txt")]
    documents = []
    for file_path in file_paths:
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    vectorstore.add_documents(documents)
    vectorstore.persist()
else:
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
llm = OpenAI()
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

st.title("W3B Chatbot")

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'input' not in st.session_state:
    st.session_state.input = ''

def get_chatbot_response(query):
    response = qa_chain.run({"question": query, "chat_history": st.session_state.chat_history})
    return response

def process_input():
    user_input = st.session_state.input
    if user_input:
        # Append user input to conversation
        st.session_state.conversation.append({"role": "user", "text": user_input})

        # Get response from chatbot
        response = get_chatbot_response(user_input)

        # Append chatbot response to conversation
        st.session_state.conversation.append({"role": "bot", "text": response})

        # Update chat history with the latest interaction
        st.session_state.chat_history.append((user_input, response))

        # Clear the input field
        st.session_state.input = ''  # Safe to modify within the callback

# User input with on_change callback
st.text_input("You:", key='input', on_change=process_input)

# Display the conversation
for chat in st.session_state.conversation:
    if chat['role'] == 'user':
        st.write(f"**You:** {chat['text']}")
    else:
        st.write(f"**Bot:** {chat['text']}")