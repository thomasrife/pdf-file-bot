import streamlit as st
from streamlit_chat import message
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import tempfile

temp_dir = tempfile.TemporaryDirectory()

# Get the path of the temporary directory
temp_path = temp_dir.name

#Creating the chatbot interface
st.title("File-Bot 🤖📚")

st.markdown("With our AI-powered PDF reader app, you can now chat with your documents in real-time. Simply load your PDF file and engage in natural language conversation to get instant answers to your questions. \n\n ")

def get_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

openai_key = get_api_key()

def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=.7, openai_api_key=openai_api_key)
    return llm

def qa(query, file):
    # load document
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=load_LLM(openai_api_key=openai_key), chain_type="stuff", retriever=retriever, return_source_documents=False)
    result = qa({"query": query})
    return result['result']

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

uploaded_file = st.file_uploader("Upload your PDF file: ", type=["pdf"])

file_path = None

# Check if a file was uploaded
if uploaded_file is not None:
    # Get the filename
    filename = uploaded_file.name

    # Create a temporary file in the temporary directory
    with open(f"{temp_path}/{filename}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Store the path of the temporary file in a variable
    file_path = f"{temp_path}/{filename}"

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("Your question: ", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = qa(user_input, file_path)
    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

st.markdown("This tool is powered by [LangChain](https://langchain.com/) and [OpenAI](https://openai.com) and made by \
                [@elthomate](https://twitter.com/elthomate). \n\n View Source Code on [Github](https://github.com/thomasrife/pdf-file-bot/blob/main/main.py).")


temp_dir.cleanup()
