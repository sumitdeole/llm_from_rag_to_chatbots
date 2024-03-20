# %%writefile rag_app.py

from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
import streamlit as st

# llm
@st.cache_resource
def init_llm():
  return LlamaCpp(model_path = "/content/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                  max_tokens = 2000,
                  temperature = 0.1,
                  top_p = 1,
                  n_gpu_layers = -1,
                  n_ctx = 1024)
llm = init_llm()

# embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "/content/embeddings"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)

# load Vector Database
# allow_dangerous_deserialization is needed. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine
vector_db = FAISS.load_local("/content/faiss_index", embeddings, allow_dangerous_deserialization=True)

# retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# memory
@st.cache_resource
def init_memory(_llm):
    return ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
memory = init_memory(llm)

# prompt
template = """
<s> [INST]
You are polite and professional question-answering AI assistant. You must provide a helpful response to the user.

In your response, PLEASE ALWAYS:
  (0) Be a detail-oriented reader: read the question and context and understand both before answering
  (1) Start your answer with a friendly tone, and reiterate the question so the user is sure you understood it
  (2) If the context enables you to answer the question, write a detailed, helpful, and easily understandable answer. If you can't find the answer, respond with an explanation, starting with: "I couldn't find the answer in the information I have access to".
  (3) Ensure your answer answers the question, is helpful, professional, and formatted to be easily readable.
[/INST]
[INST]
Answer the following question using the context provided.
The question is surrounded by the tags <q> </q>.
The context is surrounded by the tags <c> </c>.
<q>
{question}
</q>
<c>
{context}
</c>
[/INST]
</s>
[INST]
Helpful Answer:
[INST]
"""

prompt = PromptTemplate(template=template,
                        input_variables=["context", "question"])

# chain
chain = ConversationalRetrievalChain.from_llm(llm,
                                              retriever=retriever,
                                              memory=memory,
                                              return_source_documents=True,
                                              combine_docs_chain_kwargs={"prompt": prompt})


##### streamlit #####

st.title("Chatier & chatier: conversations in Wonderland")

# Initialise chat history
# Chat history saves the previous messages to be displayed
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Curious minds wanted!"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Going down the rabbithole for answers..."):

        # send question to chain to get answer
        answer = chain(prompt)

        # extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer["answer"])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})