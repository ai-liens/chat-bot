import os
import json
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document

load_dotenv()

# Initialize OpenAI GPT-5 model
llm = ChatOpenAI(
    model="gpt-5",
    temperature=0.7,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

# Load internal knowledge base documents from file
def load_knowledge_base(file_path="knowledge_base.json"):
    """Load documents from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            doc = Document(
                page_content=item['content'],
                metadata={"source": item['source']}
            )
            documents.append(doc)
        
        return documents
    except FileNotFoundError:
        st.error(f"Knowledge base file '{file_path}' not found!")
        return []
    except json.JSONDecodeError:
        st.error(f"Error parsing knowledge base file '{file_path}'!")
        return []

# Load documents from file
internal_documents = load_knowledge_base()

# Create vector store from documents
if internal_documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(internal_documents)
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    st.error("No documents loaded. Please check your knowledge base file.")
    retriever = None

st.set_page_config(page_title="Internal AI Assistant", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #d4fc79, #96e6a1);
            font-family: 'Segoe UI', sans-serif;
        }
        .stChatMessage {
            border-radius: 20px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .user {
            background-color: #d1ecf1;
            text-align: right;
        }
        .bot {
            background-color: #f8d7da;
            text-align: left;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712039.png", width=100)
st.sidebar.title("Internal AI Assistant 🤖")
st.sidebar.markdown("### 🧠 Chat Memory")
memory_enabled = st.sidebar.toggle("Enable Chat Memory", value=True)
if memory_enabled:
    st.sidebar.markdown("Chat memory is enabled. Your conversation history will be saved.")
st.sidebar.markdown("### 📚 RAG Enabled")
st.sidebar.markdown("This assistant uses **Retrieval-Augmented Generation** to answer internal company queries.")
st.sidebar.markdown(f"**Knowledge Base**: {len(internal_documents)} documents loaded")
st.sidebar.markdown("Built using **OpenAI GPT-5** via **LangChain**")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.title("💬 AI Assistant")
st.caption("Ask anything — your AI assistant is here to help!")

if st.session_state.chat_history:
    chat_text = "\n\n".join(
        [f"User: {msg['content']}" if msg["role"] == "user" else f"Assistant: {msg['content']}" for msg in st.session_state.chat_history]
    )

    st.download_button(
        label="💾 Download Chat History",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain",
    )


for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='stChatMessage user'>🧑‍💻: {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stChatMessage bot'>🤖: {msg['content']}</div>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", key="input", placeholder="Ask me anything...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Check if retriever is available
    if retriever is None:
        st.error("Knowledge base not loaded. Cannot process query.")
        st.stop()

    # Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Build system message with RAG context
    system_message = f"""You are an intelligent internal company assistant designed to help employees with their queries. 
You have access to internal company knowledge and policies.

Use the following context from our internal knowledge base to answer the question. 
If the context doesn't contain relevant information, you can still provide a helpful response based on your general knowledge, 
but clearly indicate when you're not using internal documentation.

Context from internal knowledge base:
{context}

Be professional, helpful, and concise in your responses."""

    if memory_enabled:
        # Build conversation history for context
        conversation_context = []
        for msg in st.session_state.chat_history[:-1]:  # Exclude the current user message
            if msg["role"] == "user":
                conversation_context.append(f"User: {msg['content']}")
            else:
                conversation_context.append(f"Assistant: {msg['content']}")
        
        full_context = "\n".join(conversation_context) if conversation_context else ""
        
        # Create prompt with memory
        prompt = f"{system_message}\n\nPrevious conversation:\n{full_context}\n\nCurrent question: {user_input}"
    else:
        prompt = f"{system_message}\n\nQuestion: {user_input}"

    # Get response from LLM
    response = llm.invoke(prompt)
    bot_reply = response.content

    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

    st.rerun()