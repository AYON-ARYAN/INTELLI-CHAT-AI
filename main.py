import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import requests
import logging
import shutil

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
DOCS_PATH = "docs"
VECTOR_DB_PATH = "db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
WEB_SEARCH_RESULTS_COUNT = 3
OLLAMA_MODEL_NAME = "mistral"
OLLAMA_TIMEOUT = 10  # Seconds for Ollama connection check

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# --- Ensure Directories Exist ---
os.makedirs(DOCS_PATH, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="IntelliChat AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a minimal and clean look ---
st.markdown("""
    <style>
    /* General app styling */
    .stApp {
        background-color: #f7f9fc; /* Very light background */
        color: #333333;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    .stApp > header {
        background-color: transparent;
    }

    /* Main title */
    h1 {
        color: #2c3e50; /* Darker blue-gray for titles */
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 0.5em;
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    /* Subtitles and sections */
    h2 {
        color: #34495e;
        font-size: 1.6em;
        margin-top: 1.8em;
        margin-bottom: 0.7em;
        border-bottom: 1px solid #eceff1; /* Subtle separator */
        padding-bottom: 5px;
    }
    h3 {
        color: #555555;
        font-size: 1.2em;
        margin-top: 1.2em;
        margin-bottom: 0.5em;
    }

    /* Streamlit specific component styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px; /* Reduced gap for more compact look */
        justify-content: center;
        margin-bottom: 1.5em;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: nowrap;
        background-color: #e9ecef; /* Lighter gray for inactive tabs */
        border-radius: 8px; /* Fully rounded corners for tabs */
        gap: 8px;
        padding: 10px 18px;
        transition: all 0.2s ease-in-out;
        font-weight: 500;
        color: #666666;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff; /* White background for active tab */
        border: 1px solid #4CAF50; /* Green border for active tab */
        color: #2c3e50;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* Soft shadow for active tab */
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #dee2e6;
    }

    /* Buttons */
    .stButton > button {
        background-color: #4CAF50; /* Primary green */
        color: white;
        padding: 8px 18px;
        border-radius: 6px;
        border: none;
        cursor: pointer;
        font-weight: 500;
        transition: background-color 0.2s ease, transform 0.1s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #45a049; /* Darker green on hover */
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Secondary buttons (e.g., Clear All Data) */
    .stButton[data-testid="stFormSubmitButton"] > button {
        background-color: #007bff; /* Blue for submit */
    }
    .stButton[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #0056b3;
    }

    .stButton[data-testid^="stButton-secondary"] > button {
        background-color: #dc3545; /* Red for destructive actions */
        color: white;
    }
    .stButton[data-testid^="stButton-secondary"] > button:hover {
        background-color: #c82333;
    }

    /* Sidebar */
    .stSidebar {
        background-color: #ffffff; /* White sidebar background */
        padding: 20px 25px;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 8px rgba(0,0,0,0.05); /* Subtle shadow on sidebar edge */
    }
    .stSidebar h2 {
        border-bottom: none;
    }
    .stSidebar .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 15px;
        padding: 10px;
    }
    .stSidebar .stExpander > div > div {
        padding: 0; /* Remove internal padding of expander */
    }

    /* File Uploader */
    .stFileUploader label span {
        background-color: #007bff; /* Blue for file uploader */
        color: white;
        padding: 8px 15px;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    .stFileUploader label span:hover {
        background-color: #0056b3;
    }

    /* Text Inputs and Text Areas */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 1px solid #ced4da;
        padding: 10px 12px;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.075);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #80bdff;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
        outline: none;
    }

    /* Chat Messages */
    .chat-message-container {
        display: flex;
        margin-bottom: 15px;
        gap: 12px;
        align-items: flex-start;
    }
    .chat-icon {
        font-size: 20px; /* Slightly smaller emojis */
        line-height: 1; /* Align vertically */
        padding-top: 5px;
    }
    .user-message {
        background-color: #e6f7ff; /* Very light blue */
        border-radius: 16px 16px 4px 16px; /* Softer, asymmetrical corners */
        padding: 12px 18px;
        max-width: 75%;
        margin-left: auto;
        text-align: right;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08); /* Softer shadow */
        word-break: break-word; /* Ensure long words wrap */
    }
    .bot-message {
        background-color: #e0f2f1; /* Very light green */
        border-radius: 16px 16px 16px 4px; /* Softer, asymmetrical corners */
        padding: 12px 18px;
        max-width: 75%;
        text-align: left;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08); /* Softer shadow */
        word-break: break-word; /* Ensure long words wrap */
    }

    /* Info/Warning/Success messages */
    .stAlert {
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 0.95em;
    }
    .stAlert.success {
        background-color: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
    }
    .stAlert.info {
        background-color: #d1ecf1;
        color: #0c5460;
        border-color: #bee5eb;
    }
    .stAlert.warning {
        background-color: #fff3cd;
        color: #856404;
        border-color: #ffeeba;
    }
    .stAlert.error {
        background-color: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
    }

    /* Horizontal Rule */
    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 2em 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("IntelliChat AI")
st.markdown("Your smart assistant for documents & web. Ask anything!")

# --- Session State Initialization ---
def initialize_session_state():
    defaults = {
        "doc_chat_history": [],
        "web_chat_history": [],
        "qa_chain": None,
        "retriever": None,
        "llm_general_chat": None,
        "llm_init_attempted": False,
        "retriever_ready": False,
        "ollama_status_checked": False
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# --- Document Processing and Embedding ---
@st.cache_resource(show_spinner="Processing documents and creating embeddings...")
def process_documents_and_create_retriever():
    logger.info("Attempting to process documents...")
    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logger.info("No PDF documents found in the 'docs' directory for processing.")
        st.session_state.retriever_ready = False
        return None

    try:
        loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        if not docs:
            logger.warning("No content extracted from PDF documents.")
            st.session_state.retriever_ready = False
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        logger.info(f"Split {len(docs)} documents into {len(chunks)} chunks.")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        vectorstore.persist()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        logger.info("Documents processed, embeddings created, and vectorstore persisted.")
        st.session_state.retriever_ready = True
        return retriever
    except Exception as e:
        logger.error(f"Error during document processing: {e}", exc_info=True)
        st.error(f"Failed to process documents. Error: {e}. Check console for details.")
        st.session_state.retriever_ready = False
        return None

# --- LLM and Chain Initialization ---
@st.cache_resource(show_spinner=f"Connecting to Ollama ({OLLAMA_MODEL_NAME}) and preparing chatbot...")
def initialize_llm():
    llm = None
    try:
        llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.7)
        test_response = llm.invoke("Are you ready?", config={'timeout': OLLAMA_TIMEOUT})
        logger.info(f"Ollama ({OLLAMA_MODEL_NAME}) test successful. Response: {test_response.content[:50]}...")
        st.session_state.ollama_status_checked = True
        return llm
    except requests.exceptions.ConnectionError:
        st.error(
            f"**Ollama Server Error:** Cannot connect to Ollama. Please ensure `ollama serve` is running in your terminal.")
        logger.error("Connection to Ollama server failed.", exc_info=True)
        st.session_state.ollama_status_checked = True
        return None
    except Exception as e:
        st.error(
            f"**Ollama Model Error:** Failed to initialize LLM ({OLLAMA_MODEL_NAME}). Ensure it's downloaded (`ollama run {OLLAMA_MODEL_NAME}`). Error: {e}")
        logger.error(f"LLM initialization error: {e}", exc_info=True)
        st.session_state.ollama_status_checked = True
        return None

@st.cache_resource(show_spinner="Building document QA chain...")
def create_qa_chain(_llm_instance, _retriever_instance):
    if _llm_instance is None or _retriever_instance is None:
        logger.warning("Cannot create QA chain: LLM or Retriever not ready.")
        return None
    try:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=_llm_instance,
            retriever=_retriever_instance,
            memory=memory,
            chain_type="stuff",
        )
        logger.info("Conversational Retrieval Chain built successfully.")
        return qa_chain
    except Exception as e:
        st.error(f"Failed to build conversational chain. Check retriever status. Error: {e}")
        logger.error(f"Chain creation error: {e}", exc_info=True)
        return None

# --- Web Search Function ---
def perform_web_search(query: str, num_results: int = WEB_SEARCH_RESULTS_COUNT) -> str:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        st.warning(
            "Google API credentials (GOOGLE_API_KEY, GOOGLE_CSE_ID) are missing or invalid in your `.env` file. Web search will not work.")
        logger.warning("Google API credentials missing for web search.")
        return "Google API credentials are not configured. Web search is disabled."

    try:
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": num_results
        }
        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        search_results = []
        if "items" in data:
            for item in data["items"]:
                snippet = item.get('snippet')
                link = item.get('link')
                if snippet and link:
                    search_results.append(f"Snippet: {snippet}\nLink: {link}")

        logger.info(f"Web search for '{query}' retrieved {len(search_results)} results.")
        return "\n\n".join(search_results) if search_results else "No relevant web results found."

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"Web search HTTP error: {http_err} - Response: {response.text}", exc_info=True)
        return f"Web search failed: API responded with an error. ({http_err.response.status_code})"
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Web search connection error: {conn_err}", exc_info=True)
        return "Web search failed: Could not connect to Google API. Check your internet connection."
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Web search timeout error: {timeout_err}", exc_info=True)
        return "Web search failed: Request timed out."
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Web search general request error: {req_err}", exc_info=True)
        return f"Web search failed due to a request error: {req_err}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during web search: {e}", exc_info=True)
        return f"An unexpected error occurred during web search: {e}"

# --- Sidebar UI and Actions ---
with st.sidebar:
    st.header("Configuration & Management")

    with st.expander("Upload Documents", expanded=True):
        st.markdown("Upload your PDF files here. New uploads will re-index your documents.")
        uploaded_files = st.file_uploader(
            "Select PDFs",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader_sidebar",
            help="Accepted formats: PDF. Max file size: 200MB."
        )
        if uploaded_files:
            new_files_added = False
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DOCS_PATH, uploaded_file.name)
                if not os.path.exists(file_path):
                    try:
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.success(f"Added: **{uploaded_file.name}**")
                        new_files_added = True
                        logger.info(f"Successfully saved {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error saving {uploaded_file.name}: {e}")
                        logger.error(f"Error saving uploaded file {uploaded_file.name}", exc_info=True)
                else:
                    st.info(f"Skipped: **{uploaded_file.name}** (already exists)")

            if new_files_added:
                st.session_state.qa_chain = None
                st.session_state.retriever = None
                st.session_state.retriever_ready = False
                st.cache_resource.clear()
                logger.info("New files uploaded. Clearing cache and preparing for re-indexing.")
                st.info("New PDFs detected! Re-indexing documents and re-initializing the chatbot. Please wait...")
                st.rerun()

    st.subheader("Document Status")
    current_pdfs = [f for f in os.listdir(DOCS_PATH) if f.lower().endswith(".pdf")]
    if current_pdfs:
        st.markdown(f"**Found {len(current_pdfs)} PDF(s) in `docs/`:**")
        for pdf_name in current_pdfs[:5]:
            st.markdown(f"- {pdf_name}")
        if len(current_pdfs) > 5:
            st.markdown(f"- ... and {len(current_pdfs) - 5} more.")
    else:
        st.info("No PDFs uploaded yet. Use the uploader above to get started.")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", help="Clears all chat messages in both tabs"):
            st.session_state.doc_chat_history = []
            st.session_state.web_chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
    with col2:
        if st.button("Clear All Data", help="Deletes all uploaded PDFs and the vector database. Requires app restart.", type="secondary"):
            if os.path.exists(DOCS_PATH):
                try:
                    shutil.rmtree(DOCS_PATH)
                    os.makedirs(DOCS_PATH, exist_ok=True)
                    st.warning(f"Deleted all PDFs in `{DOCS_PATH}/`.")
                    logger.info(f"Deleted contents of {DOCS_PATH}")
                except Exception as e:
                    st.error(f"Could not delete {DOCS_PATH}: {e}")
                    logger.error(f"Error deleting {DOCS_PATH}", exc_info=True)
            if os.path.exists(VECTOR_DB_PATH):
                try:
                    shutil.rmtree(VECTOR_DB_PATH)
                    st.warning(f"Deleted vector database in `{VECTOR_DB_PATH}/`.")
                    logger.info(f"Deleted contents of {VECTOR_DB_PATH}")
                except Exception as e:
                    st.error(f"Could not delete {VECTOR_DB_PATH}: {e}")
                    logger.error(f"Error deleting {VECTOR_DB_PATH}", exc_info=True)

            st.session_state.clear()
            st.cache_resource.clear()
            initialize_session_state()
            st.success("All data cleared! Please refresh the Streamlit app for a clean start.")
            st.stop()

# --- Main Logic Flow for Processing and Initialization Status ---
st.markdown("---")
st.subheader("Chatbot Status")

status_placeholder = st.empty()

if not st.session_state.llm_init_attempted:
    with status_placeholder:
        st.info(f"Connecting to Ollama ({OLLAMA_MODEL_NAME}) and preparing general chatbot...")
    llm_instance = initialize_llm()
    if llm_instance:
        st.session_state.llm_general_chat = llm_instance
        with status_placeholder:
            st.success("General chatbot is **active and ready!**")
    else:
        with status_placeholder:
            st.error("General chatbot initialization **failed**. Check Ollama server and model download.")
    st.session_state.llm_init_attempted = True

elif not st.session_state.llm_general_chat:
    with status_placeholder:
        st.error("General chatbot initialization **failed** in a previous attempt. Check Ollama server and model, then refresh.")
elif st.session_state.llm_general_chat:
    with status_placeholder:
        st.success("General chatbot is **active and ready!**")

if not st.session_state.retriever_ready and current_pdfs:
    with status_placeholder:
        st.info("Indexing documents. This may take a moment...")
    retriever_instance = process_documents_and_create_retriever()
    if retriever_instance:
        st.session_state.retriever = retriever_instance
        st.session_state.retriever_ready = True
        with status_placeholder:
            st.success("Document retriever is **ready!**")
    else:
        with status_placeholder:
            st.error("Document indexing **failed**. See errors above or in terminal.")
        st.session_state.retriever_ready = False
        st.session_state.retriever = None
        st.session_state.qa_chain = None

elif not current_pdfs:
    with status_placeholder:
        st.info("Upload PDFs to enable document chat. No documents found for indexing.")
    st.session_state.retriever_ready = False
    st.session_state.retriever = None

elif st.session_state.retriever_ready:
    with status_placeholder:
        st.success("Document retriever is **ready** for use.")

if st.session_state.llm_general_chat and st.session_state.retriever_ready and not st.session_state.qa_chain:
    with status_placeholder:
        st.info("Building document QA chain...")
    qa_chain_instance = create_qa_chain(st.session_state.llm_general_chat, st.session_state.retriever)
    if qa_chain_instance:
        st.session_state.qa_chain = qa_chain_instance
        with status_placeholder:
            st.success("Document chat is **fully operational!**")
    else:
        with status_placeholder:
            st.error("Document chat initialization **failed**. Check LLM and Retriever status.")
        st.session_state.qa_chain = None

# --- Chat Interface Tabs ---
st.markdown("---")
tabs = st.tabs(["Document Chat", "Web & General Chat"])

# --- Document Chat Tab ---
with tabs[0]:
    st.subheader("Ask questions about your uploaded PDFs")
    if not st.session_state.qa_chain:
        st.warning(
            "Document chat is not active. Please upload PDFs and ensure your Ollama server is running (with 'mistral' model). Check **Chatbot Status** above.")
    else:
        with st.form("doc_chat_form", clear_on_submit=True):
            user_query = st.text_area(
                "Your question:",
                placeholder="E.g., What is this document about? Summarize the main points.",
                key="doc_query_input",
                height=80
            )
            submit_button = st.form_submit_button("Ask Document Chat")

            if submit_button and user_query:
                with st.spinner("IntelliChat is thinking..."):
                    try:
                        response = st.session_state.qa_chain({"question": user_query})
                        answer = response.get("answer", "No answer found.")
                        st.session_state.doc_chat_history.append({"question": user_query, "answer": answer})
                        logger.info(f"Doc chat query processed: '{user_query}'")
                    except Exception as e:
                        st.error(f"An error occurred during document chat: {e}. Please check the console.")
                        logger.error(f"Error in document chat: {e}", exc_info=True)
                        st.session_state.doc_chat_history.append(
                            {"question": user_query, "answer": "Error: Failed to get an answer."})

        st.markdown("---")
        st.markdown("### Conversation History")
        if not st.session_state.doc_chat_history:
            st.markdown("_No document chat history yet. Ask a question above!_")
        else:
            for chat in reversed(st.session_state.doc_chat_history):
                st.markdown(f"""
                <div class="chat-message-container">
                    <div class="chat-icon">🤔</div>
                    <div class="user-message">{chat['question']}</div>
                </div>
                <div class="chat-message-container">
                    <div class="chat-icon">✨</div>
                    <div class="bot-message">{chat['answer']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("---")

# --- Web & General Chat Tab ---
with tabs[1]:
    st.subheader("General questions or direct web search")
    st.info("💡 Type `web search: your query` for a web search; otherwise, it's a general chat with the LLM.")

    if not st.session_state.llm_general_chat:
        st.warning("General chat is not active. Please ensure your Ollama server is running (with 'mistral' model). Check **Chatbot Status** above.")
    else:
        with st.form("web_general_chat_form", clear_on_submit=True):
            user_query_web = st.text_area(
                "Your prompt:",
                placeholder="E.g., What is the capital of France? or web search: latest news on AI",
                key="web_query_input",
                height=80
            )
            submit_button_web = st.form_submit_button("Send Query")

            if submit_button_web and user_query_web:
                with st.spinner("IntelliChat is thinking..."):
                    answer_web = ""
                    if user_query_web.lower().startswith("web search:"):
                        search_query = user_query_web.split("web search:", 1)[-1].strip()
                        answer_web = perform_web_search(search_query)
                        logger.info(f"Web search performed for: '{search_query}'")
                    else:
                        try:
                            llm_response = st.session_state.llm_general_chat.invoke(user_query_web)
                            answer_web = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                            logger.info(f"General chat query processed: '{user_query_web}'")
                        except Exception as e:
                            st.error(f"An error occurred during general chat: {e}. Please check the console.")
                            logger.error(f"Error in general chat: {e}", exc_info=True)
                            answer_web = "Error: LLM failed to generate a response."

                st.session_state.web_chat_history.append({"question": user_query_web, "answer": answer_web})

        st.markdown("---")
        st.markdown("### Conversation History")
        if not st.session_state.web_chat_history:
            st.markdown("_No web/general chat history yet. Ask a question above!_")
        else:
            for chat in reversed(st.session_state.web_chat_history):
                st.markdown(f"""
                <div class="chat-message-container">
                    <div class="chat-icon">🤔</div>
                    <div class="user-message">{chat['question']}</div>
                </div>
                <div class="chat-message-container">
                    <div class="chat-icon">✨</div>
                    <div class="bot-message">{chat['answer']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("---")