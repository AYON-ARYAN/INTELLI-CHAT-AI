IntelliChat AI: Your Smart Document & Web Assistant


IntelliChat AI is a versatile chatbot built with LangChain, Streamlit, and Ollama. It helps you quickly get answers from your documents and the web, all through a clean, intuitive interface.

Key Features
Document Q&amp;A: Upload PDFs and ask questions directly about their content. IntelliChat intelligently retrieves and summarizes information from your documents using HuggingFaceEmbeddings and a Chroma vector database.
General Chat: Engage in free-form conversations powered by the Mistral large language model, running locally via Ollama.
Real-time Web Search: Use the web search: prefix to query the internet and get up-to-date information, thanks to integration with the Google Custom Search API.
User-Friendly Design: A modern, minimal Streamlit UI ensures a smooth and enjoyable experience.
Easy Management: Clear chat history or completely reset your data (documents and database) with simple sidebar controls.
How It Works
IntelliChat AI processes your uploaded PDFs, converting them into searchable embeddings. It then uses the locally hosted Ollama server (running Mistral) to answer your document-specific questions or handle general queries. For web searches, it integrates with Google Custom Search to provide current information, which the LLM then synthesizes.

Setup & Local Installation
Ready to give it a try? Here's how to set it up locally:

1. Clone the Repository

Bash
git clone https://github.com/AYON-ARYAN/INTELLI-CHAT-AI
cd IntelliChat-AI

2. Set up Ollama

IntelliChat AI relies on Ollama to run the Mistral language model locally.

Download & Install Ollama: Follow the instructions on the Ollama website to install Ollama for your operating system.
Run Ollama Server: Open your terminal and start the Ollama server:
Bash
ollama serve
Pull Mistral Model: In a separate terminal, download the Mistral model:
Bash
ollama pull mistral
Ensure Ollama is running and the Mistral model is pulled before starting the Streamlit app.
3. Create a Virtual Environment & Install Dependencies

It's highly recommended to use a virtual environment.

Bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
4. Configure Environment Variables

 Create a .env file in the root directory of the project and add your Google API credentials for web search functionality.

GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
GOOGLE_CSE_ID="YOUR_GOOGLE_CUSTOM_SEARCH_ENGINE_ID"
Get your GOOGLE_API_KEY from the Google Cloud Console.
Create a GOOGLE_CSE_ID by setting up a Google Custom Search Engine. Make sure to enable "Search the entire web" for broad results.
5. Run the Streamlit App

With your Ollama server running and dependencies installed, start the Streamlit application:

Bash
streamlit run app.py
Your browser should automatically open to the Streamlit interface.

Project Structure
├── .env                  # Environment variables (Google API keys)
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── docs/                 # Directory to store uploaded PDF documents
├── db/                   # Directory to store Chroma vector database
└── README.md             # This README file
