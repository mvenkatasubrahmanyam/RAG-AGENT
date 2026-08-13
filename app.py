
import os
from flask import Flask, request, jsonify

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent

app = Flask(__name__)

# Get Gemini API key from Render environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")


# =========================
# 1. KT DOCUMENT
# =========================

big_paragraph = """
PASTE YOUR COMPLETE KT DOCUMENT HERE
"""


# =========================
# 2. CREATE DOCUMENT
# =========================

documents = [
    Document(page_content=big_paragraph)
]


# =========================
# 3. SPLIT DOCUMENT
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)


# =========================
# 4. CREATE EMBEDDINGS
# =========================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)


# =========================
# 5. CREATE FAISS VECTOR STORE
# =========================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# =========================
# 6. GEMINI LLM
# =========================

llm = ChatGoogleGenerativeAI(
    model="models/gemma-4-31b-it",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# =========================
# 7. RETRIEVAL TOOL
# =========================

@tool
def retrieve_kt_context(query: str):
    """Search the KT document for information relevant to the user's question."""

    retrieved_docs = vector_store.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    return context


# =========================
# 8. AGENTIC RAG
# =========================

agent = create_agent(
    llm,
    tools=[retrieve_kt_context],
    system_prompt="""
    You are a Knowledge Transfer assistant.

    Answer questions using the retrieved information from the KT document.

    Do not invent project-specific information.

    If the answer is not available in the KT document,
    clearly say that the information is not available
    in the Knowledge Transfer document.
    """
)


# =========================
# 9. HOME PAGE
# =========================

@app.route("/")
def home():

    return """
    <html>
    <head>
        <title>KT Agentic RAG</title>
    </head>

    <body>

        <h1>KT Agentic RAG Assistant</h1>

        <input
            id="question"
            type="text"
            placeholder="Ask a question"
            style="width:400px;padding:10px;"
        >

        <button onclick="askQuestion()">
            Ask
        </button>

        <h3>Answer:</h3>

        <div id="answer"></div>

        <script>

        async function askQuestion() {

            let question =
                document.getElementById("question").value;

            let response = await fetch("/ask", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    question: question
                })

            });

            let data = await response.json();

            document.getElementById("answer").innerText =
                data.answer || data.error;

        }

        </script>

    </body>
    </html>
    """


# =========================
# 10. ASK ROUTE
# =========================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        question = data.get("question", "").strip()

        if not question:

            return jsonify({
                "error": "Please enter a question."
            })

        result = agent.invoke({

            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]

        })

        answer = result["messages"][-1].content

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# 11. RUN APPLICATION
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
