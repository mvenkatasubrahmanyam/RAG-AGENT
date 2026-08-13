
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
KNOWLEDGE TRANSFER DOCUMENT

PROJECT TITLE:
Smart College Complaint Tracking Portal

==================================================
1. PROJECT INTRODUCTION
==================================================

The Smart College Complaint Tracking Portal is a web-based application designed to provide a centralized platform for students and college users to submit, track, and manage complaints related to college infrastructure, facilities, and other campus-related issues.

In a traditional college environment, students may report problems verbally, through messages, or by directly approaching staff members. These methods can make it difficult to track complaints, identify responsible departments, monitor progress, and ensure that issues are resolved within a reasonable period.

The Smart College Complaint Tracking Portal solves this problem by providing a structured digital complaint management system. Students can submit complaints through the portal by entering relevant information about the issue. Administrators can view submitted complaints, monitor their status, assign or manage issues, and update the progress of complaints.

The system maintains complaint information digitally so that users can easily track the progress of an issue. This improves transparency, communication, accountability, and efficiency in college complaint management.

==================================================
2. PROBLEM STATEMENT
==================================================

In many colleges, infrastructure and facility-related problems are reported using informal methods. Students may have to personally approach faculty members, department staff, or administrative staff to report an issue.

This process creates several problems. Complaints may be forgotten, misplaced, delayed, or difficult to track. Students may not know whether their complaint has been received or whether action has been taken. Administrators may also find it difficult to maintain records and identify pending complaints.

The Smart College Complaint Tracking Portal provides a centralized digital solution where complaints can be submitted, stored, tracked, and managed systematically.

==================================================
3. OBJECTIVES OF THE PROJECT
==================================================

The main objectives of the Smart College Complaint Tracking Portal are:

1. To provide a centralized platform for submitting college-related complaints.

2. To allow students to report infrastructure and facility problems easily.

3. To maintain digital records of submitted complaints.

4. To allow administrators to view and manage complaints.

5. To provide complaint status tracking.

6. To improve communication between students and college administration.

7. To reduce the possibility of complaints being lost or ignored.

8. To improve transparency in complaint management.

9. To help administrators identify pending and completed complaints.

10. To improve the overall efficiency of the complaint resolution process.

==================================================
4. SYSTEM OVERVIEW
==================================================

The Smart College Complaint Tracking Portal consists of different components that work together to manage complaints.

A student can access the system and submit a complaint by providing information such as the complaint category, description, location, and other required details.

The complaint is stored in the system. The administrator can access the complaint management interface and view the submitted complaint.

The administrator can review the complaint, update its status, and take appropriate action.

The student can subsequently check the status of the complaint through the portal.

The basic workflow of the system is:

Student
   |
   v
Login / Access Portal
   |
   v
Submit Complaint
   |
   v
Complaint Stored in Database
   |
   v
Administrator Reviews Complaint
   |
   v
Complaint Status Updated
   |
   v
Student Tracks Complaint
   |
   v
Complaint Resolved

==================================================
5. USERS OF THE SYSTEM
==================================================

The main users of the Smart College Complaint Tracking Portal are students and administrators.

5.1 STUDENT

The student is the primary user who reports problems through the system.

The student can:

- Access the complaint portal.
- Submit a new complaint.
- Provide complaint details.
- Select an appropriate complaint category.
- Provide the location of the issue.
- Track submitted complaints.
- Check complaint status.
- View updates related to complaints.

5.2 ADMINISTRATOR

The administrator manages the complaints submitted by students.

The administrator can:

- Access the administrative interface.
- View submitted complaints.
- Review complaint details.
- Manage complaints.
- Update complaint status.
- Monitor pending complaints.
- Monitor resolved complaints.
- Manage the overall complaint workflow.

==================================================
6. STUDENT MODULE
==================================================

The Student Module provides the functionality required for students to report and monitor issues.

The student first accesses the portal. After authentication, the student can navigate to the complaint submission section.

The student provides the necessary information about the issue. The information can include the complaint title, complaint category, description, and location.

After submitting the complaint, the system stores the complaint information.

The student can later access the complaint tracking section to check the current status of the complaint.

The Student Module focuses on simplicity and accessibility so that students can report problems without requiring complex procedures.

==================================================
7. ADMINISTRATOR MODULE
==================================================

The Administrator Module is responsible for managing complaints.

The administrator can access the list of complaints submitted by students.

For every complaint, the administrator can review the available information and determine the appropriate action.

The administrator can update the complaint status as the issue progresses.

The administrator can monitor pending complaints and identify complaints that require attention.

The Administrator Module provides a centralized view of complaint information and helps college management maintain an organized complaint resolution process.

==================================================
8. COMPLAINT SUBMISSION
==================================================

Complaint submission is one of the primary functions of the system.

A student enters the relevant details of an issue into the complaint form.

Typical complaint information can include:

- Complaint title.
- Complaint category.
- Complaint description.
- Location of the issue.
- Date of submission.
- User information.

After entering the information, the student submits the complaint.

The system validates the required information and stores the complaint.

A complaint should contain enough information to allow the administrator or responsible department to understand the issue and take appropriate action.

==================================================
9. COMPLAINT CATEGORIES
==================================================

Complaints can be organized into categories to make complaint management easier.

Possible categories include:

1. Classroom Infrastructure
2. Electrical Problems
3. Plumbing Problems
4. Laboratory Equipment
5. Computer and Network Issues
6. Furniture Problems
7. Cleanliness and Sanitation
8. Water Supply
9. Hostel Facilities
10. Library Facilities
11. Campus Infrastructure
12. Other College Facilities

Categorization helps administrators identify the nature of an issue and manage complaints more efficiently.

==================================================
10. COMPLAINT DESCRIPTION
==================================================

The complaint description contains detailed information about the problem.

Students should provide a clear description of the issue.

For example, instead of submitting a complaint with only the text "fan not working", the student can provide additional information such as the classroom location and the nature of the problem.

A clear complaint description helps the administrator understand the issue without requiring repeated communication with the student.

==================================================
11. COMPLAINT LOCATION
==================================================

The location field identifies where the problem exists.

Examples of locations include:

- Classroom
- Laboratory
- Library
- Hostel
- Washroom
- Corridor
- Department Office
- Seminar Hall
- Campus Area

The location information helps the responsible staff identify where action is required.

==================================================
12. COMPLAINT STATUS
==================================================

Complaint status is used to indicate the current state of a complaint.

A complaint can move through different stages during its lifecycle.

Common complaint statuses include:

1. Submitted
2. Pending
3. In Progress
4. Resolved
5. Closed

SUBMITTED:
The complaint has been successfully submitted by the student.

PENDING:
The complaint has been received but action has not yet started.

IN PROGRESS:
The complaint is currently being investigated or worked on.

RESOLVED:
The reported issue has been addressed.

CLOSED:
The complaint process has been completed and the complaint is no longer active.

Status tracking allows students and administrators to understand the current state of each complaint.

==================================================
13. COMPLAINT TRACKING
==================================================

Complaint tracking allows students to monitor the progress of their submitted complaints.

Instead of repeatedly approaching college staff to ask about the progress of an issue, students can check the complaint status through the portal.

The tracking feature improves transparency because students can see whether their complaint has been submitted, is pending, is being processed, or has been resolved.

Administrators can also use complaint tracking to monitor unresolved issues.

==================================================
14. COMPLAINT MANAGEMENT WORKFLOW
==================================================

The complaint management workflow consists of several stages.

Stage 1:
The student identifies an issue.

Stage 2:
The student accesses the portal.

Stage 3:
The student enters the complaint information.

Stage 4:
The student submits the complaint.

Stage 5:
The system stores the complaint.

Stage 6:
The administrator views the complaint.

Stage 7:
The administrator reviews the issue.

Stage 8:
The complaint is processed.

Stage 9:
The administrator updates the complaint status.

Stage 10:
The student checks the updated status.

Stage 11:
The issue is resolved.

Stage 12:
The complaint is closed.

==================================================
15. SYSTEM ARCHITECTURE
==================================================

The system follows a web-based application architecture.

The main components are:

1. User Interface
2. Application Server
3. Complaint Management Logic
4. Database
5. Administrator Interface

The user interacts with the application through a web browser.

The application server receives user requests and processes them.

Complaint information is stored in the database.

The administrator interacts with the application through the administrative interface.

The architecture can be represented as:

User
 |
 v
Web Interface
 |
 v
Application Server
 |
 +----------------+
 |                |
 v                v
Complaint Logic   Database
 |
 v
Administrator Interface

==================================================
16. DATABASE
==================================================

The database is responsible for storing system information.

Possible complaint-related information includes:

- Complaint ID
- Student ID
- Student Name
- Complaint Title
- Complaint Category
- Complaint Description
- Complaint Location
- Submission Date
- Complaint Status
- Resolution Information

The Complaint ID can be used as a unique identifier for each complaint.

The database allows the application to retrieve, update, and manage complaint records.

==================================================
17. COMPLAINT ID
==================================================

Each complaint should have a unique identification value.

The Complaint ID helps the system distinguish one complaint from another.

For example:

Complaint ID: CMP001
Complaint ID: CMP002
Complaint ID: CMP003

The Complaint ID can also be used when tracking or managing a specific complaint.

==================================================
18. AUTHENTICATION
==================================================

Authentication is used to identify users accessing the system.

A student can log into the system using appropriate credentials.

An administrator can use administrator credentials to access management functionality.

Authentication helps prevent unauthorized users from accessing protected areas of the application.

==================================================
19. AUTHORIZATION
==================================================

Authorization determines what actions a user is allowed to perform.

Students and administrators have different responsibilities.

Students should be able to submit and track their complaints.

Administrators should be able to view and manage complaints.

Role-based access helps protect administrative functionality from unauthorized access.

==================================================
20. ADMIN DASHBOARD
==================================================

The administrator dashboard provides an overview of complaint information.

The dashboard can display information such as:

- Total complaints.
- Pending complaints.
- Complaints in progress.
- Resolved complaints.
- Closed complaints.

The dashboard helps administrators quickly understand the current complaint situation.

==================================================
21. SEARCH AND FILTERING
==================================================

Search and filtering can help administrators find specific complaints.

Complaints can be filtered based on:

- Complaint ID
- Category
- Location
- Status
- Date
- Student

Filtering makes it easier to manage a large number of complaints.

==================================================
22. NOTIFICATION AND COMMUNICATION
==================================================

The system can be extended to provide notifications when important changes occur.

For example, a student can receive a notification when:

- A complaint is submitted.
- A complaint is accepted.
- A complaint status changes.
- A complaint is resolved.

Notifications can improve communication between students and administrators.

==================================================
23. SECURITY
==================================================

Security is important because the system handles user and complaint information.

Important security practices include:

- Authentication.
- Authorization.
- Secure password handling.
- Input validation.
- Protection against unauthorized access.
- Secure API configuration.
- Protection of environment variables.
- Avoiding sensitive information in source code.

API keys should never be directly stored in public source code.

For deployment, sensitive credentials should be stored as environment variables.

==================================================
24. TECHNOLOGY STACK
==================================================

The Smart College Complaint Tracking Portal can be implemented using modern web technologies.

Possible technologies include:

Frontend:
HTML, CSS, JavaScript

Backend:
Python and Flask

Database:
A relational or document-based database depending on system requirements.

Deployment:
Cloud-based hosting platforms such as Render.

Version Control:
Git and GitHub.

The technology stack can be changed depending on project requirements.

==================================================
25. FLASK APPLICATION
==================================================

Flask is used as the backend web framework for the application.

The Flask application handles HTTP requests and provides routes for different operations.

The application can contain routes such as:

/
The home page.

 /ask
Receives a question from the user and returns an answer from the Agentic RAG system.

Other routes can be added for authentication, complaint submission, complaint tracking, and administration.

==================================================
26. AGENTIC RAG SYSTEM
==================================================

The Agentic RAG system combines retrieval-augmented generation with an AI agent.

RAG stands for Retrieval-Augmented Generation.

The system first searches the available knowledge base for information relevant to the user's question.

The retrieved information is then provided to the language model.

The language model generates an answer based on the retrieved information.

An agent can use tools to decide when information needs to be retrieved.

The overall process is:

User Question
     |
     v
AI Agent
     |
     v
Retrieval Tool
     |
     v
Vector Database
     |
     v
Relevant KT Information
     |
     v
Language Model
     |
     v
Final Answer

==================================================
27. KNOWLEDGE BASE
==================================================

The knowledge base contains the information used by the RAG system.

In this project, the Knowledge Transfer document acts as the primary knowledge source.

The KT document contains information about the project, its objectives, modules, workflow, users, technologies, and other relevant information.

The document is divided into smaller chunks before creating embeddings.

==================================================
28. TEXT CHUNKING
==================================================

Large documents are divided into smaller pieces called chunks.

Chunking makes it easier for the retrieval system to find relevant information.

The application uses a text splitter to divide the KT document.

The RecursiveCharacterTextSplitter can be configured using parameters such as chunk size and chunk overlap.

Chunk overlap helps preserve context between neighboring chunks.

==================================================
29. EMBEDDINGS
==================================================

Embeddings convert text into numerical vector representations.

Text with similar meanings can have similar vector representations.

The RAG system generates embeddings for the chunks of the KT document.

These embeddings are stored in a vector database.

When the user asks a question, the question is also converted into an embedding.

The system compares the question embedding with stored document embeddings to find relevant information.

==================================================
30. FAISS VECTOR DATABASE
==================================================

FAISS is a vector similarity search library.

The system uses FAISS to store and search the embeddings created from the KT document.

When a user asks a question, FAISS performs similarity search and returns the most relevant document chunks.

These retrieved chunks provide context for the language model.

==================================================
31. RETRIEVAL PROCESS
==================================================

The retrieval process consists of the following steps:

1. User submits a question.

2. The question is converted into an embedding.

3. The vector store searches for similar document chunks.

4. The most relevant chunks are retrieved.

5. Retrieved information is provided to the AI system.

6. The AI system generates an answer using the retrieved information.

==================================================
32. LANGUAGE MODEL
==================================================

The language model is responsible for generating the final response.

The application uses a Google Gemini-compatible language model through LangChain.

The model receives the user's question and relevant retrieved context.

The temperature can be set to zero when more deterministic responses are preferred.

==================================================
33. RETRIEVAL TOOL
==================================================

The retrieval tool allows the agent to search the KT knowledge base.

The tool receives a query.

It performs similarity search against the FAISS vector store.

The most relevant documents are returned to the agent.

The agent can use this information to answer the user's question.

==================================================
34. AGENT
==================================================

The agent is responsible for processing user questions and using available tools.

The agent has access to the retrieval tool.

When a question requires information from the KT document, the agent can use the retrieval tool.

The retrieved information is then used to produce the final response.

The agent should avoid inventing project-specific information.

If the required information is not present in the KT document, the system should clearly state that the information is not available in the knowledge base.

==================================================
35. RAG ANSWERING PRINCIPLE
==================================================

The RAG system should prioritize information retrieved from the KT document.

For example, if the user asks:

"What is the purpose of the project?"

The system should search the KT document and use the relevant project description to generate the answer.

If the user asks something unrelated to the KT document, the system should not create unsupported project information.

This helps reduce hallucination and improves answer reliability.

==================================================
36. EXAMPLE QUESTIONS
==================================================

The following questions can be asked to the Agentic RAG system:

1. What is the purpose of the Smart College Complaint Tracking Portal?

2. What problem does the system solve?

3. What are the main objectives of the project?

4. Who are the users of the system?

5. What can a student do in the system?

6. What can an administrator do?

7. How does complaint submission work?

8. How does complaint tracking work?

9. What are the different complaint statuses?

10. What information is stored for a complaint?

11. What is the role of the administrator?

12. What is the role of the student?

13. What technologies are used?

14. What is RAG?

15. How does the Agentic RAG system work?

16. What is FAISS?

17. What are embeddings?

18. Why is text chunking required?

19. How does the retrieval process work?

20. How does the system improve complaint management?

==================================================
37. DEPLOYMENT
==================================================

The application can be deployed using a cloud hosting platform such as Render.

The project requires an application file and dependency file.

The main application file is:

app.py

The dependency file is:

requirements.txt

The requirements file contains the Python packages required by the application.

A typical deployment process is:

1. Develop the application in Google Colab or a local environment.

2. Create app.py.

3. Create requirements.txt.

4. Upload the files to GitHub.

5. Create a Web Service on Render.

6. Connect the GitHub repository.

7. Set the Python environment.

8. Configure the build command.

9. Configure the start command.

10. Add required environment variables.

11. Deploy the application.

==================================================
38. ENVIRONMENT VARIABLES
==================================================

Sensitive values such as API keys should be stored as environment variables.

The Gemini API key can be stored using the environment variable:

GEMINI_API_KEY

The application can retrieve the value from the environment at runtime.

This prevents the API key from being stored directly inside the source code.

The same environment variable must be configured on the deployment platform.

==================================================
39. GITHUB
==================================================

GitHub is used to store and manage the source code.

The repository can contain:

app.py
requirements.txt
README.md

GitHub allows changes to the application to be tracked.

When a new commit is pushed to the repository, the deployment platform can automatically rebuild and deploy the application.

==================================================
40. RENDER DEPLOYMENT
==================================================

Render can be used to host the Flask application.

The build command installs the dependencies:

pip install -r requirements.txt

The start command starts the Flask application using Gunicorn:

gunicorn app:app

The application must listen on the port provided by the deployment environment.

The Flask application can use the PORT environment variable.

==================================================
41. APPLICATION FLOW
==================================================

The complete application flow is:

1. User opens the web application.

2. User enters a question.

3. The frontend sends the question to the Flask /ask endpoint.

4. Flask receives the question.

5. The Agent processes the question.

6. The retrieval tool searches the FAISS vector store.

7. Relevant KT chunks are retrieved.

8. The language model receives the relevant information.

9. The language model generates the answer.

10. Flask returns the answer as JSON.

11. The frontend displays the answer to the user.

==================================================
42. ADVANTAGES
==================================================

The Smart College Complaint Tracking Portal provides several advantages.

1. Centralized complaint management.

2. Easy complaint submission.

3. Digital complaint records.

4. Complaint status tracking.

5. Improved transparency.

6. Better accountability.

7. Reduced manual work.

8. Faster access to complaint information.

9. Organized complaint management.

10. Improved communication.

The Agentic RAG component additionally provides an intelligent question-answering interface based on the project knowledge base.

==================================================
43. LIMITATIONS
==================================================

Possible limitations include:

1. The quality of answers depends on the quality of the KT document.

2. Incorrect or incomplete knowledge can affect retrieval results.

3. The system requires an internet connection when using cloud-based AI services.

4. API usage may have limits depending on the provider.

5. Large knowledge bases may require additional optimization.

6. Vector search quality depends on the quality of embeddings and chunking.

7. The application requires appropriate security configuration for production use.

==================================================
44. FUTURE ENHANCEMENTS
==================================================

Possible future enhancements include:

1. Email notifications.

2. WhatsApp or SMS notifications.

3. Image upload for complaints.

4. Automatic complaint categorization.

5. Automatic priority detection.

6. Department-based complaint assignment.

7. Advanced analytics dashboard.

8. Complaint resolution time analysis.

9. Mobile application support.

10. AI-based complaint summarization.

11. Multilingual support.

12. Voice-based complaint submission.

13. Improved authentication.

14. Advanced search.

15. Historical complaint analysis.

==================================================
45. TESTING
==================================================

The system should be tested to verify that all major functions work correctly.

Important test cases include:

1. Opening the application.

2. Submitting a valid complaint.

3. Submitting an incomplete complaint.

4. Viewing submitted complaints.

5. Updating complaint status.

6. Tracking complaint status.

7. Asking questions to the RAG system.

8. Retrieving relevant KT information.

9. Handling questions not covered by the KT.

10. Checking application behavior after deployment.

==================================================
46. ERROR HANDLING
==================================================

The application should handle errors gracefully.

Possible errors include:

- Missing API key.
- Invalid user input.
- Empty questions.
- Retrieval errors.
- Language model errors.
- Database errors.
- Network errors.
- Deployment configuration errors.

The system should return meaningful error messages instead of crashing without explanation.

==================================================
47. SECURITY OF API KEYS
==================================================

API keys are sensitive credentials.

The Gemini API key should not be hard-coded into app.py.

Instead, it should be stored as:

GEMINI_API_KEY

in the environment variables of the deployment platform.

The API key should not be committed to GitHub.

If an API key is accidentally exposed publicly, it should be revoked and replaced.

==================================================
48. ROLE OF LANGCHAIN
==================================================

LangChain provides components used to build the AI application.

It can be used for:

- Language model integration.
- Document processing.
- Text splitting.
- Embeddings.
- Vector stores.
- Retrieval.
- Tools.
- Agents.

In this project, LangChain connects the Gemini model, document processing components, FAISS vector store, retrieval tool, and agent.

==================================================
49. ROLE OF GEMINI
==================================================

Gemini is used as the language model component of the system.

The model processes questions and generates natural-language responses.

Gemini can also be used for generating embeddings through the appropriate embedding model.

The application accesses Gemini through the LangChain Google integration.

==================================================
50. ROLE OF FAISS
==================================================

FAISS acts as the vector search component.

The KT document is converted into chunks.

Each chunk is converted into an embedding.

The embeddings are stored in FAISS.

When a user asks a question, the system searches FAISS to find the most relevant chunks.

==================================================
51. ROLE OF THE KT DOCUMENT
==================================================

The KT document is the main knowledge source for the RAG system.

The system does not need to rely entirely on the language model's general knowledge.

Instead, the KT document provides project-specific information.

This makes the system useful for answering questions about the specific project.

The KT document should be clear, organized, and sufficiently detailed.

==================================================
52. COMPLETE RAG PIPELINE
==================================================

The complete RAG pipeline is:

KT Document
     |
     v
Document Creation
     |
     v
Text Splitting
     |
     v
Document Chunks
     |
     v
Embeddings
     |
     v
FAISS Vector Store
     |
     v
User Question
     |
     v
Question Retrieval
     |
     v
Relevant Chunks
     |
     v
AI Agent
     |
     v
Gemini Language Model
     |
     v
Generated Answer
     |
     v
Web Interface

==================================================
53. SAMPLE QUESTION AND ANSWER FLOW
==================================================

User Question:

"What is the purpose of the Smart College Complaint Tracking Portal?"

Step 1:
The question is sent to the Flask application.

Step 2:
The AI agent processes the question.

Step 3:
The retrieval tool searches the FAISS vector store.

Step 4:
Relevant chunks containing the project purpose are retrieved.

Step 5:
The retrieved information is provided to the language model.

Step 6:
The language model generates a response.

Step 7:
The Flask application sends the response back to the webpage.

Step 8:
The webpage displays the answer.

==================================================
54. PROJECT SUMMARY
==================================================

The Smart College Complaint Tracking Portal is a digital complaint management system designed to improve the way college-related issues are reported and managed.

The system provides students with a convenient method for submitting and tracking complaints.

Administrators can manage complaints, monitor their progress, and update complaint statuses.

The Agentic RAG component provides an intelligent question-answering interface using the Knowledge Transfer document as the primary knowledge source.

The combination of Flask, LangChain, Gemini, embeddings, FAISS, and an AI agent provides a complete AI-powered knowledge retrieval and question-answering application.

==================================================
55. FINAL CONCLUSION
==================================================

The Smart College Complaint Tracking Portal provides a structured and centralized approach to college complaint management.

By digitizing the complaint process, the system can improve accessibility, transparency, organization, and communication between students and administrators.

The Agentic RAG component further enhances the project by allowing users to ask natural-language questions about the project knowledge base.

The RAG pipeline processes the Knowledge Transfer document, creates embeddings, stores them in FAISS, retrieves relevant information, and uses a language model to generate answers.

The application can be packaged using app.py and requirements.txt and deployed to a cloud platform such as Render.

The project demonstrates how traditional web application development can be combined with modern Retrieval-Augmented Generation and agent-based AI techniques to create an intelligent application.
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
                "answer": "Please enter a question."
            })

        # Run the RAG agent
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        final_message = result["messages"][-1]
        answer = final_message.content

        def extract_text(content):

            if isinstance(content, str):
                return content

            if isinstance(content, list):
                parts = []

                for item in content:

                    if isinstance(item, str):
                        parts.append(item)

                    elif isinstance(item, dict):

                        if "text" in item:
                            parts.append(extract_text(item["text"]))

                        elif "content" in item:
                            parts.append(extract_text(item["content"]))

                return "\n".join(parts)

            if isinstance(content, dict):

                if "text" in content:
                    return extract_text(content["text"])

                if "content" in content:
                    return extract_text(content["content"])

                return ""

            return str(content)

        final_answer = extract_text(answer)

        return jsonify({
            "answer": final_answer
        })

    except Exception as e:

        return jsonify({
            "answer": f"Error: {str(e)}"
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
