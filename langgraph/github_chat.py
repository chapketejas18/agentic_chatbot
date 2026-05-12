# ==========================================================
# github_chat.py
# Advanced GitHub Repository Chatbot
# LangGraph + Groq + Streamlit + ChromaDB
# ==========================================================

# ==========================================================
# INSTALL REQUIRED PACKAGES
# ==========================================================

# pip install -U streamlit langgraph langchain \
# langchain-groq langchain-community \
# langchain-text-splitters langchain-huggingface \
# sentence-transformers chromadb \
# python-dotenv GitPython tiktoken


# ==========================================================
# IMPORTS
# ==========================================================

import os
import shutil
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv
from git import Repo

from langchain_groq import ChatGroq

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from langchain_community.document_loaders import (
    TextLoader
)

from langchain_community.vectorstores import (
    Chroma
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.graph.message import (
    add_messages
)


# ==========================================================
# LOAD ENV VARIABLES
# ==========================================================

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# ==========================================================
# MODEL
# ==========================================================

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0
)


# ==========================================================
# STREAMLIT CONFIG
# ==========================================================

st.set_page_config(
    page_title="GitHub Repo Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GitHub Repository Chatbot")

st.markdown("""
Chat with any GitHub repository using:

- LangGraph
- Groq
- ChromaDB
- Streamlit
- Repository Metadata Understanding
""")


# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "repo_loaded" not in st.session_state:
    st.session_state.repo_loaded = False

if "repo_metadata" not in st.session_state:
    st.session_state.repo_metadata = None


# ==========================================================
# GITHUB URL INPUT
# ==========================================================

repo_url = st.text_input(
    "Enter GitHub Repository URL",
    placeholder="https://github.com/username/repository"
)


# ==========================================================
# CLONE REPOSITORY
# ==========================================================

def clone_repo(repo_url):

    repo_path = "./temp_repo"

    # Remove old repo
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    # Clone repo
    Repo.clone_from(repo_url, repo_path)

    return repo_path


# ==========================================================
# GET REPOSITORY METADATA
# ==========================================================

def get_repo_metadata(repo_path):

    repo = Repo(repo_path)

    # Get Branches
    branches = [
        branch.name
        for branch in repo.branches
    ]

    # Create Folder Tree
    tree = []

    ignored_directories = [
        ".git",
        "__pycache__",
        "node_modules",
        ".next",
        "dist",
        "build",
        ".venv",
        "venv"
    ]

    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [
            d for d in dirs
            if d not in ignored_directories
        ]

        level = root.replace(
            repo_path,
            ""
        ).count(os.sep)

        indent = "  " * level

        folder_name = os.path.basename(root)

        if folder_name == "":
            folder_name = "root"

        tree.append(
            f"{indent}📁 {folder_name}"
        )

        sub_indent = "  " * (level + 1)

        for file in files:

            tree.append(
                f"{sub_indent}📄 {file}"
            )

    return {
        "branches": branches,
        "tree": "\n".join(tree)
    }


# ==========================================================
# LOAD REPOSITORY FILES
# ==========================================================

def load_repo_files(repo_path):

    documents = []

    allowed_extensions = [
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".json",
        ".md",
        ".txt",
        ".html",
        ".css",
        ".yml",
        ".yaml",
        ".xml",
        ".sql"
    ]

    ignored_directories = [
        ".git",
        "__pycache__",
        "node_modules",
        ".next",
        "dist",
        "build",
        ".venv",
        "venv"
    ]

    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [
            d for d in dirs
            if d not in ignored_directories
        ]

        for file in files:

            if file.endswith(
                tuple(allowed_extensions)
            ):

                file_path = os.path.join(
                    root,
                    file
                )

                try:

                    loader = TextLoader(
                        file_path,
                        encoding="utf-8"
                    )

                    docs = loader.load()

                    for doc in docs:

                        doc.metadata["source"] = (
                            file_path
                        )

                    documents.extend(docs)

                except Exception:
                    pass

    return documents


# ==========================================================
# CREATE VECTOR DATABASE
# ==========================================================

def create_vector_store(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(
        documents
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Remove old ChromaDB
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore


# ==========================================================
# LANGGRAPH STATE
# ==========================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ==========================================================
# CHATBOT NODE
# ==========================================================

def chatbot(state: State):

    user_question = (
        state["messages"][-1].content
    )

    # Retrieve Relevant Docs
    docs = (
        st.session_state.vectorstore
        .similarity_search(
            user_question,
            k=5
        )
    )

    context = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    branches = (
        st.session_state.repo_metadata[
            "branches"
        ]
    )

    tree = (
        st.session_state.repo_metadata[
            "tree"
        ]
    )

    prompt = f"""
You are an expert GitHub repository assistant.

You help users:
- Understand repository structure
- Explain codebase
- Explain architecture
- Explain functions/classes
- Understand folders/files
- Explain branches
- Debug repository code

========================
REPOSITORY BRANCHES
========================

{branches}

========================
REPOSITORY STRUCTURE
========================

{tree}

========================
REPOSITORY CODE CONTEXT
========================

{context}

========================
USER QUESTION
========================

{user_question}

Rules:
- Answer ONLY using repository data.
- If information is unavailable,
say:
"I could not find that in the repository."
"""

    response = model.invoke(prompt)

    return {
        "messages": [response]
    }


# ==========================================================
# BUILD LANGGRAPH
# ==========================================================

graph = StateGraph(State)

graph.add_node(
    "chatbot",
    chatbot
)

graph.add_edge(
    START,
    "chatbot"
)

graph.add_edge(
    "chatbot",
    END
)

graph_builder = graph.compile()


# ==========================================================
# LOAD REPOSITORY BUTTON
# ==========================================================

if st.button("Load Repository"):

    if not repo_url:

        st.warning(
            "Please enter repository URL"
        )

    else:

        try:

            # Clone Repo
            with st.spinner(
                "Cloning repository..."
            ):

                repo_path = clone_repo(
                    repo_url
                )

            # Get Metadata
            with st.spinner(
                "Reading repository structure..."
            ):

                repo_metadata = (
                    get_repo_metadata(
                        repo_path
                    )
                )

            # Load Files
            with st.spinner(
                "Reading repository files..."
            ):

                documents = load_repo_files(
                    repo_path
                )

            # Create Vector DB
            with st.spinner(
                "Creating vector database..."
            ):

                vectorstore = (
                    create_vector_store(
                        documents
                    )
                )

            # Save Session State
            st.session_state.vectorstore = (
                vectorstore
            )

            st.session_state.repo_metadata = (
                repo_metadata
            )

            st.session_state.repo_loaded = True

            # Success Message
            st.success(f"""
Repository Loaded Successfully!

Files Loaded: {len(documents)}

Branches Found:
{repo_metadata["branches"]}
""")

            # Show Repository Structure
            with st.expander(
                "📁 Repository Structure"
            ):

                st.code(
                    repo_metadata["tree"]
                )

        except Exception as e:

            st.error(
                f"Error:\n{str(e)}"
            )


# ==========================================================
# CHAT SECTION
# ==========================================================

if st.session_state.repo_loaded:

    st.divider()

    st.subheader(
        "💬 Chat With Repository"
    )

    # Display Messages
    for message in (
        st.session_state.messages
    ):

        if isinstance(
            message,
            HumanMessage
        ):

            with st.chat_message("user"):

                st.markdown(
                    message.content
                )

        elif isinstance(
            message,
            AIMessage
        ):

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    message.content
                )

    # Chat Input
    user_input = st.chat_input(
        "Ask something about repository..."
    )

    if user_input:

        # Store User Message
        human_message = HumanMessage(
            content=user_input
        )

        st.session_state.messages.append(
            human_message
        )

        # Show User Message
        with st.chat_message("user"):

            st.markdown(user_input)

        # Generate Response
        with st.spinner("Thinking..."):

            result = graph_builder.invoke({
                "messages":
                st.session_state.messages
            })

            ai_response = result[
                "messages"
            ][-1]

        # Show AI Response
        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                ai_response.content
            )

        # Save AI Response
        st.session_state.messages.append(
            AIMessage(
                content=ai_response.content
            )
        )

else:

    st.info("""
Steps:

1. Enter GitHub Repository URL
2. Click "Load Repository"
3. Chat with the repository

You can ask:
- What branches exist?
- Explain project structure
- Which file contains API code?
- Explain authentication flow
- Which folder contains frontend?
- Explain LangGraph implementation
""")