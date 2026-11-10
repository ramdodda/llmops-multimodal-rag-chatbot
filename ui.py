import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 180
MAX_FILE_SIZE = 20 * 1024 * 1024

FILE_ICONS = {"pdf": "📄", "csv": "📊", "image": "🖼️"}

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="LLMOps Multimodal Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #312e81);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.1rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #e0e7ff;
        margin-bottom: 0;
        font-size: 1rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 12px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Session state
# --------------------------------------------------
DEFAULTS = {
    "active_file": None,
    "file_info": None,
    "file_messages": [],
    "general_messages": [],
    "last_uploaded_id": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------------------------------
# API helpers
# --------------------------------------------------
def api_get(path: str):
    return requests.get(f"{API_URL}{path}", timeout=10)


def api_post_json(path: str, payload: dict):
    return requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)


def display_api_error(response):
    try:
        body = response.json()
        detail = body.get("detail", body)
    except ValueError:
        detail = response.text or "No error details returned."

    st.error(f"API error ({response.status_code}): {detail}")

    if response.status_code == 422:
        st.warning(
            "The backend could not process this request. "
            "Check the FastAPI logs for the underlying exception, "
            "including Google credentials, Vertex AI permissions, "
            "embedding errors, or unsupported file content."
        )


def display_connection_error(exc):
    st.error(
        "Cannot connect to FastAPI. "
        f"Configured backend URL: {API_URL}. "
        f"Details: {exc}"
    )


def display_token_usage(data: dict):
    usage = data.get("token_usage") or {}

    if not usage:
        # Expected when no chunk passed the similarity threshold:
        # the backend skips the LLM call entirely.
        st.caption("No LLM call was made for this answer.")
        return

    st.markdown("##### Token usage")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Input", usage.get("prompt_token_count", 0))
    col2.metric("Output", usage.get("candidates_token_count", 0))
    col3.metric("Thinking", usage.get("thoughts_token_count") or 0)
    col4.metric("Total", usage.get("total_token_count", 0))

    with st.expander("View complete token metadata"):
        st.json(usage)


def display_retrieved_chunks(data: dict):
    retrieved = data.get("retrieved_chunks") or []

    if not retrieved:
        return

    with st.expander("View retrieved document chunks"):
        for index, item in enumerate(retrieved, start=1):
            st.markdown(
                f"**Retrieved chunk {index}** "
                f"— {item.get('filename', '')}, ID: {item.get('chunk_id', 'N/A')}"
            )

            similarity = item.get("similarity")
            if similarity is not None:
                st.caption(f"Similarity: {similarity}")

            chunk_text = item.get("text") or item.get("content")
            if chunk_text:
                st.write(chunk_text)

            st.divider()


def render_assistant_details(message: dict):
    usage = message.get("token_usage")
    if usage:
        with st.expander("Token usage"):
            st.json(usage)

    chunks = message.get("retrieved_chunks") or []
    if chunks:
        with st.expander("Retrieved document chunks"):
            st.json(chunks)


def handle_chat_response(response, chat_type: str):
    """Render a successful API response and save it to session history."""
    data = response.json()
    answer = data.get("answer") or "The model returned no text."

    st.markdown(answer)
    display_token_usage(data)

    if chat_type == "file":
        display_retrieved_chunks(data)

    message = {
        "role": "assistant",
        "content": answer,
        "token_usage": data.get("token_usage"),
        "retrieved_chunks": data.get("retrieved_chunks", []),
    }

    if chat_type == "file":
        st.session_state.file_messages.append(message)
    else:
        st.session_state.general_messages.append(message)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📚 LLMOps Multimodal Chatbot</h1>
        <p>
            Upload documents, ask questions, and explore answers
            powered by Google Gemini and Vertex AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.markdown("**Backend URL**")
    st.code(API_URL, language=None)

    if st.button("Check API health", use_container_width=True):
        try:
            response = api_get("/health")

            if response.ok:
                health = response.json()
                st.success("FastAPI is healthy")
                st.json(health)

                if health.get("qdrant") != "connected":
                    st.warning("Qdrant is not reporting a connected status.")
            else:
                display_api_error(response)

        except requests.RequestException as exc:
            display_connection_error(exc)

    st.divider()

    top_k = st.slider(
        "Chunks to retrieve (top_k)",
        min_value=1,
        max_value=10,
        value=4,
        help="How many document chunks are searched for each question.",
    )

    st.divider()

    st.markdown("**Supported formats**")
    st.markdown(
        "- 📄 PDF documents\n"
        "- 📊 CSV spreadsheets\n"
        "- 🖼️ JPG / JPEG images\n"
        "- 🖼️ PNG images"
    )

    st.divider()

    if st.session_state.active_file:
        st.markdown("**Last uploaded file**")
        st.write(st.session_state.active_file)

    st.caption(
        "Documents are indexed in Qdrant and document metadata "
        "is stored in SQLite."
    )

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.file_messages = []
        st.session_state.general_messages = []
        st.rerun()

# --------------------------------------------------
# 1. Upload section
# --------------------------------------------------
st.header("1. Upload a file")

st.write(
    "Choose a PDF, CSV, or image. The backend determines how the "
    "uploaded file is processed and indexed."
)

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "csv", "jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Maximum supported file size: 20 MB.",
)

if uploaded_file is not None:
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    file_col, preview_col = st.columns([1, 2])

    with file_col:
        st.markdown("**Selected file**")
        st.write(uploaded_file.name)
        st.write(f"Size: {uploaded_file.size / 1024:.1f} KB")
        st.write(f"Format: {extension.upper()}")

    with preview_col:
        if extension in ("jpg", "jpeg", "png"):
            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                use_container_width=True,
            )

        elif extension == "csv":
            try:
                uploaded_file.seek(0)
                preview_df = pd.read_csv(uploaded_file).head(10)
                st.dataframe(preview_df, use_container_width=True)
                uploaded_file.seek(0)
            except Exception as exc:
                st.info(f"CSV preview unavailable: {exc}")

        else:
            st.info("PDF selected. Upload it to extract and index its text.")

    if st.button(
        "⬆️ Upload and process file",
        type="primary",
        use_container_width=True,
    ):
        if uploaded_file.size == 0:
            st.error("The selected file is empty.")

        elif uploaded_file.size > MAX_FILE_SIZE:
            st.error("The file exceeds the 20 MB size limit.")

        else:
            try:
                uploaded_file.seek(0)

                with st.spinner("Uploading file and processing its contents..."):
                    response = requests.post(
                        f"{API_URL}/upload-file",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        },
                        timeout=TIMEOUT,
                    )

                if response.ok:
                    info = response.json()

                    st.session_state.active_file = uploaded_file.name
                    st.session_state.file_info = info
                    st.session_state.file_messages = []
                    # Pre-select the new file in the document selector.
                    st.session_state.last_uploaded_id = info.get("document_id")

                    st.success(
                        info.get(
                            "message",
                            "File uploaded and processed successfully.",
                        )
                    )

                    st.rerun()

                else:
                    display_api_error(response)

            except requests.Timeout:
                st.error(
                    "The upload request timed out. Check the backend logs "
                    "to determine whether processing completed."
                )

            except requests.RequestException as exc:
                display_connection_error(exc)

# --------------------------------------------------
# Active file information
# --------------------------------------------------
if st.session_state.file_info:
    info = st.session_state.file_info

    st.divider()
    st.subheader("Document information")

    st.write(
        f"**File:** {info.get('filename', st.session_state.active_file or 'N/A')}"
    )
    st.write(f"**Type:** {str(info.get('file_type', 'N/A')).upper()}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Pages", info.get("pages", "N/A"))
    col2.metric("Chunks", info.get("chunks_created", "N/A"))
    col3.metric("Embedding model", info.get("embedding_model", "Not applicable"))

    with st.expander("Full upload response"):
        st.json(info)

# --------------------------------------------------
# 2. Document selector
# --------------------------------------------------
st.divider()
st.header("2. Select documents for chat")

selected_document_id = None
selected_label = None
selected_type = None
searchable_documents = []  # defined up front so later checks never fail

try:
    documents_response = api_get("/documents")

    if documents_response.ok:
        documents = documents_response.json().get("documents", [])

        # PDF/CSV have Qdrant chunks; images are answered with Gemini vision.
        searchable_documents = [
            doc for doc in documents if doc.get("file_type") in ("pdf", "csv")
        ]
        image_documents = [
            doc for doc in documents if doc.get("file_type") == "image"
        ]

        # label -> (document_id, file_type)
        document_options = {"All documents (PDF + CSV)": ("__all__", "multiple")}

        for doc in searchable_documents + image_documents:
            icon = FILE_ICONS.get(doc["file_type"], "📁")
            label = f"{icon} {doc['filename']} ({doc['document_id'][:8]})"
            document_options[label] = (doc["document_id"], doc["file_type"])

        option_ids = [value[0] for value in document_options.values()]
        default_index = (
            option_ids.index(st.session_state.last_uploaded_id)
            if st.session_state.last_uploaded_id in option_ids
            else 0
        )

        selected_label = st.selectbox(
            "Choose a document",
            options=list(document_options.keys()),
            index=default_index,
            help=(
                "Choose one PDF, CSV or image, or search across all "
                "indexed PDF and CSV documents."
            ),
        )

        selected_document_id, selected_type = document_options[selected_label]

    else:
        display_api_error(documents_response)

except requests.RequestException as exc:
    display_connection_error(exc)

# --------------------------------------------------
# 3. Chat tabs
# --------------------------------------------------
st.divider()
st.header("3. Chat")

file_tab, general_tab = st.tabs(["💬 Chat with File", "✨ General Gemini Chat"])

# --------------------------------------------------
# File question-answering chat
# --------------------------------------------------
with file_tab:
    if not selected_document_id:
        st.info("Please select a document to get started.")

    elif selected_document_id == "__all__" and not searchable_documents:
        st.info(
            "No searchable PDF or CSV documents found. "
            "Please upload a file first."
        )

    else:
        if selected_document_id == "__all__":
            st.caption("Chatting with all indexed PDF and CSV documents.")
            placeholder = "Ask a question about your documents..."
        elif selected_type == "image":
            st.caption(f"Selected image: {selected_label}")
            placeholder = "Ask a question about this image..."
        else:
            st.caption(f"Selected document: {selected_label}")
            placeholder = "Ask a question about this document..."

        for message in st.session_state.file_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant":
                    render_assistant_details(message)

        question = st.chat_input(placeholder, key="file_chat_input")

        if question:
            st.session_state.file_messages.append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Analyzing and generating an answer..."):
                        response = api_post_json(
                            "/ask-file",
                            {
                                "question": question,
                                "top_k": top_k,
                                "document_id": selected_document_id,
                            },
                        )

                    if response.ok:
                        handle_chat_response(response, "file")
                    else:
                        display_api_error(response)

                except requests.Timeout:
                    st.error("The question timed out. Check the FastAPI logs.")

                except requests.RequestException as exc:
                    display_connection_error(exc)

# --------------------------------------------------
# General Gemini chat
# --------------------------------------------------
with general_tab:
    st.caption("This chat does not use your uploaded files as context.")

    for message in st.session_state.general_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_assistant_details(message)

    general_question = st.chat_input(
        "Ask Gemini anything...",
        key="general_chat_input",
    )

    if general_question:
        st.session_state.general_messages.append(
            {"role": "user", "content": general_question}
        )

        with st.chat_message("user"):
            st.markdown(general_question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Gemini is generating an answer..."):
                    response = api_post_json("/ask", {"question": general_question})

                if response.ok:
                    handle_chat_response(response, "general")
                else:
                    display_api_error(response)

            except requests.Timeout:
                st.error("The request timed out. Check FastAPI logs.")

            except requests.RequestException as exc:
                display_connection_error(exc)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "LLMOps learning project • FastAPI • Streamlit • "
    "Google Vertex AI • Gemini • Qdrant • RAG"
)