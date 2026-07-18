"""
RAG pipeline for FairCollab: builds a searchable evidence store out of the
raw student contribution logs so the assessment agents can retrieve the
specific records that justify a fairness judgement about one student.
"""

import os

import json

from dotenv import load_dotenv

from langchain_core.documents import Document

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUDENTS_PATH = os.path.join(BASE_DIR, "data", "students.json")

RESEARCH_STUDENTS_PATH = os.path.join(BASE_DIR, "data", "research_students.json")

INDEX_DIR = os.path.join(BASE_DIR, "data", "faircollab_index")

EMBEDDING_MODEL = "models/gemini-embedding-001"

load_dotenv()


def _load_json(path: str) -> dict:
    """Read one contribution-log JSON file and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _contribution_to_document(project: dict, student: dict, contribution: dict) -> Document:
    """Turn a single contribution entry into one embeddable Document."""
    date = contribution.get("date", "unknown date")
    ctype = contribution.get("type", "unknown type")
    description = contribution.get("description", "")

    extra_bits = []
    for key in ("complexity", "status", "role", "recipient"):
        if key in contribution:
            extra_bits.append(f"{key}: {contribution[key]}")
    extra_text = f" ({', '.join(extra_bits)})" if extra_bits else ""

    page_content = (
        f"Student: {student['student_name']} | Project: {project['project_name']} | "
        f"Date: {date} | Type: {ctype} | Description: {description}{extra_text}"
    )

    metadata = {
        "student_name": student["student_name"],
        "student_id": student.get("student_id"),
        "project_id": project.get("project_id"),
        "project_name": project.get("project_name"),
        "record_type": "contribution",
        "contribution_type": ctype,
        "date": date,
    }

    return Document(page_content=page_content, metadata=metadata)


def _feedback_to_document(project: dict, student: dict, feedback_text: str, index: int) -> Document:
    """Turn a single peer-feedback quote into one embeddable Document."""
    page_content = (
        f"Student: {student['student_name']} | Project: {project['project_name']} | "
        f"Peer feedback: {feedback_text}"
    )
    metadata = {
        "student_name": student["student_name"],
        "student_id": student.get("student_id"),
        "project_id": project.get("project_id"),
        "project_name": project.get("project_name"),
        "record_type": "peer_feedback",
        "feedback_index": index,
    }
    return Document(page_content=page_content, metadata=metadata)


def build_documents() -> list:
    """Load both data files and flatten every record into a list of Documents."""
    projects = [_load_json(STUDENTS_PATH), _load_json(RESEARCH_STUDENTS_PATH)]

    documents = []
    for project in projects:
        for student in project["students"]:
            for contribution in student.get("contributions", []):
                documents.append(_contribution_to_document(project, student, contribution))

            for i, feedback_text in enumerate(student.get("peer_feedback", [])):
                documents.append(_feedback_to_document(project, student, feedback_text, i))

    return documents


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Construct the Gemini embeddings client, failing fast if the key is missing."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file before running this script."
        )
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)


def build_and_save_index() -> FAISS:
    """Embed every contribution/feedback record and persist the FAISS index to disk."""
    documents = build_documents()
    embeddings = _get_embeddings()

    vector_store = FAISS.from_documents(documents, embeddings)

    vector_store.save_local(INDEX_DIR)

    return vector_store


def _load_index() -> FAISS:
    """Load a previously saved FAISS index from disk, embeddings client included."""
    embeddings = _get_embeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def retrieve_evidence(student_name: str, k: int = 8) -> list:
    """Return the top-k most relevant contribution/feedback records for one student."""
    vector_store = _load_index()

    results = vector_store.similarity_search(
        student_name, k=k, filter={"student_name": student_name}, fetch_k=200
    )

    if not results:
        results = vector_store.similarity_search(student_name, k=k)

    return results


if __name__ == "__main__":
    print(f"Loading data from {STUDENTS_PATH} and {RESEARCH_STUDENTS_PATH}...")
    docs = build_documents()
    print(f"Built {len(docs)} documents. Embedding and building FAISS index...")
    build_and_save_index()
    print(f"Index saved to {INDEX_DIR}")
