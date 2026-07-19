# Week 01 — 19 June to 25 June 2026

## What I did this week

Set up the complete development environment for FairCollab. 
Created the GitLab repository at git.cs.bham.ac.uk/projects-2025-26/gxm574 
and configured SSH key authentication. Installed Python 3.11.9 alongside 
existing Python versions (FAISS requires 3.11 specifically). Created a 
virtual environment and installed all core libraries: LangChain, 
langchain-google-genai, faiss-cpu, streamlit, python-dotenv, langgraph, 
and google-generativeai. Obtained a Google Gemini API key and stored it 
securely in .env. Set up VS Code with the Python extension.

## Design decisions made

Chose Python 3.11.9 over 3.14 because faiss-cpu requires Python 3.11 
for compatibility. Chose FAISS over alternatives (ChromaDB, Pinecone) 
because it runs entirely locally with no cloud account or subscription 
needed — suitable for a dissertation prototype with a fixed dataset. 
Chose Google Gemini API over OpenAI because the free tier covers both 
chat completions and embeddings under one API key. Stored all secrets 
in .env and added it to .gitignore to prevent accidental exposure.

## Problems encountered

Discovered that Python 3.14 (already installed) is incompatible with 
faiss-cpu. Resolved by installing Python 3.11.9 alongside it and 
creating the virtual environment explicitly with py -3.11 -m venv venv.

## Plan for next week

Design and populate the simulated student dataset 
(students.json and research_students.json). Begin 
building the RAG pipeline (rag_pipeline.py) using 
FAISS and Gemini embeddings.