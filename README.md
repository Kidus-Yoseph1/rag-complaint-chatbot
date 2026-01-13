#  CrediTrust: AI-Powered Complaint Resolver

CrediTrust is a **Retrieval-Augmented Generation (RAG)** application designed to help financial analysts resolve customer complaints using historical policy data. It uses **FAISS** for lightning-fast similarity search and **Groq (Llama 3.3)** for generating professional resolutions.

##  Quick Start Guide

### 1. Prerequisites

Ensure you have a virtual environment with Python 3.9+ installed and a Groq API Key.

```bash
pip install -r requirements.txt

```

### 2. Step 1: Generate the Vector Store

The Streamlit app requires a pre-built index to function. You **must** run the processing notebook first to sample the data and create the embeddings.

1. Open `Sampling_and_Embedding.ipynb`.
2. Run all cells.
3. Run src/vetore_store.py
4. Verify that a folder named `vector_store/` has been created containing:
* `complaints_index.faiss`
* `metadata.pkl`



### 3. Step 2: Configure Secrets

Create a `.streamlit/secrets.toml` file in the root directory and add your Groq API key:

```toml
GROQ_API_KEY = "your_groq_api_key_here"

```

### 4. Step 3: Launch the App

Run the following command in your terminal:

```bash
streamlit run app.py

```

---

##  How it Works

| Component | Technology | Description |
| --- | --- | --- |
| **Vector Database** | FAISS | Stores 15,000+ complaint chunks for semantic retrieval. |
| **Embeddings** | all-MiniLM-L6-v2 | Converts text into 384-dimensional vectors locally on CPU. |
| **LLM Engine** | Groq (Llama 3.3 70B) | Generates professional, data-driven resolutions. |
| **Interface** | Streamlit | A user-friendly dashboard for financial specialists. |

---

##  Project Structure

* `Sampling_and_Embedding.ipynb`: Data cleaning, stratified sampling, and FAISS indexing.
* `app.py`: The Streamlit frontend logic.
* `src/RAG.py`: The RAG core that handles retrieval and LLM generation.
* `vector_store/`: (Generated) Contains the searchable vector index and metadata.
* `Data/`: Folder containing the raw and filtered CSV datasets.


