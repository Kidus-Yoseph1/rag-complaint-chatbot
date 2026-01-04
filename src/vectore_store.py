import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


df = pd.read_csv('Data/filtered/sampled_complaints.csv')


model = SentenceTransformer('all-MiniLM-L6-v2')
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

all_chunks = []
all_metadata = []

print(f"Starting process for {len(df)} sampled complaints...")

#  Chunking & Metadata Preparation
for _, row in df.iterrows():
    # Split text into chunks
    chunks = splitter.split_text(str(row['cleaned_narrative']))
    
    for chunk in chunks:
        all_chunks.append(chunk)
        # Store metadata to link chunks back to original sources
        all_metadata.append({
            'complaint_id': row['Complaint ID'],
            'product': row['Product'],
            'company': row['Company'],
            'text': chunk  
        })

# Generate Embeddings
print(f"Generating embeddings for {len(all_chunks)} chunks. This may take a moment...")
embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=64)

# Build FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))


os.makedirs('vector_store', exist_ok=True)
faiss.write_index(index, "vector_store/complaints_index.faiss")

with open("../vector_store/metadata.pkl", "wb") as f:
    pickle.dump(all_metadata, f)

print(f"Success! Vector store created with {len(all_chunks)} chunks.")
