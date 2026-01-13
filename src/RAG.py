import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

class ComplaintRAGEngine:
    def __init__(self, api_key, index_path="vector_store/complaints_index.faiss", meta_path="vector_store/metadata.pkl"):
        # Local embedding model (CPU)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Groq Client (Replaces Gemini)
        self.client = Groq(api_key=api_key)
        
        # Load FAISS and Metadata
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def get_context(self, query, k=3):
        query_vector = self.embed_model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        # Extract text from metadata list of dicts
        results = [self.metadata[i]['text'] for i in indices[0]]
        return "\n---\n".join(results)

    def generate_response(self, query, context):
        system_prompt = """
        You are a Senior Financial Dispute Resolution Specialist at CrediTrust.
        Your goal is to provide clear, actionable, and empathetic resolutions based on company policy.

        OUTPUT STRUCTURE:
        ###  Executive Summary
        [A 2-sentence overview of the issue and the proposed outcome]

        ###  Internal Policy Findings
        * [Finding 1 based on context]
        * [Finding 2 based on context]

        ###  Recommended Action Plan
        1. [Step 1 for the agent]
        2. [Step 2 for the agent]

        ---
        **Note:** If the context provided does not contain enough data to resolve the specific issue, 
        explicitly state: "I apologize, but current internal records provide insufficient data for a definitive resolution."
        """

        user_content = f"CONTEXT FROM POLICY RECORDS:\n{context}\n\nCUSTOMER COMPLAINT:\n{query}"
        
        # Groq completion call (using Llama 3.3 70B for high quality)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1024
        )
        return chat_completion.choices[0].message.content
