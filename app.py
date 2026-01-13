import os
import streamlit as st
from src.RAG import ComplaintRAGEngine

st.set_page_config(page_title="CrediTrust Resolver", layout="wide")

# Use Groq API Key
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def init_engine():
    return ComplaintRAGEngine(api_key=GROQ_API_KEY)

engine = init_engine()

st.title("⚖️ CrediTrust Complaint Resolution Engine")
st.markdown("---")

# Main UI layout
query = st.text_area("Input Customer Narrative:", placeholder="Enter the full complaint text here...", height=200)

if st.button("Generate Resolution", type="primary"):
    if not query:
        st.warning("Please enter a complaint first.")
    else:
        with st.spinner("Retrieving policy context and drafting response..."):
            try:
                context = engine.get_context(query)
                answer = engine.generate_response(query, context)
                
                # Use a container for better visual separation
                with st.container():
                    st.markdown(answer)
                
                with st.expander("📚 View Sourced Policy Segments"):
                    st.info(context)
                    
            except Exception as e:
                st.error(f"Operational Error: {e}")

if st.sidebar.button("Reset Session"):
    st.rerun()
