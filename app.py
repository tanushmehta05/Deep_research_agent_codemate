import os
import glob
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from PyPDF2 import PdfReader

# ----------------------------
# Paths to your local snapshot models
# ----------------------------
EMBEDDER_MODEL_PATH = r"C:\Users\ASUS\Desktop\CodeMate\models\embedder\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
RERANKER_MODEL_PATH = r"C:\Users\ASUS\Desktop\CodeMate\models\encoder\models--cross-encoder--ms-marco-MiniLM-L-6-v2\snapshots\c5ee24cb16019beea0893ab7796b1df96625c6b8"
SUMMARIZER_MODEL_PATH = r"C:\Users\ASUS\Desktop\CodeMate\models\summarizer\models--google--flan-t5-small\snapshots\0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"

# Folder containing your PDF documents
DOCS_PATH = r"C:\Users\ASUS\Desktop\CodeMate\docs"

# ----------------------------
# Load models with caching
# ----------------------------
@st.cache_resource
def load_models():
    st.info("Loading models...")
    
    embedder = SentenceTransformer(EMBEDDER_MODEL_PATH, local_files_only=True)
    reranker = CrossEncoder(RERANKER_MODEL_PATH, local_files_only=True)
    summarizer = pipeline(
        "text2text-generation",
        model=SUMMARIZER_MODEL_PATH,
        tokenizer=SUMMARIZER_MODEL_PATH,
        device=-1,  # CPU; use 0 for GPU
        local_files_only=True
    )
    
    st.success("Models loaded successfully!")
    return embedder, reranker, summarizer

# ----------------------------
# Load PDFs from folder
# ----------------------------
def load_pdfs(folder_path):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    documents = []
    doc_names = []

    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        documents.append(text)
        doc_names.append(os.path.basename(pdf_file))
    
    return documents, doc_names

# ----------------------------
# Main Streamlit app
# ----------------------------
def main():
    st.title("🧠 Deep Researcher Agent (Offline Prototype)")

    # Load models
    embedder, reranker, summarizer = load_models()

    # Load documents
    documents, doc_names = load_pdfs(DOCS_PATH)
    st.info(f"Loaded {len(documents)} documents.")

    # User query input
    query = st.text_input("Enter your research query:")
    
    if query and documents:
        # Embed documents and query
        doc_embeddings = embedder.encode(documents, convert_to_tensor=True)
        query_embedding = embedder.encode([query], convert_to_tensor=True)

        # Compute similarity scores (dot product)
        import torch
        scores = torch.matmul(query_embedding, doc_embeddings.T).cpu().numpy()[0]

        # Select top-k documents (top 5)
        top_k = 5
        top_indices = scores.argsort()[-top_k:][::-1]
        top_docs = [documents[i] for i in top_indices]
        top_doc_names = [doc_names[i] for i in top_indices]

        # Optional reranker
        reranker_scores = reranker.predict([(query, doc) for doc in top_docs])
        reranked_indices = reranker_scores.argsort()[::-1]
        top_docs = [top_docs[i] for i in reranked_indices]
        top_doc_names = [top_doc_names[i] for i in reranked_indices]

        st.write("### Top relevant documents:")
        for name, doc in zip(top_doc_names, top_docs):
            with st.expander(name):
                st.write(doc[:1000] + "...")  # Show first 1000 chars

        # Summarize the top document
        st.write("### Summary of the most relevant document:")
        summary_text = summarizer(top_docs[0], max_length=200, do_sample=False)[0]['generated_text']
        st.write(summary_text)

if __name__ == "__main__":
    main()
