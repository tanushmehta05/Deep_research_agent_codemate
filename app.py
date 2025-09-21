import os
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import pipeline
import pdfplumber
from fpdf import FPDF

# ------------------------------
# Paths / names for offline models
# ------------------------------
EMBEDDER_MODEL_PATH = "./models/embedder/models--sentence-transformers--all-MiniLM-L6-v2"
RERANKER_MODEL_PATH = "./models/encoder/models--cross-encoder--ms-marco-MiniLM-L-6-v2"
SUMMARIZER_MODEL_PATH = "./models/summarizer/models--google--flan-t5-small"
DOCS_PATH = "./docs"  # folder containing PDFs

# ------------------------------
# Load models (with caching)
# ------------------------------
@st.cache_resource
def load_models():
    st.info("Loading models...")
    embedder = SentenceTransformer(EMBEDDER_MODEL_PATH, local_files_only=True)
    reranker = CrossEncoder(RERANKER_MODEL_PATH, local_files_only=True)
    summarizer = pipeline(
        "text2text-generation",
        model=SUMMARIZER_MODEL_PATH,
        tokenizer=SUMMARIZER_MODEL_PATH,
        local_files_only=True
    )
    st.success("Models loaded successfully!")
    return embedder, reranker, summarizer

# ------------------------------
# Load PDFs
# ------------------------------
def load_pdfs(pdf_folder):
    docs = []
    doc_names = []
    for file_name in os.listdir(pdf_folder):
        if file_name.lower().endswith(".pdf"):
            with pdfplumber.open(os.path.join(pdf_folder, file_name)) as pdf:
                text = "".join([page.extract_text() + "\n" for page in pdf.pages])
                docs.append(text.strip())
                doc_names.append(file_name)
    return docs, doc_names

# ------------------------------
# Chunk documents
# ------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ------------------------------
# Save summary as PDF
# ------------------------------
def save_summary_pdf(summary, file_name="summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, summary)
    pdf.output(file_name)
    return file_name

# ------------------------------
# Main App
# ------------------------------
def main():
    st.title("🧠 Deep Researcher Agent (Offline Prototype)")

    # Load models
    embedder, reranker, summarizer = load_models()

    # Load documents
    documents, doc_names = load_pdfs(DOCS_PATH)
    if not documents:
        st.warning("No PDF documents found in docs folder!")
        return

    # Chunk documents
    chunked_docs = []
    chunked_names = []
    for doc, name in zip(documents, doc_names):
        chunks = chunk_text(doc)
        chunked_docs.extend(chunks)
        chunked_names.extend([name]*len(chunks))

    st.write(f"Loaded {len(documents)} document(s), split into {len(chunked_docs)} chunks.")

    # Initialize session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # User query loop
    while True:
        query_key = f"query_{len(st.session_state.conversation_history)}"
        query = st.text_input("Enter your research question:", key=query_key)
        if not query:
            break

        # Include previous conversation
        full_query = query + " " + " ".join(st.session_state.conversation_history)

        # Embed and rank
        embeddings = embedder.encode(chunked_docs, convert_to_tensor=True)
        query_emb = embedder.encode(full_query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_emb, embeddings)[0]
        top_k = min(5, len(chunked_docs))
        top_indices = cosine_scores.topk(top_k).indices
        top_chunks = [chunked_docs[i] for i in top_indices]
        top_names = [chunked_names[i] for i in top_indices]

        # Rerank
        pairs = [[full_query, c] for c in top_chunks]
        rerank_scores = reranker.predict(pairs)
        rerank_sorted = sorted(zip(top_chunks, top_names, rerank_scores), key=lambda x: x[2], reverse=True)

        st.subheader("Top Document Chunks:")
        for i, (chunk, name, score) in enumerate(rerank_sorted, 1):
            st.markdown(f"**{i}. {name}** (Cosine: {cosine_scores[top_indices[i-1]]:.3f}, Rerank: {score:.3f})")
            st.write(chunk[:500] + "...\n")

        # Summarize top chunk with citation
        summary_input = rerank_sorted[0][0]
        summary_output = summarizer(summary_input, max_new_tokens=200)
        summary_text = summary_output[0]['generated_text']
        citation = f"Source: {rerank_sorted[0][1]}"
        output_text = summary_text + "\n\n" + citation

        st.subheader("Summary of Top Chunk:")
        st.write(output_text)

        # Add to conversation history
        st.session_state.conversation_history.append(query)

    # Export summary as PDF
    if st.button("Export Summary as PDF"):
        pdf_file = save_summary_pdf(output_text)
        st.success(f"Summary saved to {pdf_file}")

if __name__ == "__main__":
    main()
