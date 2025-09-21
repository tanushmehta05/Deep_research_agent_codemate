import os
import shutil
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

# ------------------------------
# Paths
# ------------------------------
MODELS_DIR = "./models"
EMBEDDER_DIR = os.path.join(MODELS_DIR, "embedder")
RERANKER_DIR = os.path.join(MODELS_DIR, "reranker")  # manually move after download
SUMMARIZER_DIR = os.path.join(MODELS_DIR, "summarizer")

EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # smaller & Git-friendly
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SUMMARIZER_MODEL = "google/flan-t5-small"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EMBEDDER_DIR, exist_ok=True)
os.makedirs(SUMMARIZER_DIR, exist_ok=True)

# ------------------------------
# Download embedder
# ------------------------------
print("Downloading embedder...")
embedder = SentenceTransformer(EMBEDDER_MODEL, cache_folder=EMBEDDER_DIR)
print("Embedder downloaded to:", EMBEDDER_DIR)

# ------------------------------
# Download reranker
# ------------------------------
print("Downloading reranker... (cannot force folder, will use HuggingFace cache)")
reranker = CrossEncoder(RERANKER_MODEL)  # downloads to default HF cache
print("Reranker downloaded! Please copy its folder from your HF cache to:", RERANKER_DIR)
print("You can find it in ~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2")

# ------------------------------
# Download summarizer
# ------------------------------
print("Downloading summarizer...")
tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL, cache_dir=SUMMARIZER_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL, cache_dir=SUMMARIZER_DIR)
print("Summarizer downloaded to:", SUMMARIZER_DIR)

print("All downloads complete! Models ready in ./models folder.")
