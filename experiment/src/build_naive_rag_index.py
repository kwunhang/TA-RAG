import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json # Assuming your data is in JSON format as in the example

# --- Constants ---
DIMENSION = 768  # Dimension of embeddings from nomic-embed-text-v1.5
METRIC_TYPE_FAISS = faiss.METRIC_INNER_PRODUCT # Corresponds to COSINE similarity for normalized vectors
                                             # faiss.METRIC_L2 for Euclidean distance
BATCH_SIZE = 128
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# It's good practice to normalize embeddings for COSINE similarity when using INNER_PRODUCT in Faiss
# Nomic embeddings are already normalized to unit length, so METRIC_INNER_PRODUCT is equivalent to cosine similarity.
# If they were not, we would normalize them before adding to Faiss or use faiss.IndexFlatIP then faiss.normalize_L2(embeddings)

# --- Initialize Embedding Model ---
try:
    EMBED_MODEL = SentenceTransformer(
        EMBED_MODEL_NAME, device="cuda:0", trust_remote_code=True
    )
    print(f"Successfully loaded sentence transformer model: {EMBED_MODEL_NAME}")
except Exception as e:
    print(f"Failed to load sentence transformer model: {e}")
    exit(1)

# --- Faiss Index Setup ---
def create_faiss_index(dimension, metric_type):
    """
    Creates a Faiss index. For HNSW, we'll use IndexHNSWFlat.
    HNSW (Hierarchical Navigable Small World) is a graph-based index, good for ANNS.
    """
    # The HNSW index takes the vector dimension and the number of connections per layer (M)
    # M is a crucial parameter for HNSW; typical values are between 4 and 64.
    # efConstruction is another important parameter, influencing build time vs. index quality.
    # efSearch influences search time vs. accuracy.
    
    # For HNSWFlat, the index stores the full vectors.
    index = faiss.IndexHNSWFlat(dimension, 32, metric_type) # 32 is a common value for M
    index.hnsw.efConstruction = 40 # Example value, tune as needed
    index.hnsw.efSearch = 32       # Example value, tune at search time
    
    print(f"Faiss HNSWFlat index created with dimension {dimension} and metric {metric_type}")
    # If using GPU:
    # if faiss.get_num_gpus() > 0:
    #     print(f"Moving Faiss index to GPU.")
    #     res = faiss.StandardGpuResources() # Or other GPU resources
    #     index = faiss.index_cpu_to_gpu(res, 0, index) # Move to GPU 0
    #     print("Faiss index moved to GPU.")
    return index

# --- Data Loading and Embedding Generation ---
def generate_embedding(texts: list[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts using Sentence Transformers.
    Returns a NumPy array of embeddings.
    """
    embeddings = EMBED_MODEL.encode(texts, batch_size=BATCH_SIZE, convert_to_tensor=False, normalize_embeddings=True)
    # Nomic model already provides normalized embeddings suitable for cosine similarity.
    # If not, ensure normalization:
    # embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.asarray(embeddings, dtype=np.float32)

def load_and_embed_faiss(data_source, faiss_index, batch_size=BATCH_SIZE):
    """
    Loads data, generates embeddings, and adds them to the Faiss index.
    It also stores metadata separately as Faiss primarily handles vectors.
    """
    all_embeddings = []
    metadata_list = [] # To store corresponding metadata for each vector

    batch_texts = []
    batch_metadata_items = []

    # Determine if data_source is a file path or a list of dicts
    if isinstance(data_source, str): # Assuming it's a file path
        try:
            with open(data_source, 'r') as f:
                data_iterable = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading data from {data_source}: {e}")
            return None, None # Return None for both index and metadata
    elif isinstance(data_source, list): # Assuming it's already loaded data
        data_iterable = data_source
    else:
        print("Invalid data_source format. Provide a file path or a list of dictionaries.")
        return None, None


    for cur_data in tqdm(data_iterable, desc="Processing chunks and embedding"):
        # corpus_uid = cur_data.get('uid', cur_data.get('corpus_uid')) # Handle both 'uid' and 'corpus_uid'
        corpus_uid = cur_data.get('corpus_uid')
        chunk_number = cur_data.get('chunk_number')
        chunk_text = cur_data.get('chunk_text')

        if not chunk_text:
            print(f"Warning: Skipping item with empty chunk_text for corpus_uid {corpus_uid}, chunk_number {chunk_number}")
            continue

        # For nomic-embed-text-v1.5, it's recommended to prefix task-specific instructions
        # For retrieval, "search_document: " is used for documents.
        batch_texts.append("search_document: " + chunk_text)
        batch_metadata_items.append({
            "corpus_uid": corpus_uid,
            "chunk_number": int(chunk_number) if chunk_number is not None else -1,
            "chunk_text": chunk_text, # Storing original text for reference
        })

        if len(batch_texts) >= batch_size:
            embeddings = generate_embedding(batch_texts)
            if faiss_index.is_trained: # HNSW is trained on-the-fly (or rather, doesn't require separate training step like IVF)
                 faiss_index.add(embeddings)
            else: # Some Faiss indexes require training, HNSWFlat does not.
                 # For HNSW, this 'else' block might not be strictly necessary if 'add' can be called directly.
                 # However, it's good to be explicit that HNSWFlat doesn't need a separate training phase.
                 faiss_index.add(embeddings) # HNSW builds its structure incrementally

            metadata_list.extend(batch_metadata_items)
            batch_texts = []
            batch_metadata_items = []
            print(f"Added a batch of {len(embeddings)} embeddings to Faiss index.")

    # Process any remaining data in the last batch
    if batch_texts:
        embeddings = generate_embedding(batch_texts)
        if faiss_index.is_trained or not faiss_index.is_trained: # HNSWFlat doesn't need explicit training
            faiss_index.add(embeddings)
        metadata_list.extend(batch_metadata_items)
        print(f"Added the final batch of {len(embeddings)} embeddings to Faiss index.")

    print(f"\nData embedding and indexing complete. Total documents indexed: {faiss_index.ntotal}")
    print(f"Total metadata entries: {len(metadata_list)}")
    
    # Sanity check
    if faiss_index.ntotal != len(metadata_list):
        print(f"Warning: Mismatch between number of indexed vectors ({faiss_index.ntotal}) and metadata entries ({len(metadata_list)}).")

    return faiss_index, metadata_list


# --- Main Execution Logic (Example) ---
if __name__ == "__main__":
    # 1. Create Faiss Index
    faiss_index = create_faiss_index(DIMENSION, METRIC_TYPE_FAISS)
    json_filepath = "kb_data/corpus_combine.json"
    data_list = []
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
            data_list.extend(data)
    except FileNotFoundError:
        print(f"Error: JSON file '{json_filepath}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_filepath}'.")

    # updated_faiss_index will be the same object as faiss_index but populated
    # metadata will store the text and other info corresponding to each vector's original ID
    populated_faiss_index, metadata_store = load_and_embed_faiss(data_list, faiss_index)

    if populated_faiss_index and populated_faiss_index.ntotal > 0:
        print(f"\nSuccessfully populated Faiss index. Total vectors: {populated_faiss_index.ntotal}")
        
        # To save the index and metadata (optional):
        FAISS_INDEX_FILE = "index/naive_rag_hnsw.faissindex"
        METADATA_FILE = "index/naive_rag_metadata.json"
        
        try:
            faiss.write_index(populated_faiss_index, FAISS_INDEX_FILE)
            print(f"Faiss index saved to: {FAISS_INDEX_FILE}")
            
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata_store, f)
            print(f"Metadata saved to: {METADATA_FILE}")
        except Exception as e:
            print(f"Error saving index or metadata: {e}")

        # --- Placeholder for Retrieval and RAG steps (to be implemented later) ---
        print("\n--- Retrieval and RAG steps would follow here ---")

    else:
        print("\nFailed to populate Faiss index or no data was processed.")