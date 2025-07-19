import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
from datetime import datetime, timezone
from ncls import NCLS

# --- Constants ---
DIMENSION = 768  # Dimension of embeddings from nomic-embed-text-v1.5
METRIC_TYPE_FAISS = faiss.METRIC_INNER_PRODUCT # Corresponds to COSINE similarity for normalized vectors
BATCH_SIZE = 128
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Nomic embeddings are already normalized to unit length, so METRIC_INNER_PRODUCT is equivalent to cosine similarity.

# --- Initialize Embedding Model ---
try:
    EMBED_MODEL = SentenceTransformer(
        EMBED_MODEL_NAME, device="cuda:1", trust_remote_code=True
    )
    print(f"Successfully loaded sentence transformer model: {EMBED_MODEL_NAME}")
except Exception as e:
    print(f"Failed to load sentence transformer model: {e}")
    exit(1)

# --- Faiss Index Setup (Flat Index for TA-RAG) ---
def parse_iso_time_to_timestamp(time_str):
    """Converts an ISO 8601 time string to a Unix timestamp (float)."""
    if not time_str:
        return None
    try:
        # Handle 'Z' for UTC explicitly for broader compatibility
        if time_str.endswith('Z'):
            dt_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        else:
            dt_obj = datetime.fromisoformat(time_str)
        # Ensure timezone awareness, assuming UTC if naive, then convert to UTC timestamp
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        else:
            dt_obj = dt_obj.astimezone(timezone.utc)
        return dt_obj.timestamp()
    except ValueError as e:
        print(f"Warning: Could not parse time string '{time_str}': {e}")
        return None
    except AttributeError as e:
        print(f"Warning: Could not parse time string '{str(time_str)}': {e}")
        return None

# --- Faiss Index Setup (Flat Index for TA-RAG) ---
def create_faiss_flat_index(dimension, metric_type):
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        index = faiss.IndexFlatIP(dimension)
        print(f"Faiss IndexFlatIP index created with dimension {dimension}.")
    elif metric_type == faiss.METRIC_L2:
        index = faiss.IndexFlatL2(dimension)
        print(f"Faiss IndexFlatL2 index created with dimension {dimension}.")
    else:
        raise ValueError(f"Unsupported metric_type for flat index: {metric_type}.")
    return index


# --- Data Loading, Embedding, and Indexing for TA-RAG (Handles Multiple Intervals) ---
def generate_embedding(texts: list[str]) -> np.ndarray:
    embeddings = EMBED_MODEL.encode(texts, batch_size=BATCH_SIZE, convert_to_tensor=False, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)

def load_embed_and_build_indices_for_tarag(data_source, faiss_index, batch_size=BATCH_SIZE):
    metadata_list = []
    processed_doc_count = 0

    #  Lists to collect data for NCLS construction
    ncls_starts = []
    ncls_ends = []
    ncls_ids = [] #

    # Temporary lists to hold data for batch processing
    current_batch_texts_to_embed = []
    current_batch_metadata_payload = [] # Stores all info needed post-embedding

    if isinstance(data_source, str):
        try:
            with open(data_source, 'r') as f:
                data_iterable = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading data from {data_source}: {e}")
            return None, None, None
    elif isinstance(data_source, list):
        data_iterable = data_source
    else:
        print("Invalid data_source format. Provide a file path or a list of dictionaries.")
        return None, None, None

    fail_chunk_count = 0
    for cur_data in tqdm(data_iterable, desc="Preparing batches for TA-RAG"):
        corpus_uid = cur_data.get('corpus_uid', cur_data.get('uid')) # Handle both uid and corpus_uid
        chunk_number = cur_data.get('chunk_number')
        chunk_text = cur_data.get('chunk_text')

        if not chunk_text:
            print(f"Warning: Skipping item with empty chunk_text for corpus_uid {corpus_uid}, chunk_number {chunk_number}")
            fail_chunk_count += 1
            continue

        # --- Extract and Parse Time Intervals ---
        all_parsed_intervals_for_chunk = [] # List of (start_ts, end_ts) tuples

        # store the estimate_doc_create_date into the metadata
        if 'new_background' in cur_data and cur_data['new_background'] and 'estimate_doc_create_date' in cur_data['new_background'] and isinstance(cur_data['new_background']['estimate_doc_create_date'], str):
            estimate_doc_create_date = cur_data['new_background']['estimate_doc_create_date']
        else:
            estimate_doc_create_date = ""

        # 2. Event-specific time intervals from chunk_time_info
        chunk_time_events = cur_data.get("chunk_time_info", [])
        if not isinstance(chunk_time_events, list):
            fail_chunk_count += 1
            continue

        for event_info in chunk_time_events:
            event_interval_data = event_info.get("event_time_interval", {})
            if event_interval_data is not None and isinstance(event_interval_data, dict):
                event_begin_str = event_interval_data.get("begin")
                event_end_str = event_interval_data.get("end")

                event_begin_ts = parse_iso_time_to_timestamp(event_begin_str)
                event_end_ts = parse_iso_time_to_timestamp(event_end_str)

                if event_begin_ts is not None and event_end_ts is not None:
                    if event_begin_ts <= event_end_ts:
                        all_parsed_intervals_for_chunk.append((int(event_begin_ts), int(event_end_ts)))
                    else:
                        print(f"Warning: Event interval invalid (start > end) for {corpus_uid} c{chunk_number}. Skipping this interval.")
                elif event_begin_str or event_end_str: # if one is present but not the other, or parsing failed
                    print(f"Warning: Incomplete or unparsable event interval for {corpus_uid} c{chunk_number}. Details: begin='{event_begin_str}', end='{event_end_str}'. Skipping this interval.")


        if not all_parsed_intervals_for_chunk:
            # Try to use document time
            try:
                background_time_info = cur_data.get("new_background", {})
                # The example shows new_background.begin, not new_background.time_interval.begin
                bg_begin_str = background_time_info.get("begin")
                bg_end_str = background_time_info.get("end")
                
                # Adjusting based on your Milvus code's access:
                # It seems your Milvus code used `cur_data["new_background"]["time_interval"]["begin"]`
                # but your example data shows `cur_data["new_background"]["begin"]`.
                # I'll use the latter based on the example data structure. If "time_interval" is an additional nesting, adjust accordingly.
                if not (bg_begin_str and bg_end_str) and "time_interval" in background_time_info: # Fallback to Milvus-like structure
                    bg_time_interval_nested = background_time_info.get("time_interval", {})
                    bg_begin_str = bg_time_interval_nested.get("begin")
                    bg_end_str = bg_time_interval_nested.get("end")


                bg_begin_ts = parse_iso_time_to_timestamp(bg_begin_str)
                bg_end_ts = parse_iso_time_to_timestamp(bg_end_str)

                if bg_begin_ts is not None and bg_end_ts is not None:
                    if bg_begin_ts <= bg_end_ts:
                        all_parsed_intervals_for_chunk.append((int(bg_begin_ts), int(bg_end_ts)))
            except Exception as e:
                pass
            
            if not all_parsed_intervals_for_chunk:
                fail_chunk_count += 1
                print(f"Warning: No valid time intervals found for corpus_uid {corpus_uid}, chunk_number {chunk_number}. This chunk will not be temporally indexed.")


            # Decide if you still want to index its vector without temporal linkage,
            # or skip it entirely. For TA-RAG, skipping temporal indexing is significant.
            # For now, we'll still embed it and add to metadata, but it won't be found by time filters.

        current_batch_texts_to_embed.append("search_document: " + chunk_text)
        current_batch_metadata_payload.append({
            "corpus_uid": corpus_uid,
            "chunk_number": int(chunk_number) if chunk_number is not None else -1,
            "chunk_text": chunk_text,
            "raw_intervals": all_parsed_intervals_for_chunk, # Store for metadata
            "estimate_doc_create_date": estimate_doc_create_date,
            # "original_index" will be added after Faiss insertion
        })
        processed_doc_count +=1

        # --- Process Batch when Full ---
        if len(current_batch_texts_to_embed) >= batch_size:
            embeddings_np = generate_embedding(current_batch_texts_to_embed)
            start_faiss_id = faiss_index.ntotal
            faiss_index.add(embeddings_np)
            # end_faiss_id = faiss_index.ntotal

            for i, meta_payload in enumerate(current_batch_metadata_payload):
                faiss_id = start_faiss_id + i
                meta_payload["original_index"] = faiss_id # This is the Faiss ID
                metadata_list.append(meta_payload)

                for start_ts, end_ts in meta_payload["raw_intervals"]:
                    if start_ts == end_ts:
                        end_ts = end_ts+1

                    ncls_starts.append(start_ts)
                    ncls_ends.append(end_ts) # NCLS is exclusive for end: [start, end)
                    ncls_ids.append(faiss_id)
            
            current_batch_texts_to_embed = []
            current_batch_metadata_payload = []

    # --- Process any Remaining Data in the Last Batch ---
    if current_batch_texts_to_embed:
        embeddings_np = generate_embedding(current_batch_texts_to_embed)
        start_faiss_id = faiss_index.ntotal
        faiss_index.add(embeddings_np)

        for i, meta_payload in enumerate(current_batch_metadata_payload):
            faiss_id = start_faiss_id + i
            meta_payload["original_index"] = faiss_id
            metadata_list.append(meta_payload)

            for start_ts, end_ts in meta_payload["raw_intervals"]:
                if start_ts == end_ts:
                    end_ts = end_ts+1
                ncls_starts.append(start_ts)
                ncls_ends.append(end_ts)
                ncls_ids.append(faiss_id)

    # --- Build NCLS Index ---
    ncls_index = None
    if ncls_starts and ncls_ends and ncls_ids:
        print(f"Preparing to build NCLS index with {len(ncls_starts)} intervals.")
        try:
            starts_np = np.array(ncls_starts, dtype=np.float64) # NCLS handles float64 well
            ends_np = np.array(ncls_ends, dtype=np.float64)
            ids_np = np.array(ncls_ids, dtype=np.int64) # Assuming Faiss IDs are integers

            # NCLS requires starts to be sorted.
            # We need to sort all three arrays based on the start times.
            sort_indices = np.argsort(starts_np)
            starts_np_sorted = starts_np[sort_indices]
            ends_np_sorted = ends_np[sort_indices]
            ids_np_sorted = ids_np[sort_indices]


            if len(starts_np_sorted) > 0 : # Ensure there's data to build from
                 ncls_index = NCLS(starts_np_sorted, ends_np_sorted, ids_np_sorted)
                 print(f"NCLS index built successfully with {len(starts_np_sorted)} intervals.")
            else:
                print("No valid intervals to build NCLS index after sorting/filtering.")

        except Exception as e:
            print(f"Error building NCLS index: {e}")
            print("NCLS starts sample:", ncls_starts[:10])
            print("NCLS ends sample:", ncls_ends[:10])
            print("NCLS ids sample:", ncls_ids[:10])
            if 'starts_np_sorted' in locals():
                print("Sorted NCLS starts sample:", starts_np_sorted[:10])
                print("Sorted NCLS ends sample:", ends_np_sorted[:10])
    else:
        print("No intervals collected to build NCLS index.")
            
    print(f"\nData embedding and indexing complete.")
    print(f"Total documents processed: {processed_doc_count}")
    print(f"Total vectors in Faiss index: {faiss_index.ntotal}")
    print(f"Total metadata entries: {len(metadata_list)}")

    print(f"Number of chunk no time intervals: {fail_chunk_count}")

    if faiss_index.ntotal != len(metadata_list):
        print(f"CRITICAL WARNING: Mismatch between Faiss vectors ({faiss_index.ntotal}) and metadata entries ({len(metadata_list)}). Investigate immediately!")
    
    # It's expected that intervals_added_to_tree >= len(metadata_list)
    # if len(metadata_list) > intervals_added_to_tree:
    #     print(f"Warning: Some documents might not have any valid time intervals associated in the IntervalTree.")


    return faiss_index, metadata_list, ncls_index


# --- Main Execution Logic (Example for TA-RAG Indexing) ---
if __name__ == "__main__":
    # 1. Create Faiss Flat Index
    ta_rag_faiss_index = create_faiss_flat_index(DIMENSION, METRIC_TYPE_FAISS)


    # if not data_list: # If file loading failed or file was empty
    json_filepath = "kb_data/corpus_combine.json"
    data_list = []
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
            data_list.extend(data)
    except FileNotFoundError:
        print(f"Error: JSON file '{json_filepath}' not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_filepath}'.")
        exit()
    


    # 4. Populate Faiss index, metadata store, and interval tree
    populated_faiss_index, metadata_store, populated_ncls_index = load_embed_and_build_indices_for_tarag(
        data_list, ta_rag_faiss_index
    )

    if populated_faiss_index and populated_faiss_index.ntotal > 0:
        print(f"\nSuccessfully populated TA-RAG indices.")
        print(f"Total vectors in Faiss: {populated_faiss_index.ntotal}")
        print(f"Total items in metadata: {len(metadata_store)}")

        # To save the index, metadata, and interval tree (optional for interval tree if rebuilt on load):
        TA_RAG_FAISS_INDEX_FILE = "index/ta_rag_flat.faissindex"
        TA_RAG_METADATA_FILE = "index/ta_rag_flat_metadata.json"
 
        try:
            faiss.write_index(populated_faiss_index, TA_RAG_FAISS_INDEX_FILE)
            print(f"TA-RAG Faiss index saved to: {TA_RAG_FAISS_INDEX_FILE}")

            with open(TA_RAG_METADATA_FILE, 'w') as f:
                json.dump(metadata_store, f, indent=2)
            print(f"TA-RAG Metadata (including time intervals) saved to: {TA_RAG_METADATA_FILE}")

        except Exception as e:
            print(f"Error saving TA-RAG index or metadata: {e}")

        print("\n--- TA-RAG Indexing Complete. Retrieval logic will use these components. ---")

    else:
        print("\nTA-RAG indexing failed or produced no results.")