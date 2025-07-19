import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import pandas as pd
import sys
import logging
import os
import time
import concurrent.futures
from datetime import datetime, timezone
import re
# from tools.llm_response import response_rag_time_mc, temporal_process_sentence
from tools.llm_response import LLMClient
# LLM path use
from ncls import NCLS

LLM_URL = os.environ.get("LLM_URL")

client_A_config = {
    "llm_base_url": LLM_URL,
    "llm_api_key": "",
    "model": "meta-llama/Llama-3.3-70B-Instruct"
}

LLM_CLIENT_A = LLMClient(**client_A_config)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DIMENSION = 768
MAX_WORKERS_FOR_LLM_CALLS = 4
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# TA-RAG Specific Files
# Ensure these paths are correct for your environment
# Using relative paths for demonstration, adjust as needed.
TA_RAG_FAISS_INDEX_FILE = "index/ta_rag_flat.faissindex"
TA_RAG_METADATA_FILE = "index/ta_rag_flat_metadata.json"
TEST_DATA_FILE = 'dqabench_MCQA.json'
BATCH_SIZE = 128
# RESULT_DIR = "result/" # Adjusted to a local relative path for portability
TA_OUTPUT_CSV_FILE = "ta_rag_org_70b_batch_faiss_flat_results.csv"
TA_OUTPUT_ACCURACY_TXT_FILE = "ta_rag_org_70b_batch_faiss_flat_accuracy_summary.txt"
RESULT_DIR = os.environ.get("RESULT_DIR", "result_new/")

# --- Initialize Embedding Model ---
try:
    # EMBED_MODEL = SentenceTransformer(
    #     EMBED_MODEL_NAME, device="cuda:1" if faiss.get_num_gpus() > 0 else "cpu", trust_remote_code=True
    # )
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    EMBED_MODEL = SentenceTransformer(
        EMBED_MODEL_NAME, device="cuda:0", trust_remote_code=True
    )
    logging.info(f"Successfully loaded sentence transformer model: {EMBED_MODEL_NAME} on device: cuda:0")
except Exception as e:
    logging.error(f"Failed to load sentence transformer model: {e}")
    sys.exit(1)

def _parse_iso_time_to_timestamp_for_query(time_str: str):
    """Helper to parse ISO time string to Unix timestamp for query intervals."""
    if not time_str: return None
    try:
        dt_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        return dt_obj.timestamp()
    except ValueError:
        logging.error(f"!!!!Invalid time string format: {time_str}. Returning None.")
        return None

def generate_embedding(texts: list[str]) -> np.ndarray:
    embeddings = EMBED_MODEL.encode(texts, batch_size=BATCH_SIZE, convert_to_tensor=False, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)

def rank_candidates_semantically(query_embedding: np.ndarray,
                                 candidate_faiss_ids: set,
                                 faiss_index: faiss.Index,
                                 metadata_store: list,
                                 top_k: int,
                                 perform_full_search_if_no_temporal_candidates: bool = True) -> list:
    """
    Performs semantic search.
    If candidate_faiss_ids is provided, searches within them.
    Otherwise, if perform_full_search_if_no_temporal_candidates is True, searches the whole index.
    Returns a list of metadata items, ordered by semantic similarity.
    """
    semantically_ranked_metadata_items = []

    if candidate_faiss_ids:
        logging.debug(f"Ranking {len(candidate_faiss_ids)} temporal candidates semantically.")
        # Ensure IDs are valid integers and within bounds
        valid_candidate_ids_list = [
            cid for cid in candidate_faiss_ids if 0 <= cid < faiss_index.ntotal
        ]

        if not valid_candidate_ids_list:
            logging.warning("No valid candidate IDs after filtering for Faiss index bounds.")
            return []
        try:
            candidate_vectors = faiss_index.reconstruct_batch(np.array(valid_candidate_ids_list, dtype=np.int64))
        except RuntimeError as e: # More specific Faiss error
            logging.error(f"Faiss runtime error reconstructing vectors: {e}. IDs: {valid_candidate_ids_list[:5]}")
            return []
        except Exception as e:
            logging.error(f"Error reconstructing vectors: {e}. IDs: {valid_candidate_ids_list[:5]}")
            return []


        scores = np.dot(candidate_vectors, query_embedding.ravel()) # Ensure query_embedding is 1D
        
        # Get top-k from these scores
        actual_k = min(top_k, len(scores))
        # argsort sorts ascending, so we take from the end for descending (highest score)
        sorted_score_indices_within_candidates = np.argsort(scores)[::-1][:actual_k]
        
        final_faiss_ids = [valid_candidate_ids_list[i] for i in sorted_score_indices_within_candidates]
        semantically_ranked_metadata_items = [metadata_store[fid] for fid in final_faiss_ids]

    elif perform_full_search_if_no_temporal_candidates:
        logging.debug("No temporal candidates or fallback enabled; performing full semantic search.")
        try:
            logging.error("!!!!!!Performing full search on Faiss index.")
            distances, indices = faiss_index.search(query_embedding, top_k)
            # indices is shape (1, top_k)
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if 0 <= idx < len(metadata_store): # Ensure valid index
                    semantically_ranked_metadata_items.append(metadata_store[idx])
        except Exception as e:
            logging.error(f"Error during full Faiss search: {e}")
            return []
    else:
        logging.debug("No temporal candidates and full search fallback is disabled.")

    return semantically_ranked_metadata_items


def get_repack_sort_key(metadata_item):
    """
    Generates a key for sorting based on the time string.
    Handles None values and parsing errors.
    Returns a tuple (sort_priority, comparable_value).
    - Lower sort_priority comes first.
    - Nones and errors get a higher priority (go last).
    """
    time_str = metadata_item.get("estimate_doc_create_date", "") # Use .get for safety if dict structure varies

    if time_str is None:
        return (1, None)
    try:
        # Parse the ISO format string
        # Handle 'Z' explicitly if fromisoformat doesn't always do it (depends on Python version)
        if time_str.endswith('Z'):
             time_str = time_str[:-1] + '+00:00'

        dt_object = datetime.fromisoformat(time_str)

        # Check if the datetime object is naive (lacks timezone info)
        if dt_object.tzinfo is None or dt_object.tzinfo.utcoffset(dt_object) is None:
            # If naive, assume UTC and make it offset-aware
            dt_object = dt_object.replace(tzinfo=timezone.utc)
            # print(f"Made naive date aware: {time_str} -> {dt_object}") # Optional: for debugging

        return (0, dt_object) 
    except Exception as e:
        return (1, None)

def repack_result(retrieved_metadata_items: list) -> list:
    """
    Sorts a list of retrieved metadata items chronologically based on their representative time.
    """
    # Sort ascending by timestamp (chronological order)
    return sorted(retrieved_metadata_items, key=get_repack_sort_key)


def ta_rag_retrieval(original_query_text: str,
                               faiss_index: faiss.Index,
                               metadata_store: list,
                               ncls_index: NCLS,
                               top_k: int = 10,
                               test_item_data: dict = None) -> tuple:
    """
    Orchestrates the TA-RAG retrieval process.
    Returns: (texts, ids, total_search_time, semantic_search_time, temporal_processing_time, sorted_full_metadata)
    """
    logging.info(f"Starting TA-RAG retrieval for query: '{original_query_text[:50]}...'")
    retrieval_start_time = time.time()

    # 1. Analyze Query for Temporal Aspects
    temporal_analysis_start_time = time.time()
    query_analysis_result = LLM_CLIENT_A.temporal_process_sentence(original_query_text, chance=20)
    rephrased_query = query_analysis_result["rephrased_sentence"]
    query_temporal_list = query_analysis_result["temporal_decomposition"]
    temporal_processing_time = time.time() - temporal_analysis_start_time
    # logging.info(f"Query: '{original_query_text[:50]}...' -> Rephrased: '{rephrased_query[:50]}...', Temporal List: {query_temporal_list}")

    # 2. Filter Candidates with Interval Tree
    time_interval_pairs = [(_parse_iso_time_to_timestamp_for_query(q_interval.get("begin")), _parse_iso_time_to_timestamp_for_query(q_interval.get("end"))) for q_interval in query_temporal_list]

    candidate_faiss_ids = filter_candidates_with_ncls(time_interval_pairs, ncls_index)
    # logging.info(f"Found {len(candidate_faiss_ids)} candidate documents from interval tree.")

    # 3. Embed Query
    query_embed_start_time = time.time()
    if time_interval_pairs:  # Check if temporal_list is not empty
        hypo_querys = []
        min_date_dt = datetime(2012, 1, 1, tzinfo=timezone.utc) # Aware datetime
        for begin_ts, end_ts in time_interval_pairs:
            if begin_ts is None:
                begin_ts = 0
            if end_ts is None:
                # End_ts be Today timestamp if none
                end_ts = int(time.time())
            # Convert timestamps back to datetime objects for iteration logic
            # Assuming UTC for timestamps if no other info.
            begin_date_dt = datetime.fromtimestamp(begin_ts, tz=timezone.utc)
            end_date_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

            # Now use begin_date_dt and end_date_dt in your existing loop
            # (Logic for min_date, current_date, etc. using these datetime objects)
            # Example for min_date:
            begin_date_dt = max(begin_date_dt, min_date_dt)

            effective_start_dt = datetime(begin_date_dt.year, begin_date_dt.month, 1, tzinfo=begin_date_dt.tzinfo)
            if effective_start_dt <= end_date_dt:
                monthly_dates = pd.date_range(
                    start=effective_start_dt,
                    end=end_date_dt,
                    freq='MS', # Use 'ME' for Month End if that's preferred by logic
                    tz=effective_start_dt.tzinfo # Pass timezone info
                )
                for month_date in monthly_dates:
                    # month_date is a pandas Timestamp object, which has strftime
                    month_year_str = month_date.strftime("%B %Y")
                    query = f"search_query: In {month_year_str}, {rephrased_query}"
                    hypo_querys.append(query)
    else:
        hypo_querys = [f"search_query: {rephrased_query}"]
    hypo_embeddings = generate_embedding(hypo_querys)
    query_mean_embedding = np.mean(hypo_embeddings, axis=0)
    query_embed_time = time.time() - query_embed_start_time

    perform_full_search = not query_temporal_list # Fallback if query had no discernible time
    if perform_full_search:
        logging.error("!!!!!! Query had no discernible time; performing full semantic search.")
    

    semantic_rank_start_time = time.time()
    semantically_ranked_metadata = rank_candidates_semantically(
        query_mean_embedding,
        candidate_faiss_ids,
        faiss_index,
        metadata_store,
        top_k,
        perform_full_search_if_no_temporal_candidates=perform_full_search
    )
    # Semantic search time includes embedding time as it's part of the retrieval step
    semantic_search_time = (time.time() - semantic_rank_start_time) + query_embed_time
    # logging.info(f"Retrieved {len(semantically_ranked_metadata)} items after semantic ranking.")

    # 5. Sort Final Results Chronologically (if needed, or keep semantic order)
    # Your previous logic implied sorting for final presentation.
    
    total_retrieval_time = time.time() - retrieval_start_time
    
    return (
        semantically_ranked_metadata,
        total_retrieval_time,
        semantic_search_time,
        temporal_processing_time,
        query_embed_time
    )

# --- Artifact Loading ---
def load_ta_rag_artifacts(index_path: str, metadata_path: str) -> tuple:
    # Load Faiss Index
    if not os.path.exists(index_path):
        logging.error(f"TA-RAG Faiss index file not found: {index_path}")
        return None, None, None
    try:
        index = faiss.read_index(index_path)
        logging.info(f"TA-RAG Faiss index loaded from {index_path}. Total vectors: {index.ntotal}")
    except Exception as e:
        logging.error(f"Failed to load TA-RAG Faiss index: {e}")
        return None, None, None

    # Load Metadata
    if not os.path.exists(metadata_path):
        logging.error(f"TA-RAG Metadata file not found: {metadata_path}")
        return None, None, None
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info(f"TA-RAG Metadata loaded from {metadata_path}. Total entries: {len(metadata)}")
    except Exception as e:
        logging.error(f"Failed to load TA-RAG metadata: {e}")
        return None, None, None

    if index.ntotal == 0:
        logging.warning("Faiss index is empty. Proceeding with empty index and tree.")
        empty_starts = np.array([], dtype=np.float64)
        empty_ends = np.array([], dtype=np.float64)
        empty_ids = np.array([], dtype=np.int64)
        ncls_index = NCLS(empty_starts, empty_ends, empty_ids)
        return index, metadata, ncls_index

    if index.ntotal != len(metadata):
        # This might be acceptable if some metadata items were not indexed for some reason,
        # but original_index in metadata should still map correctly to Faiss IDs.
        logging.warning(f"Potential Mismatch: Faiss index has {index.ntotal} vectors, metadata has {len(metadata)} entries.")

    # 3. Prepare data for NCLS
    all_starts_list = []
    all_ends_list = []
    all_faiss_ids_list = []
    processed_intervals_count = 0
    skipped_meta_items_for_ncls = 0

    logging.info("Preparing data for NCLS construction...")
    for item_meta in tqdm(metadata, desc="Processing metadata for NCLS"):
        faiss_id = item_meta.get("original_index")
        
        if not isinstance(faiss_id, int) or faiss_id < 0 or faiss_id >= index.ntotal:
            logging.warning(f"Skipping metadata item (corpus_uid: {item_meta.get('corpus_uid')}) for NCLS due to invalid 'original_index': {faiss_id}.")
            skipped_meta_items_for_ncls += 1
            continue
        
        raw_intervals_data = item_meta.get("raw_intervals", [])
        if not isinstance(raw_intervals_data, list):
            logging.warning(f"Malformed raw_intervals for faiss_id {faiss_id} (NCLS). Expected list, got {type(raw_intervals_data)}. Skipping.")
            skipped_meta_items_for_ncls +=1 # Count the item once even if it has multiple bad intervals
            continue

        item_has_valid_interval_for_ncls = False

        merged_interval_data = merge_document_intervals(raw_intervals_data)

        for interval_timestamps in merged_interval_data:
            # Optimize: simplify the tree with merge the overlap intervals
            start_ts, end_ts = interval_timestamps
            if isinstance(start_ts, (int, float)) and isinstance(end_ts, (int, float)) and \
                np.isfinite(start_ts) and np.isfinite(end_ts): # Check for NaN/Inf
                if int(start_ts) < int(end_ts) : # Ensure start is less than end+epsilon
                    all_starts_list.append(int(start_ts))
                    all_ends_list.append(int(end_ts))
                    all_faiss_ids_list.append(int(faiss_id))
                    processed_intervals_count += 1
                    item_has_valid_interval_for_ncls = True
                elif int(start_ts) == int(end_ts): # Handle exact point if NCLS requires start < end
                    # For point events, make end slightly larger if NCLS doesn't like start == end
                    all_starts_list.append(int(start_ts))
                    all_ends_list.append(int(end_ts) + 1) 
                    all_faiss_ids_list.append(int(faiss_id))
                    processed_intervals_count +=1
                    item_has_valid_interval_for_ncls = True
                else:
                    logging.debug(f"Skipping interval for NCLS (start_ts >= end_ts+epsilon or other issue): [{start_ts}, {end_ts}] for faiss_id {faiss_id}")
            else:
                logging.debug(f"Skipping non-finite or invalid type interval timestamps for NCLS: [{start_ts}, {end_ts}] for faiss_id {faiss_id}")
        if not item_has_valid_interval_for_ncls and raw_intervals_data: # Item had intervals but none were valid
            skipped_meta_items_for_ncls +=1


    if not all_starts_list: # No valid intervals found to build NCLS
        logging.warning("No valid intervals found in metadata to build NCLS index. NCLS will be empty.")
        # Create empty NCLS
        empty_starts = np.array([], dtype=np.int64)
        empty_ends = np.array([], dtype=np.int64)
        empty_ids = np.array([], dtype=np.int64)
        ncls_index = NCLS(empty_starts, empty_ends, empty_ids)
        return index, metadata, ncls_index

    # 4. Create NCLS Index
    starts_np = np.array(all_starts_list, dtype=np.int64)
    ends_np = np.array(all_ends_list, dtype=np.int64)
    faiss_ids_np = np.array(all_faiss_ids_list, dtype=np.int64) # These are the 'values' for NCLS

    logging.info(f"Constructing NCLS index with {len(starts_np)} intervals. Skipped metadata items for NCLS: {skipped_meta_items_for_ncls}")
    ncls_construction_start_time = time.time()
    ncls_index = NCLS(starts_np, ends_np, faiss_ids_np)
    ncls_construction_time = time.time() - ncls_construction_start_time
    logging.info(f"NCLS index constructed in {ncls_construction_time:.4f} seconds.")
    
    return index, metadata, ncls_index 



def filter_candidates_with_ncls(time_interval_pairs: list, ncls_index: NCLS) -> set:
    """
    Filters Faiss IDs using the NCLS index based on query's temporal list.
    """
    candidate_faiss_ids = set()
    if not time_interval_pairs:
        logging.debug("filter_candidates_with_ncls called with empty time_interval_pairs.")
        return candidate_faiss_ids

    logging.debug(f"NCLS: Processing {len(time_interval_pairs)} temporal dicts from query.")

    for q_begin_ts_float, q_end_ts_float in time_interval_pairs:        
        if q_begin_ts_float is None or q_end_ts_float is None:
            logging.debug(f"NCLS: Parsed timestamps are None.")
            continue

        # Adjust end timestamp for partial query strings (YYYY, YYYY-MM)
        # This adjustment should make q_end_ts_float the actual end of the period.
        final_q_begin_ts, final_q_end_ts = int(q_begin_ts_float), int(q_end_ts_float) # Start with parsed

        if not (np.isfinite(final_q_begin_ts) and np.isfinite(final_q_end_ts)):
            logging.warning(f"NCLS: Non-finite query timestamps after adjustment. Begin: {final_q_begin_ts}, End: {final_q_end_ts}. SKIPPING.")
            continue
        if final_q_begin_ts > final_q_end_ts:
            logging.warning(f"NCLS: Query interval begin > end after adjustment: [{final_q_begin_ts}, {final_q_end_ts}]. SKIPPING.")
            continue
        
        logging.debug(f"NCLS: Querying with floats: begin=[{final_q_begin_ts}], final_q_end_ts=[{final_q_end_ts}]")
        
        try:
            if final_q_begin_ts >= final_q_end_ts: # Should not happen if final_q_end_ts was valid
                logging.warning(f"NCLS: Query start >= query end for NCLS call [{final_q_begin_ts}, {final_q_end_ts}]. Skipping.")
                continue

            iter_result = ncls_index.find_overlap(final_q_begin_ts, final_q_end_ts)

            for _,_, target_faiss_id in iter_result:
                candidate_faiss_ids.add(target_faiss_id) # .values is the ids_np array

        except Exception as e:
            logging.error(f"NCLS: Error during NCLS find_overlap or processing for interval [{final_q_begin_ts}, {final_q_end_ts}]: {e}", exc_info=True)
            
    logging.debug(f"NCLS: filter_candidates finished. Total unique candidates: {len(candidate_faiss_ids)}")
    return candidate_faiss_ids


# --- Context Construction & LLM Call (Mostly from previous script) ---
def construct_ta_rag_context(sorted_metadata_items: list) -> str:
    context = ""
    for i, meta_item in enumerate(sorted_metadata_items):
        context += f"Snippet {i+1}:\n"
        est_doc_create_date = meta_item.get("estimate_doc_create_date", "")
        if est_doc_create_date:
            try:
                dt_obj = datetime.fromisoformat(est_doc_create_date.replace('Z', '+00:00'))
                chunk_time_readable = dt_obj.strftime('%Y-%m-%d')
                context += f"(Estimate publish date: {chunk_time_readable})\n"
            except Exception as e:
                pass
        context += f"{meta_item['chunk_text']}\n\n"
    return context.strip()

def load_test_data(json_file: str) -> list:
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading test data from {json_file}: {e}")
        return [] # Return empty list on error

def merge_document_intervals(intervals: list[tuple[int, int]]) -> list[tuple[float, float]]:
    if not intervals:
        return intervals
    
    sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    
    merged = []
    if not sorted_intervals: # Should be caught by the first 'if not intervals'
        exit() # it should not happen
        
    current_start, current_end = sorted_intervals[0]
    
    for i in range(1, len(sorted_intervals)):
        next_start, next_end = sorted_intervals[i]
        if next_start <= current_end: 
            current_end = max(current_end, next_end) # Extend the current interval
        else:
            merged.append((current_start, current_end)) # Finalize the current merged interval
            current_start, current_end = next_start, next_end # Start a new one
            
    merged.append((current_start, current_end)) # Add the last processed interval
    return merged

def timed_llm_call_wrapper(llm_call_args: tuple) -> tuple:
    question_for_llm, context_list, k_value, item_id_for_log = llm_call_args
    gen_start_time = time.time()
    llm_response_raw = LLM_CLIENT_A.response_rag_time_mc(question_for_llm, context_list)
    generation_time = time.time() - gen_start_time
    return k_value, llm_response_raw, generation_time

# --- Main Evaluation Loop (Adapted for TA-RAG) ---
def evaluate_and_store_ta_rag(test_data_file: str,
                              faiss_index: faiss.Index,
                              metadata_store: list,
                              ncls_index: NCLS,
                              output_accuracy_txt_file: str):
    test_data = load_test_data(test_data_file)
    if not test_data:
        return pd.DataFrame()

    results_list = []
    top_k_values_for_llm = [5, 10, 20, 50]
    max_k_retrieval = 50
    
    accuracies = {k: 0 for k in top_k_values_for_llm}
    total_mcq_questions_processed = 0
    
    # Determine number of workers based on CPU cores if MAX_WORKERS_FOR_LLM_CALLS is too high or not set
    num_workers = min(MAX_WORKERS_FOR_LLM_CALLS, os.cpu_count() or 1)


    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for item_index, item_data in enumerate(tqdm(test_data, desc="Evaluating TA-RAG performance")):
            org_question = item_data.get("question", "")
            ground_truth_answer = str(item_data.get("correct_answer_key", "")).lower() # Ensure GT is lower string
            item_id = item_data.get("id", f"item_{item_index}")
            question_type_id = item_data.get("query_type_info", {}).get("id", "N/A")

            current_result_item = {
                "question_id": item_id,
                "question": org_question, # Will be updated if MCQ
                "ground_truth_answer": ground_truth_answer,
                "total_retrieval_time_seconds": 0.0,
                "semantic_search_time_seconds": 0.0,
                "temporal_processing_time_seconds": 0.0,
                "query_embedding_time_seconds": 0.0,
                "repack_time_seconds": 0.0,
                "top_50_retrieved_ids": [],
                "question_type_id": question_type_id
            }
            for k_val in top_k_values_for_llm: # Initialize fields
                current_result_item[f"llm_response_k{k_val}"] = "N/A"
                current_result_item[f"generation_time_k{k_val}_seconds"] = 0.0

            # Perform one TA-RAG search for the maximum k needed for LLM context or ID logging
            # `ta_rag_retrieval` is now the main entry point for retrieval
            # retrieved_texts, retrieved_ids, total_ret_time, sem_search_time, temp_proc_time, query_embed_time = ta_rag_retrieval(
            retrieval_metadata, total_ret_time, sem_search_time, temp_proc_time, query_embed_time = ta_rag_retrieval(
                org_question, faiss_index, metadata_store, ncls_index, top_k=max_k_retrieval, test_item_data=item_data
            )
            
            # # Repack the metadata for final output
            # repack_start_time = time.time()
            # sorted_retrieved_metadata = repack_result(retrieval_metadata)
            # repack_time = time.time() - repack_start_time

            # retrieved_texts = [item.get("chunk_text", "") for item in sorted_retrieved_metadata]
            retrieved_ids = [f"{item.get('corpus_uid', 'N/A')}" for item in retrieval_metadata]

            current_result_item["total_retrieval_time_seconds"] = total_ret_time
            current_result_item["semantic_search_time_seconds"] = sem_search_time
            current_result_item["temporal_processing_time_seconds"] = temp_proc_time
            current_result_item["query_embedding_time_seconds"] = query_embed_time
            current_result_item["top_50_retrieved_ids"] = retrieved_ids[:max_k_retrieval]

            if not retrieval_metadata:
                logging.warning(f"No documents retrieved by TA-RAG for Q_id: {item_id}, Question: '{org_question[:50]}...'")
                results_list.append(current_result_item)
                continue
            
            question_for_llm_final = org_question
            is_mcq_for_accuracy = False
            if "choices" in item_data and isinstance(item_data["choices"], dict):
                question_for_llm_final = org_question + "\nPlease select the most accurate description from the options below:\n"
                for key, choice_text in item_data["choices"].items():
                    question_for_llm_final += f"\n{key}) {choice_text}"
                current_result_item["question"] = question_for_llm_final # Update question in results with choices
                is_mcq_for_accuracy = True

            llm_tasks_args_list = []
            for k_value in top_k_values_for_llm:
                context_metadata_for_k = retrieval_metadata[:k_value]
                # Repack the metadata for final output
                sorted_retrieved_metadata = repack_result(context_metadata_for_k)
                # structure the context
                construct_context = construct_ta_rag_context(sorted_retrieved_metadata)
                llm_tasks_args_list.append((question_for_llm_final, construct_context, k_value, item_id))

            future_to_k_val_map = {}
            if llm_tasks_args_list:
                future_to_k_val_map = {
                    executor.submit(timed_llm_call_wrapper, args): args[2] for args in llm_tasks_args_list
                }

            all_k_variants_processed_ok_for_mcq = bool(llm_tasks_args_list) # Start true if tasks were made

            for future in concurrent.futures.as_completed(future_to_k_val_map):
                k_val_result = future_to_k_val_map[future]
                try:
                    _, llm_response_raw, gen_time_result = future.result()
                    current_result_item[f"llm_response_k{k_val_result}"] = llm_response_raw
                    current_result_item[f"generation_time_k{k_val_result}_seconds"] = gen_time_result

                    if "LLM_Error" in llm_response_raw or "Error: Empty Context" in llm_response_raw:
                        all_k_variants_processed_ok_for_mcq = False
                    
                    # Standardize LLM answer for comparison
                    processed_llm_ans = str(llm_response_raw).lower() # Ensure string and lower
                    if "answer:" in processed_llm_ans:
                        processed_llm_ans = processed_llm_ans.split("answer:")[-1]
                    processed_llm_ans = processed_llm_ans.replace('.', '').replace('*', '').strip()
                    processed_llm_ans = processed_llm_ans.replace('(', '').replace(')', '').strip()

                    logging.debug(f"Q_id: {item_id} | k={k_val_result} | GT: {ground_truth_answer} | LLM_ans_raw: '{llm_response_raw[:30]}...' | LLM_proc: '{processed_llm_ans}'")
                    if processed_llm_ans.lower()  == ground_truth_answer.lower() : # GT is already lower
                        accuracies[k_val_result] += 1
                except Exception as e:
                    logging.error(f"Error processing future for Q_id {item_id}, k={k_val_result}: {e}")
                    logging.error(f"Error processing future for Q_id {item_id}, k={k_val_result}: {e}")
                    current_result_item[f"llm_response_k{k_val_result}"] = f"Error: Processing future failed {e}"
                    all_k_variants_processed_ok_for_mcq = False
            
            if is_mcq_for_accuracy and all_k_variants_processed_ok_for_mcq:
                total_mcq_questions_processed += 1
            elif is_mcq_for_accuracy and not all_k_variants_processed_ok_for_mcq:
                logging.warning(f"Q_id: {item_id} (MCQ) had issues with one or more k-variants for LLM. Not counted in accuracy denominator.")

            results_list.append(current_result_item)
            if (item_index + 1) % 20 == 0:
                logging.info(f"Completed {item_index + 1} TA-RAG evaluations.")
                for k_value_llm in top_k_values_for_llm:
                    acc_count = accuracies[k_value_llm]
                    acc_percent = acc_count / total_mcq_questions_processed
                    line = f"Top {k_value_llm} context snippets: {acc_count} / {total_mcq_questions_processed} ({acc_percent:.2%})"
                    logging.info(line)
                                
    # Accuracy summary
    accuracy_summary_lines = [f"TA-RAG Evaluation Accuracy Summary ({time.strftime('%Y-%m-%d %H:%M:%S')})"]
    accuracy_summary_lines.append(f"Test Data File: {test_data_file}")
    if total_mcq_questions_processed > 0:
        accuracy_summary_lines.append(f"Total MCQ questions processed for accuracy: {total_mcq_questions_processed}")
        logging.info("TA-RAG MCQ Accuracy Summary:")
        for k_value_llm in top_k_values_for_llm:
            acc_count = accuracies[k_value_llm]
            acc_percent = acc_count / total_mcq_questions_processed
            line = f"Top {k_value_llm} context snippets: {acc_count} / {total_mcq_questions_processed} ({acc_percent:.2%})"
            accuracy_summary_lines.append(line)
            logging.info(line)
        
        accuracy_csv_row = {"question_id": "ACCURACY_MCQ_TA_RAG", "question": f"Total MCQ questions: {total_mcq_questions_processed}"}
        for k_val_llm in top_k_values_for_llm:
            acc = accuracies[k_val_llm] / total_mcq_questions_processed
            accuracy_csv_row[f"llm_response_k{k_val_llm}"] = f"{accuracies[k_val_llm]}/{total_mcq_questions_processed} ({acc:.2%})"
        results_list.append(accuracy_csv_row)
    else:
        line = "No MCQ questions processed successfully for TA-RAG accuracy calculation."
        accuracy_summary_lines.append(line)
        logging.info(line)

    try:
        with open(output_accuracy_txt_file, 'w') as f:
            for summary_line in accuracy_summary_lines: # Renamed variable
                f.write(summary_line + "\n")
        logging.info(f"TA-RAG Accuracy summary saved to: {output_accuracy_txt_file}")
    except Exception as e: # Added colon
        logging.error(f"Failed to write TA-RAG accuracy summary: {e}")
        
    return pd.DataFrame(results_list)


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure RESULT_DIR exists (using a local relative path for easier execution)
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        logging.info(f"Created result directory: {RESULT_DIR}")
    
    # Using fixed names as per original request for consistency
    output_csv_path = os.path.join(RESULT_DIR, TA_OUTPUT_CSV_FILE)
    output_accuracy_txt_path = os.path.join(RESULT_DIR, TA_OUTPUT_ACCURACY_TXT_FILE)


    total_script_start_time = time.time()
    logging.info("Loading TA-RAG artifacts...")
    ta_faiss_index, ta_metadata_store, ta_ncls_index = load_ta_rag_artifacts(
        TA_RAG_FAISS_INDEX_FILE, TA_RAG_METADATA_FILE
    )

    if not all([ta_faiss_index, ta_metadata_store, ta_ncls_index]): # Check if any are None
        logging.error("Failed to load one or more TA-RAG artifacts. Exiting.")
        sys.exit(1)
    if ta_faiss_index.ntotal == 0: # Check if index is empty AFTER loading
        logging.error("TA-RAG Faiss index is empty after loading. Cannot proceed.")
        sys.exit(1)

    logging.info(f"Starting TA-RAG evaluation (Refactored).")
    logging.info(f"Test data file: {TEST_DATA_FILE}")
    logging.info(f"Output CSV file: {output_csv_path}")
    logging.info(f"Output Accuracy TXT file: {output_accuracy_txt_path}")

    results_df = evaluate_and_store_ta_rag(
        TEST_DATA_FILE, ta_faiss_index, ta_metadata_store, ta_ncls_index, output_accuracy_txt_path
    )
    top_k_values_for_llm = [5,10,20,50]

    if not results_df.empty:
        try:
            cols_order = [
                "question_id", "question", "ground_truth_answer", 
                "total_retrieval_time_seconds", "semantic_search_time_seconds", "temporal_processing_time_seconds",
                "query_embedding_time_seconds", "repack_time_seconds"
            ]
            for k_llm in top_k_values_for_llm: # Use the variable directly
                cols_order.append(f"llm_response_k{k_llm}")
                cols_order.append(f"generation_time_k{k_llm}_seconds")
            cols_order.extend(["top_50_retrieved_ids", "question_type_id"])
            
            # Ensure all columns in cols_order exist in DataFrame, add others at the end
            present_cols = [col for col in cols_order if col in results_df.columns]
            remaining_df_cols = [col for col in results_df.columns if col not in present_cols]
            final_cols_ordered = present_cols + remaining_df_cols
            
            results_df[final_cols_ordered].to_csv(output_csv_path, index=False)
            logging.info(f"TA-RAG evaluation complete. Results saved to: {output_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save TA-RAG results to CSV: {e}")
            # Fallback: save with whatever columns exist
            results_df.to_csv(output_csv_path + "_fallback.csv", index=False)
            logging.info(f"Fallback results saved to: {output_csv_path}_fallback.csv")

    else:
        logging.info("No TA-RAG results generated to save.")

    total_script_run_time = time.time() - total_script_start_time
    logging.info(f"Total TA-RAG script execution time: {total_script_run_time:.2f} seconds.")