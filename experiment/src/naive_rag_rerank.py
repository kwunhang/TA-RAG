import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import pandas as pd
import sys
import logging
import os # For checking file existence
import time # For timing operations
import concurrent.futures 
from tools.llm_response import LLMClient
from FlagEmbedding import FlagReranker


LLM_URL = os.environ.get("LLM_URL")
client_B_config = {
    "llm_base_url": LLM_URL,
    "llm_api_key": "",
    "model": "meta-llama/Llama-3.3-70B-Instruct"
}

LLM_CLIENT_B = LLMClient(**client_B_config)



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DIMENSION = 768
MAX_WORKERS_FOR_LLM_CALLS = 4 
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
FAISS_INDEX_FILE = "index/naive_rag_hnsw.faissindex"
METADATA_FILE = "index/naive_rag_metadata.json"
TEST_DATA_FILE = 'dqabench_mcqa_eval_set_refine.json'
BATCH_SIZE = 128
OUTPUT_CSV_FILE = "naive_rag_rerank_70b_batch_faiss_hnsw_results_2.csv"
OUTPUT_ACCURACY_TXT_FILE = "naive_rag_rerank_70b_batch_faiss_hnsw_accuracy_summary_2.txt"

RESULT_DIR = os.environ.get("RESULT_DIR", "result_new/")
# --- Initialize Embedding Model ---
try:
    # EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device="cuda" if faiss.get_num_gpus() > 0 else "cpu", trust_remote_code=True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    EMBED_MODEL = SentenceTransformer(
        EMBED_MODEL_NAME, device="cuda:0", trust_remote_code=True
    )
    RERANKER = FlagReranker('BAAI/bge-reranker-v2-m3', device="cuda:0", use_fp16=True)
    logging.info(f"Successfully loaded sentence transformer model: {EMBED_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to load sentence transformer model: {e}")
    sys.exit(1)


def reranking(query: str, docs: list[str], docs_id: list[str], reranker_model: FlagReranker = RERANKER) -> tuple[list[str], list[str]]:
    """
    Reranks a list of documents based on their relevance to a given query using a FlagReranker model.

    Args:
        query: The original query string.
        docs: A list of retrieved document strings.
        docs_id: A list of IDs corresponding to the documents in 'docs'.
                 The order must match the 'docs' list.
        reranker_model: An initialized instance of FlagReranker.

    Returns:
        A tuple containing two lists:
        - reranked_docs: The list of document strings, sorted by relevance score (highest first).
        - reranked_ids: The list of document IDs, sorted according to the new document order.
        Returns the original lists if the input 'docs' is empty or 'reranker_model' is None.
    """
    if not docs:
        logging.warning("No documents provided for reranking.")
        return [], []

    if reranker_model is None:
        logging.warning("Reranker model not available. Returning original order.")
        return docs, docs_id # Return original order if model failed to load

    # Create pairs of [query, document] for the reranker
    pairs = [[query, doc] for doc in docs]

    # start_time = time.time()
    scores = reranker_model.compute_score(pairs, normalize=False)

    scored_docs = list(zip(scores, docs, docs_id))

    # Sort the combined list based on the score in descending order
    # Higher scores mean more relevant documents
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    if not scored_docs: # Handle case if somehow scored_docs becomes empty
        return [], []
        
    # We only need the sorted docs and ids, we can discard the scores now
    _, reranked_docs, reranked_ids = zip(*scored_docs)

    reranked_docs = list(reranked_docs)
    reranked_ids = list(reranked_ids)

    # logging.info(f"Reranking complete. Top score: {scored_docs[0][0]:.4f}, Bottom score: {scored_docs[-1][0]:.4f}")

    return reranked_docs, reranked_ids

# --- Helper Functions ---
def generate_embedding(texts: list[str], is_query: bool = False) -> np.ndarray:
    if is_query:
        texts_to_embed = ["search_query: " + text for text in texts]
    else:
        texts_to_embed = texts
    embeddings = EMBED_MODEL.encode(texts_to_embed, batch_size=len(texts_to_embed), convert_to_tensor=False, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)

def load_faiss_index_and_metadata(index_path, metadata_path):
    if not os.path.exists(index_path):
        logging.error(f"Faiss index file not found: {index_path}")
        return None, None
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file not found: {metadata_path}")
        return None, None
    try:
        index = faiss.read_index(index_path)
        logging.info(f"Faiss index loaded from {index_path}. Total vectors: {index.ntotal}")
    except Exception as e:
        logging.error(f"Failed to load Faiss index: {e}")
        return None, None
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info(f"Metadata loaded from {metadata_path}. Total entries: {len(metadata)}")
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        return None, None
    if index.ntotal != len(metadata):
        logging.warning(f"Mismatch: Faiss index has {index.ntotal} vectors, metadata has {len(metadata)} entries.")
    return index, metadata

def search_faiss_hnsw(faiss_index, query_text, metadata_store, top_k=10):
    if not faiss_index or faiss_index.ntotal == 0:
        logging.warning("Faiss index is not loaded or is empty.")
        return [], [], 0.0 # texts, ids, time

    embed_start_time = time.time()
    query_embedding = generate_embedding([query_text], is_query=True)
    
    retreive_start_time = time.time()
    try:
        search_param = faiss.SearchParametersHNSW(efSearch=top_k+200)
        distances, indices = faiss_index.search(query_embedding, top_k, params=search_param)
    except Exception as e:
        logging.error(f"Error during Faiss search: {e}")
        return [], [], 0.0
    embed_time = retreive_start_time - embed_start_time
    retrieval_time = time.time() - retreive_start_time

    retrieved_texts = []
    retrieved_ids = [] 
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if 0 <= idx < len(metadata_store):
            meta_item = metadata_store[idx]
            retrieved_texts.append(meta_item.get("chunk_text", ""))
            retrieved_ids.append(f"{meta_item.get('doc_uid')}")
        else:
            logging.warning(f"Retrieved index {idx} is out of bounds for metadata_store (size {len(metadata_store)}).")
    return retrieved_texts, retrieved_ids, retrieval_time, embed_time

def load_test_data(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading test data from {json_file}: {e}")
        return None

def timed_llm_call_wrapper(question_llm_tuple):
    """
    Wrapper to call response_rag_mc and time it.
    Args:
        question_llm_tuple (tuple): (question_for_llm, context_list, k_value, item_id_for_log)
    Returns:
        tuple: (k_value, llm_response_raw, generation_time)
    """
    question_for_llm, context_list, k_value, item_id_for_log = question_llm_tuple
    gen_start_time = time.time()
    try:
        llm_response_raw = LLM_CLIENT_B.response_rag_mc(question_for_llm, context_list)
    except Exception as e:
        logging.error(f"LLM Error (k={k_value}, Q_id: {item_id_for_log}): {e}")
        llm_response_raw = f"LLM_Error: {e}"
    generation_time = time.time() - gen_start_time
    # logging.debug(f"LLM Call (k={k_value}, Q_id: {item_id_for_log}) took {generation_time:.2f}s")
    return k_value, llm_response_raw, generation_time

# --- Main Evaluation Loop for Faiss RAG ---
def evaluate_and_store_faiss_rag(test_data_file, faiss_index, metadata_store, output_accuracy_txt_file):
    test_data = load_test_data(test_data_file)
    if not test_data:
        return pd.DataFrame()

    results_list = []
    top_k_values = [5, 10, 20, 50]
    accuracies = {k: 0 for k in top_k_values}
    total_mcq_questions_processed = 0 # Number of MCQs for which accuracy is calculated
    i = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_FOR_LLM_CALLS) as executor:
        for item_index, item in enumerate(tqdm(test_data, desc="Evaluating RAG performance")):
            org_question = item.get("question", "")
            question_for_llm = org_question
            # q_type = item.get("type", "").lower()
            ground_truth_answer = item.get("correct_answer_key", "")
            item_id = item.get("id", f"item_{item_index}") # Use provided ID or generate one
            question_type_id = item["query_type_info"]["id"]

            current_result_item = {
                "question_id": item_id,
                "question": question_for_llm, # Will update if MCQ with options
                "ground_truth_answer": ground_truth_answer,
                "retrieval_time_seconds": 0.0,
                "top_50_retrieved_ids": [],
                "question_type_id": question_type_id
            }
            for k_val in top_k_values: # Initialize generation time fields
                current_result_item[f"llm_response_k{k_val}"] = "N/A"
                current_result_item[f"generation_time_k{k_val}_seconds"] = 0.0

            # Perform one search for the maximum k needed (50)
            # search k*20 for reranking
            max_k_retrieval = 50 * 20
            # max_k_retrieval = 50 # As per requirement to store top-50 doc_ids

            
            retrieved_chunks_texts_full, retrieved_ids_full, retrieval_time, embed_time = search_faiss_hnsw(
                faiss_index, org_question, metadata_store, top_k=max_k_retrieval
            )

            if not retrieved_chunks_texts_full:
                logging.warning(f"No documents retrieved for question: {org_question}")
                results_list.append(current_result_item) # Append item with N/A responses
                continue
            
            # reranking
            reranked_chunks_texts_full, reranked_retrieved_ids_full = reranking(
                org_question, retrieved_chunks_texts_full, retrieved_ids_full
            )

            current_result_item["embedding_time_seconds"] = embed_time
            current_result_item["retrieval_time_seconds"] = retrieval_time
            current_result_item["top_50_retrieved_ids"] = reranked_retrieved_ids_full[:top_k_values[-1]]

            
            is_mcq_for_accuracy = False
            if "choices" in item:
                question_for_llm = org_question + "\nPlease select the most accurate description from the options below:\n"
                for key, choice in item["choices"].items():
                    question_for_llm += f"\n{key}) {choice}"
                current_result_item["question"] = question_for_llm # Update question in results
                is_mcq_for_accuracy = True

            # Prepare tasks for concurrent execution
            llm_tasks_args = []
            for k_value in top_k_values:
                context_for_current_k = reranked_chunks_texts_full[:k_value]
                llm_tasks_args.append((question_for_llm, context_for_current_k, k_value, item_id))

            # Submit tasks to executor
            # Note: futures will complete in arbitrary order, so we process them as they complete.
            future_to_k_value = {executor.submit(timed_llm_call_wrapper, args): args[2] for args in llm_tasks_args}

            processed_all_k_for_this_mcq = True

            for future in concurrent.futures.as_completed(future_to_k_value):
                k_val_result, llm_response_raw, gen_time_result = future.result()
                current_result_item[f"llm_response_k{k_val_result}"] = llm_response_raw
                current_result_item[f"generation_time_k{k_val_result}_seconds"] = gen_time_result
                if "LLM_Error" in llm_response_raw:
                    processed_all_k_for_this_mcq = False

                processed_llm_ans = llm_response_raw.lower()
                if "answer:" in processed_llm_ans:
                    processed_llm_ans = processed_llm_ans.split("answer:")[-1]
                processed_llm_ans = processed_llm_ans.replace('.', '').replace('*', '').strip()
                processed_llm_ans = processed_llm_ans.replace('(', '').replace(')', '').strip()
                if len(processed_llm_ans) > 1 and processed_llm_ans.startswith('(') and processed_llm_ans.endswith(')'):
                    processed_llm_ans = processed_llm_ans[1:-1]
                if len(processed_llm_ans) > 1 and processed_llm_ans.endswith(')'):
                    processed_llm_ans = processed_llm_ans[:-1]

                logging.debug(f"Q_id: {item_id} | k={k_val_result} | GT: {ground_truth_answer} | LLM_proc: {processed_llm_ans}")
                if processed_llm_ans.lower() == ground_truth_answer.lower():
                    accuracies[k_val_result] += 1

            if is_mcq_for_accuracy and processed_all_k_for_this_mcq:
                total_mcq_questions_processed += 1 # Count if all k-variants were processed without LLM errors
            elif is_mcq_for_accuracy and not processed_all_k_for_this_mcq:
                logging.warning(f"Q_id: {item_id} was an MCQ but not all k-variants processed successfully. Not counted in total_mcq_questions_processed for accuracy denominator.")


            results_list.append(current_result_item)
            i += 1
            if i%20 == 0:
                logging.info(f"Compleqte {i} MCQA")
                
    
    # Prepare and write accuracy summary to text file
    accuracy_summary_lines = [f"RAG Evaluation Accuracy Summary ({time.strftime('%Y-%m-%d %H:%M:%S')})"]
    accuracy_summary_lines.append(f"Test Data File: {test_data_file}")
    if total_mcq_questions_processed > 0:
        accuracy_summary_lines.append(f"Total MCQ questions processed for accuracy: {total_mcq_questions_processed}")
        logging.info("MCQ Accuracy Summary:")
        for k_value in top_k_values:
            acc_count = accuracies[k_value]
            acc_percent = acc_count / total_mcq_questions_processed if total_mcq_questions_processed > 0 else 0
            line = f"Top {k_value}: {acc_count} / {total_mcq_questions_processed} ({acc_percent:.2%})"
            accuracy_summary_lines.append(line)
            logging.info(line)
        
        # Add accuracy summary row to CSV as well for completeness
        accuracy_csv_row = {"question_id": "ACCURACY_MCQ", "question": f"Total MCQ questions: {total_mcq_questions_processed}"}
        for k_value in top_k_values:
            acc = accuracies[k_value] / total_mcq_questions_processed if total_mcq_questions_processed > 0 else 0
            accuracy_csv_row[f"llm_response_k{k_value}"] = f"{accuracies[k_value]}/{total_mcq_questions_processed} ({acc:.2%})"
        results_list.append(accuracy_csv_row)

    else:
        line = "No MCQ questions found or processed for accuracy calculation."
        accuracy_summary_lines.append(line)
        logging.info(line)

    try:
        with open(output_accuracy_txt_file, 'w') as f:
            for line in accuracy_summary_lines:
                f.write(line + "\n")
        logging.info(f"Accuracy summary saved to: {output_accuracy_txt_file}")
    except Exception as e:
        logging.error(f"Failed to write accuracy summary to {output_accuracy_txt_file}: {e}")
        
    df = pd.DataFrame(results_list)
    return df

# --- Main Execution ---
if __name__ == "__main__":

    total_start_time = time.time()
    faiss_index, metadata_store = load_faiss_index_and_metadata(FAISS_INDEX_FILE, METADATA_FILE)

    if not faiss_index or not metadata_store:
        logging.error("Failed to load Faiss index or metadata. Exiting.")
        sys.exit(1)

    output_csv_path = os.path.join(RESULT_DIR, OUTPUT_CSV_FILE)
    output_accuracy_txt_path = os.path.join(RESULT_DIR, OUTPUT_ACCURACY_TXT_FILE)


    logging.info(f"Starting RAG evaluation with Faiss HNSW index (refined output).")
    logging.info(f"Test data file: {TEST_DATA_FILE}")
    logging.info(f"Output CSV file: {output_csv_path}")
    logging.info(f"Output Accuracy TXT file: {output_accuracy_txt_path}")
    # Change logging level for less verbose output during run, if desired
    # logging.getLogger().setLevel(logging.INFO) # or logging.WARNING

    results_df = evaluate_and_store_faiss_rag(TEST_DATA_FILE, faiss_index, metadata_store, output_accuracy_txt_path)
    
    if not results_df.empty:
        try:
            # Reorder columns for better readability if desired
            cols_order = ["question_id", "question", "ground_truth_answer", "retrieval_time_seconds"]
            for k in [5, 10, 20, 50]:
                cols_order.append(f"llm_response_k{k}")
                cols_order.append(f"generation_time_k{k}_seconds")
            cols_order.append("top_50_retrieved_ids")
            cols_order.append("question_type_id")
            
            # Filter existing columns to ensure no error if some are missing (e.g. accuracy row)
            df_cols = [col for col in cols_order if col in results_df.columns]
            remaining_cols = [col for col in results_df.columns if col not in df_cols]
            final_cols_order = df_cols + remaining_cols
            
            results_df = results_df[final_cols_order]
            results_df.to_csv(output_csv_path, index=False)
            logging.info(f"RAG evaluation complete. Results saved to: {output_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save results to CSV: {e}")
    else:
        logging.info("No results generated to save.")

    total_used = time.time() - total_start_time
    print(f"Total time taken for evaluation: {total_used/60:.2f} minutes.")