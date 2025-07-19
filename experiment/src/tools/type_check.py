from datetime import datetime
import logging 

def temporal_process_sentence_validator(response):
    """
    Validates if a dictionary matches the expected structure.

    Args:
        response: The dictionary to validate.

    Returns:
        True if the dictionary is valid, False otherwise.  logging.errors error messages
        to explain why the validation failed.
    """

    if not isinstance(response, dict):
        logging.error("Error: Response is not a dictionary.")
        return False

    required_keys = {"rephrased_sentence", "temporal_decomposition"}
    if not all(key in response for key in required_keys):
        logging.error("Error: Response is missing required keys.")
        return False

    if not isinstance(response["rephrased_sentence"], str):
        logging.error("Error: 'rephrased_sentence' is not a string.")
        return False
    
    if response["temporal_decomposition"] is not None: 
        if not isinstance(response["temporal_decomposition"], list):
            logging.error("Error: 'temporal_decomposition' is not a list.")
            return False

        for item in response["temporal_decomposition"]:
            if not isinstance(item, dict):
                logging.error("Error: Item in 'temporal_decomposition' is not a dictionary.")
                return False

            required_item_keys = {"begin", "end"}
            if not all(key in item for key in required_item_keys):
                logging.error("Error: Item in 'temporal_decomposition' is missing required keys.")
                return False

            if not isinstance(item["begin"], str) or not isinstance(item["end"], str):
                logging.error("Error: 'begin' or 'end' is not a string.")
                return False

            try:
                datetime.fromisoformat(item["begin"].replace('T', ' ').replace('Z', '+00:00'))
                datetime.fromisoformat(item["end"].replace('T', ' ').replace('Z', '+00:00'))
            except ValueError:
                logging.error("Error: Invalid date format in 'begin' or 'end'.")
                return False

    return True 