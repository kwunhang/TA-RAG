from openai import OpenAI
from datetime import date, datetime
import ast
import logging
from .type_check import temporal_process_sentence_validator
from typing import AsyncGenerator

class LLMClient:
    def __init__(self, llm_base_url: str, llm_api_key: str, model: str):
        """
        Initializes the LLMClient with specific connection parameters and model.

        Args:
            llm_base_url (str): The base URL for the LLM API.
            llm_api_key (str): The API key for the LLM API.
            model (str): The model identifier to be used.
        """
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.model = model
        self.openai_client = OpenAI(
            api_key=self.llm_api_key, base_url=self.llm_base_url
        )

    def _contains_ellipsis(self, obj) -> bool:
        """
        Recursively checks if a Python object contains the Ellipsis singleton.
        This is a helper method.
        """
        if obj is Ellipsis:
            return True
        elif isinstance(obj, (list, tuple)):
            return any(self._contains_ellipsis(item) for item in obj)
        elif isinstance(obj, dict):
            return any(self._contains_ellipsis(value) for value in obj.values())
        return False


            
    def extract_temporal_interval(self, question: str, chance: int = 3):
        current_date = date.today()
        system_prompt = f"""
    You are a temporal information extraction expert specializing in identifying time intervals relevant to economic questions. You will be given a question related to Hong Kong's economy. Your task is to determine the relevant time interval implied by the question. This time interval will be used to retrieve relevant documents for a Retrieval-Augmented Generation (RAG) system.

    Analyze the question and extract the *most appropriate* time interval for document retrieval.  Return this interval in the specified format. Consider the current date when interpreting relative time expressions (e.g., "past 5 years").

    Current Date: {current_date}

    Output:

    Return a single JSON object containing the `begin` and `end` of the time interval:

    *   `begin`: The start date and time of the interval.
    *   `end`: The end date and time of the interval.

    Format both `begin` and `end` in ISO 8601 format: `YYYY-MM-DDTHH:MM:SS`.

    IMPORTANT Considerations:

    *   **Explicit Dates:** If the question explicitly mentions specific dates or years, use those.
    *   **Relative Time:** If the question uses relative time expressions (e.g., "past 5 years", "last quarter", "recent months"), calculate the absolute dates based on the `Current Date`.
    *   **Implicit Time:** If the question implies a specific time period without explicitly stating it (e.g., "What is the current unemployment rate?"), assume a reasonable, recent interval (e.g. a few recent months, current year).  Use your best judgment based on the nature of the economic indicator.
    *   **Date Range Defaults:**
        *   If only a year is mentioned, use January 1st at 00:00:00 for the `begin` and December 31st at 23:59:59 for the `end`.
        *   If only a month and year are mentioned, use the 1st of the month at 00:00:00 for the `begin` and the last day of the month at 23:59:59 for the `end`.
        *    If no specific time given, use 00:00:00
    *   **Overlapping Intervals:** If the question implies multiple, potentially overlapping intervals, choose the *widest* interval that encompasses all relevant periods.
    * **No Time Interval** If no time interval can be determined, return null

    Example Output Format (JSON):

    ```json
    {{
    "begin": "2020-01-01T00:00:00",
    "end": "2022-12-31T23:59:59"
    }}
    ```
    Example with implicit time(JSON):
    ```json
    {{
    "begin": "2024-01-01T00:00:00",
    "end": "2024-12-31T23:59:59"
    }}
    ```
    Example with no time interval(JSON):
    ```json
    {{
    "begin": null,
    "end": null
    }}
    ```
    """
        
        user_prompt = f"""
    <QUESTION>
    {question}
    </QUESTION>
    """
        attempt = 0
        while attempt < chance:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                        "role": "system",
                        "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    temperature=0,
                    max_tokens=2048,
                    top_p=0.3,
                )   
                time_interval = response.choices[0].message.content.strip()
                time_interval = time_interval.replace('"\n', '",\n')
                time_interval = time_interval[time_interval.find("{"): time_interval.rfind("}")+1]
                time_interval_oj = ast.literal_eval(time_interval)
                if datetime.fromisoformat(time_interval_oj["begin"].replace("Z", "+00:00")).timestamp() and \
                    datetime.fromisoformat(time_interval_oj["end"].replace("Z", "+00:00")).timestamp():
                    logging.info(response.choices[0].message.content.strip())
                    return time_interval_oj
            except Exception as e:
                logging.error(f"Error generating time from question': {e}")
            attempt += 1
        return None

    def response_rag(self, question: str, sources: list[str]):
        prompt = """
    You are a helpful and truthful assistant. Your task is to answer user questions based on the provided knowledge snippets.

    Below are knowledge snippets retrieved from a database. Each snippet is marked with its source (e.g., "Snippet 1 [filename:page_number]").  Treat each snippet as an independent piece of evidence.

    --- Knowledge Snippets ---
    {context}

    --- Instructions ---

    1.  **Analyze Before Answering:** Before formulating your response, carefully read and analyze *all* provided snippets. Consider how they relate to each other (e.g., corroborating, conflicting, providing different facets of information, different timestamps). This analysis will help you provide a more accurate and comprehensive answer based *solely* on the provided text.
    2.  **Answer the user's question (provided below) using ONLY the information synthesized from the knowledge snippets.**
    3.  **Handle Conflicting Information:** If the snippets contain conflicting information, identify the conflict clearly. Attempt to reconcile the differences *if the snippets themselves provide clues* (e.g., different dates, contexts, levels of detail mentioned in the text). Present the conflicting information neutrally, citing the sources for each conflicting point. Do not arbitrarily favor one snippet over another; base any reconciliation attempt strictly on the text provided in the snippets.
    4.  **Handle Insufficient Information:** If the knowledge snippets DO NOT contain sufficient information to answer the question, clearly state "I don't know" or "Insufficient information to answer based on the provided snippets." Do NOT make up an answer.
    5.  **No External Information:** Do not include information that cannot be directly supported by the provided snippets.
    6.  **Maintain Neutrality:** Maintain a neutral and objective tone throughout your response.
    7.  **Cite Sources:** If appropriate for the question, cite the relevant snippet source(s) in your response (e.g., "[Source: file1.pdf:2]"). This helps with verifiability.
    8.  **Structure Clearly:** Structure your response clearly and concisely. If a specific format (e.g., a list, a summary) or length is requested, adhere to it.

    --- User Question ---

    {user_question}"""
        context = ""
        for i, snippet in enumerate(sources):
            context += f"Snippet {i+1}:\n{snippet}\n\n"
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                "role": "user",
                "content": prompt.format(context=context, user_question=question)
                },
            ],
            temperature=0,
            max_tokens=4096,
            top_p=0.2,
        )   
        return response.choices[0].message.content.strip()


    def response_rag_time(self, question: str, context: str):
        prompt = """
    You are a helpful and truthful assistant. Your task is to answer user questions based on the provided knowledge snippets.

    Below are knowledge snippets retrieved from a database. Each snippet is marked with its source (e.g., "Snippet 1 [filename:page_number]").  Treat each snippet as an independent piece of evidence.

    --- Knowledge Snippets ---
    {context}

    --- Instructions ---

    1.  **Analyze Before Answering:** Before formulating your response, carefully read and analyze *all* provided snippets. Consider how they relate to each other (e.g., corroborating, conflicting, providing different facets of information, different timestamps). This analysis will help you provide a more accurate and comprehensive answer based *solely* on the provided text.
    2.  **Answer the user's question (provided below) using ONLY the information synthesized from the knowledge snippets.**
    3.  **Handle Conflicting Information:** If the snippets contain conflicting information, identify the conflict clearly. Attempt to reconcile the differences *if the snippets themselves provide clues* (e.g., different dates, contexts, levels of detail mentioned in the text). Present the conflicting information neutrally, citing the sources for each conflicting point. Do not arbitrarily favor one snippet over another; base any reconciliation attempt strictly on the text provided in the snippets.
    4.  **Handle Insufficient Information:** If the knowledge snippets DO NOT contain sufficient information to answer the question, clearly state "I don't know" or "Insufficient information to answer based on the provided snippets." Do NOT make up an answer.
    5.  **No External Information:** Do not include information that cannot be directly supported by the provided snippets.
    6.  **Maintain Neutrality:** Maintain a neutral and objective tone throughout your response.
    7.  **Cite Sources:** If appropriate for the question, cite the relevant snippet source(s) in your response (e.g., "[Source: file1.pdf:2]"). This helps with verifiability.
    8.  **Structure Clearly:** Structure your response clearly and concisely. If a specific format (e.g., a list, a summary) or length is requested, adhere to it.

    --- User Question ---

    {user_question}"""
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                "role": "user",
                "content": prompt.format(context=context, user_question=question)
                },
            ],
            temperature=0,
            max_tokens=4096,
            top_p=0.2,
        )   
        return response.choices[0].message.content.strip()

    def response_rag_mc(self, question: str, sources: list[str]):
        prompt = """
    You are a helpful and truthful assistant. Your task is to answer user questions based on the provided knowledge snippets.

    Below are knowledge snippets retrieved from a database. Each snippet is marked with its source (e.g., "Snippet 1 [filename:page_number]").  Treat each snippet as an independent piece of evidence.

    --- Knowledge Snippets ---
    {context}

    --- Instructions ---

    1.  **Analyze Before Answering:** Before formulating your response, carefully read and analyze *all* provided snippets. Consider how they relate to each other (e.g., corroborating, conflicting, providing different facets of information, different timestamps). This analysis will help you provide a more accurate and comprehensive answer based *solely* on the provided text.
    2.  **Answer the user's question (provided below) using ONLY the information synthesized from the knowledge snippets.**
    3.  **Handle Conflicting Information:** If the snippets contain conflicting information, identify the conflict clearly. Attempt to reconcile the differences *if the snippets themselves provide clues* (e.g., different dates, contexts, levels of detail mentioned in the text). Present the conflicting information neutrally, citing the sources for each conflicting point. Do not arbitrarily favor one snippet over another; base any reconciliation attempt strictly on the text provided in the snippets.
    4.  **Handle Insufficient Information:** If the knowledge snippets DO NOT contain sufficient information to answer the question, clearly state "I don't know" or "Insufficient information to answer based on the provided snippets." Do NOT make up an answer.
    5.  **No External Information:** Do not include information that cannot be directly supported by the provided snippets.
    6.  **Maintain Neutrality:** Maintain a neutral and objective tone throughout your response.
    7.  **Cite Sources:** If appropriate for the question, cite the relevant snippet source(s) in your response (e.g., "[Source: file1.pdf:2]"). This helps with verifiability.
    8.  **Structure Clearly:** Structure your response clearly and concisely. If a specific format (e.g., a list, a summary) or length is requested, adhere to it.
    9.  **Final Answer Line Format:** If the user question is a multiple-choice question (e.g., choose A, B, C, D)  AND you can determine an answer from the snippets: Your *entire response* must end with a final, separate line containing *only* the answer in the format: `Answer: [Letter]` (e.g., `Answer: A`, `Answer: B`, `Answer: C`, `Answer: D`).

    --- User Question ---

    {user_question}"""
        context = ""
        for i, snippet in enumerate(sources):
            context += f"Snippet {i+1}:\n{snippet}\n\n"
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                "role": "user",
                "content": prompt.format(context=context, user_question=question)
                },
            ],
            temperature=0,
            max_tokens=4096,
            top_p=0.2,
        )   
        return response.choices[0].message.content.strip()

    def response_rag_time_mc(self, question: str, context: str):
        prompt = """
    You are a helpful and truthful assistant. Your task is to answer user questions based on the provided knowledge snippets.

    Below are knowledge snippets retrieved from a database. Each snippet is marked with its source (e.g., "Snippet 1 [filename:page_number]").  Treat each snippet as an independent piece of evidence.

    --- Knowledge Snippets ---
    {context}

    --- Instructions ---

    1.  **Analyze Before Answering:** Before formulating your response, carefully read and analyze *all* provided snippets. Consider how they relate to each other (e.g., corroborating, conflicting, providing different facets of information, different timestamps). This analysis will help you provide a more accurate and comprehensive answer based *solely* on the provided text.
    2.  **Answer the user's question (provided below) using ONLY the information synthesized from the knowledge snippets.**
    3.  **Handle Conflicting Information:** If the snippets contain conflicting information, identify the conflict clearly. Attempt to reconcile the differences *if the snippets themselves provide clues* (e.g., different dates, contexts, levels of detail mentioned in the text). Present the conflicting information neutrally, citing the sources for each conflicting point. Do not arbitrarily favor one snippet over another; base any reconciliation attempt strictly on the text provided in the snippets.
    4.  **Handle Insufficient Information:** If the knowledge snippets DO NOT contain sufficient information to answer the question, clearly state "I don't know" or "Insufficient information to answer based on the provided snippets." Do NOT make up an answer.
    5.  **No External Information:** Do not include information that cannot be directly supported by the provided snippets.
    6.  **Maintain Neutrality:** Maintain a neutral and objective tone throughout your response.
    7.  **Cite Sources:** If appropriate for the question, cite the relevant snippet source(s) in your response (e.g., "[Source: file1.pdf:2]"). This helps with verifiability.
    8.  **Structure Clearly:** Structure your response clearly and concisely. If a specific format (e.g., a list, a summary) or length is requested, adhere to it.
    9.  **Final Answer Line Format:** If the user question is a multiple-choice question (e.g., choose A, B, C, D)  AND you can determine an answer from the snippets: Your *entire response* must end with a final, separate line containing *only* the answer in the format: `Answer: [Letter]` (e.g., `Answer: A`, `Answer: B`, `Answer: C`, `Answer: D`).

    --- User Question ---

    {user_question}"""
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                "role": "user",
                "content": prompt.format(context=context, user_question=question)
                },
            ],
            temperature=0,
            max_tokens=4096,
            top_p=0.2,
        )   
        return response.choices[0].message.content.strip()

    def temporal_process_sentence(self, input_sentence: str, chance = 5):
        current_date = date.today()
        system_prompt = f"""You are a sentence rephrasing and temporal information extraction expert. Your task is to process an input sentence and return a JSON object containing two fields:

    1.  **rephrased_sentence**:  A rephrased version of the input sentence, with explicit temporal information removed.  The goal is to create a more general question/statement.
    2.  **temporal_decomposition**: A list of time intervals extracted from the input sentence.  Each time interval should be represented as a JSON object with "begin" and "end" keys.  Dates should be in ISO 8601 format: `YYYY-MM-DDTHH:MM:SS`.

    Consider the current date when interpreting relative time expressions (e.g., "past 5 years").

    Current Date: {current_date}

    **Output Format (JSON):**

    ```json
    {{
    "rephrased_sentence": "...",
    "temporal_decomposition": [
        {{
        "begin": "YYYY-MM-DDTHH:MM:SS",
        "end": "YYYY-MM-DDTHH:MM:SS"
        }},
        ...
    ]
    }}
    ```

    **Detailed Instructions:**

    *   **Rephrasing:**
        *   Remove any explicit mentions of dates, years, or time periods.
        *   Maintain the core meaning of the sentence, but make it more general.

    *   **Temporal Decomposition:**
        *   Identify *all* distinct time intervals mentioned or implied in the sentence. Do NOT infer time intervals based on your internal knowledge or external data. If a time interval is not directly stated or strongly implied by the phrasing, do NOT include it.
        *   For each interval, create a JSON object with "begin" and "end" keys.
        *   **Date Formatting:**  Use the full `YYYY-MM-DDTHH:MM:SS` format for all dates and times.
        *   **Year Only:** If only a year is mentioned, use January 1st 00:00:00 for the `begin` and December 31st 23:59:59 for the `end`.
        *   **"Before" and "After":**
            *   "Before
            ```
            *   "Before <YYYY>":  `begin` should be a very early date (e.g., "1700-01-01T00:00:00") and `end` should be the last moment of the year *before* <YYYY> (e.g., "YYYY-12-31T23:59:59").
            *   "After <YYYY>": `begin` should be the first moment of the year *after* <YYYY> (e.g., "YYYY+1-01-01T00:00:00") and `end` should be a far future date (e.g., "2099-12-31T23:59:59") - or a more reasonable future date if contextually appropriate.  Consider using the current year + a buffer (e.g., +5 years) as a reasonable far-future date.
        *   **"Between":** Use the corresponding start year and end year to extract the temporal data.
        *   **No Temporal Information:** If no temporal information is present, `temporal_decomposition` should be an empty list: `[]`.
        *   **Current Date/Year:** Use the date 2024-12-31 as the current date.

    **Example 1:**

    Input Sentence: What is the economy different before and after 2020?

    ```json
    {{
    "rephrased_sentence": "What is the economy different?",
    "temporal_decomposition": [
        {{
        "begin": "1700-01-01T00:00:00",
        "end": "2020-01-01T00:00:00"
        }},
        {{
        "begin": "2020-12-31T23:59:59"",
        "end": "2024-12-31T23:59:59"
        }}
    ]
    }}
    ```

    **Example 2: Range over several years **

    IInput Sentence: What happened to the stock market from 2010 to 2015?

    ```json
    {{
    "rephrased_sentence": "What happened to the stock market?",
        "temporal_decomposition": [
            {{
                "begin": "2010-01-01T00:00:00",
                "end": "2015-12-31T23:59:59"
            }}
        ]
    }}
    ```

    **Example 3: Comparing two years**

    Input Sentence: Compare the economic performance of Hong Kong in 2018 and 2022.

    ```json
    {{
    "rephrased_sentence": "Compare the economic performance of Hong Kong.",
    "temporal_decomposition": [
        {{
            "begin": "2018-01-01T00:00:00",
            "end": "2018-12-31T23:59:59"
        }},
        {{
            "begin": "2022-01-01T00:00:00",
            "end": "2022-12-31T23:59:59"
        }}
    ]
    }}
    ```

    **Example 4:**
    Input Sentence: What is the population of New York City?
    ```json
    {{
    "rephrased_sentence": "What is the population of New York City?",
    "temporal_decomposition": []
    }}
    ```
    """
        user_prompt = f"""Input Sentence: {input_sentence}"""
        attempt = 0
        logging.info({user_prompt})
        while attempt < chance:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    temperature=0.6,
                    max_tokens=1024,
                    top_p=1,
                )   
                answer = response.choices[0].message.content.strip()
                answer = answer.replace('"\n', '",\n')
                answer = answer[answer.rfind("``json"):]
                
                answer = answer[answer.find("{"): answer.rfind("}")+1]
                answer_obj = ast.literal_eval(answer)
                if temporal_process_sentence_validator(answer_obj):
                    if not self._contains_ellipsis(answer_obj):
                        return answer_obj
                else:
                    logging.error(f"{answer_obj}")
            except Exception as e:
                logging.error(f"Error generating time from question': {e}")
                logging.error(f"{response.choices[0].message.content.strip()}")
            attempt += 1
        return None

