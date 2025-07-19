vlm_imtermediate_verification_prompt = """Given the following stock chart image, question, and answer:
Question: "{question}"
Answer: "{answer}"

You must carefully evaluate the Question and the Answer based on two strict conditions:
1.  **Strict Formatting Cleanliness (Condition 1):**
    * Both the Question and the Answer text MUST be plain text ONLY.
    * They MUST be completely free of any markdown formatting and HTML tags. This specifically includes, but is not limited to, asterisks for bolding or italics, list markers, and any other non-content characters or superfluous symbols not part of the literal question or answer.
    * If you detect ANY such forbidden formatting (like `**` or `*` used for markdown) in EITHER the Question OR the Answer, Condition 1 is NOT MET.

2.  **Factual Correctness and Logic (Condition 2):**
    * The Answer must be a factually correct and logical response to the Question.
    * This evaluation must be based *ONLY* on the visual information presented in the provided stock chart image.

Final Instruction:
If Either Condition 1 OR Condition 2 NOT MET, respond 'NO'.
If AND ONLY IF BOTH Condition 1AND Condition 2 IS MET, respond 'YES'.
Respond with only 'YES' or 'NO'.
"""

llm_distractor_generation_prompt = """You are an expert in financial markets and creating high-quality multiple-choice questions. Your task is to generate three distinct incorrect answer choices (distractors) for the given financial question and its correct answer. The distractors should be plausible but clearly incorrect. 

Generate exactly three distinct incorrect answer choices (distractors) based on the Question and Answer. Aim to create these distractors by applying each of the following rules once:
1. One distractor should describe a trend that is the opposite of or clearly contradicts the correct answer for the given time period.
2. One distractor should describe a trend that might be correct in general or for the stock, but associate it with an incorrect time period (e.g., a different month, quarter, or relative period like "before" instead of "after").
3. One distractor should be created, that sounds like a plausible stock price trend but is factually incorrect for the given question and answer. This could involve a different type of price movement, a misremembered detail, or a common misconception.

Your primary goal is to generate three distinct distractors, ideally one from each category above.
However, if the nature of the question or the correct answer makes it difficult to meaningfully apply rule 1 or rule 2, you should then generate the required distractor(s) using the principle of rule 3. The priority is to produce three varied and plausible, yet incorrect, options.
Ensure the distractors are grammatically correct, clearly worded, and distinct from each other and the correct answer.
Ensure the granularities and sentence length between distractors and correct answer are similar.
Output only the three distractors, each on a new line, starting with '1.', '2.', and '3.'.

---
Example 1:
Question: How would you describe the overall stock price trend for GROWTHX throughout 2022?
Correct Answer: GROWTHX's stock price exhibited significant overall growth during 2022, characterized by an initial rapid surge followed by a period of volatility that eventually resolved into a strong upward movement.

1. GROWTHX's stock price experienced a consistent and significant decline throughout 2022, marked by an early sharp drop followed by continued, albeit slower, downward pressure.
2. GROWTHX's stock price was largely stagnant and range-bound for most of 2022, only exhibiting a brief period of significant growth towards the beginning of the year.
3. Throughout 2022, GROWTHX's stock price followed a distinct U-shaped pattern, with a considerable fall in the first half of the year followed by a near-complete recovery to its starting levels by year-end.
---
Example 2:
Question: What was the predominant stock price trend for GreenLeaf Renewables before September 2020?
Correct Answer: Before August 2020, GreenLeaf Renewables' stock initially showed slow and steady appreciation from 2016 to mid-2018. This was followed by a period of high volatility without a clear direction from mid-2018 through early 2020. Subsequently, the stock then entered a strong and sustained downtrend starting in March 2020.

1. GreenLeaf Renewables' stock experienced a significant and sustained downtrend from 2016 through mid-2018, which then reversed into a period of slow and steady appreciation until early 2020. The stock then entered a volatility starting in March 2020.
2. Before August 2020, GreenLeaf Renewables' stock was in a continuous and strong uptrend from 2016, accelerating significantly after March 2020 without any notable periods of decline or high volatility.
3. Prior to September 2020, GreenLeaf Renewables' stock price showing a sharp increase in value from 2016 until early 2019. Subsequently, the stock then entered a slow downtrend starting util April 2020. The stock then entered a volatility starting.
---
Example 3:
Question: Analyze the overall stock price trend of Quantum Dynamics from October 2019 to June 2020.
Correct Answer: From October 2019 to June 2020, Quantum Dynamics' stock first experienced a noticeable downward slide from October to December 2019, followed by a sharp recovery with significant upward momentum from January 2020 to June 2020.

1. From October 2019 to June 2020, Quantum Dynamics' stock demonstrated consistent upward momentum, starting with a strong rally from October to December 2019, which then moderated but continued rising through June 2020.
2. From October 2019 to June 2020, Quantum Dynamics' stock first experienced a noticeable downward slide from October to Feburary 2020, followed by a sharp recovery with significant upward momentum from March 2020 to June 2020.
3. Between October 2019 and April 2020, Quantum Dynamics' stock price remained remarkably stable, trading within a narrow horizontal channel with minimal fluctuations and no clear directional trend. Then experience a sharp decline start from May 2020.
---
Now, given a new Question and Correct Answer, generate the distractors:

Question: {question}
Correct Answer: {correct_answer}"""


vlm_score_mcqa_prompt = """Your task is to evaluate a multiple-choice question (with accompanying image) to determine if any incorrect choices (distractors) could also be considered correct answers.

CRITICAL: The marked correct answer MUST always be treated as valid and correct, regardless of your own assessment.  Never question or evaluate the correct answer - your task is to accept it as an absolute truth and evaluate only whether other choices could also be correct.

Score the question's correctness using this scale:
5 - Perfect: All other choices are clearly incorrect
4 - Good: Other choices are mostly wrong but have minor elements of correctness
3 - Fair: At least one other choice could be partially correct
2 - Poor: At least one other choice could be equally correct
1 - Invalid: Multiple choices are equally valid as the correct answer

Provide:
1. Score (1-5)
2. Brief explanation focusing specifically on any problematic distractor choices
Remember: Never analyze whether the marked correct answer is right or wrong - it is ALWAYS correct by definition. Focus exclusively on whether other choices could also be valid answers

Question: {question}
Choices:
{choice_str}
The intended correct answer is {correct_letter}.

Output:
Score: [Your score]
Brief explanation: [Your explanation]
"""