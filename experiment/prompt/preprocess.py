def generate_backgroun(news_article):
    system_prompt = """
You are a helpful assistant tasked with extracting key information from a news article. You will be provided with the preceding text of the news article. Your task is to provide the following three pieces of information:

1.  Abstract: A concise summary (1-2 sentences) of the news article's main topic, including *where* and *when* the article was likely published, if discernible from the text. This should briefly describe the event or subject of the news.

2.  Estimate Document Create Date: A single, specific date and time that best represents the estimated *publication date* of the news article.  Prioritize an explicitly mentioned publication date. If no explicit publication date is available, infer a likely date based on the content of the article (e.g., dates of events mentioned, phrases like "yesterday," "today," "last week").  If inferring, briefly explain your reasoning (1-2 sentences max) appended to the date. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS). If no date can be determined or reasonably estimated, explain why.

-Steps-

1. Initial Reading and Comprehension: Read the entire `<NEWS ARTICLE>` carefully.  Focus on understanding the overall topic, the key players involved, and the general sequence of events.  Do *not* try to extract specific dates or times yet.  Just get a good grasp of the "story" the news article is telling.

2. High-Level Rumination: Before attempting detailed extraction, take a moment to consider these two key questions. Formulate *initial, tentative* answers in your internal reasoning:
    - Publication Date:  Based on the overall content, when do you *think* this article was likely published?  Are there any immediate clues (explicit dates, references to recent events) that suggest a publication timeframe?
    - Event Time Interval: What is the main event or subject being discussed?  Does the article suggest a specific timeframe for this event (e.g., a single day, a period of weeks, an ongoing situation)? What part of the article indicate the event time?
    Keep these initial thoughts in mind as you proceed to the detailed extraction steps.

3. Identify Potential Event Phrases: Scan the article again, looking for phrases that *suggest* an event or subject.  Look for action verbs, descriptions of incidents, announcements, or reports.  List these potential event phrases (e.g., "the company announced a merger," "the earthquake struck," "the festival will take place").

4. Determine the Main Event/Subject: From the list of potential event phrases (Step 3), choose the *one* that best represents the *primary* focus of the article.  If there are multiple events, select the one that is most central to the article's overall narrative. Briefly explain (1 sentence) why you selected this event/subject over others.

5. Extract Explicit Dates and Times: Examine the text *surrounding* the chosen main event/subject (Step 4).  Identify *any* explicit mentions of dates and times related to the event.  Note these down, even if incomplete (e.g., "June 2023," "Monday morning").  Record these in a temporary working space, *not* in the final JSON output yet.

6. Analyze Relative Time References: Look for relative time references *directly connected* to the main event/subject (e.g., "yesterday," "next week," "three days prior").  If the article *also* contains an explicit date (like a publication date), use that explicit date to convert these relative references into absolute dates and times. If *no* explicit date can help resolve the relative references, do *not* try to guess; instead, note that the relative reference exists but cannot be resolved.

7. Estimate Document Create Date: Look for an explicit publication date within the article.  If found, use this.  If not found, try to infer a likely publication date *based solely on the article's content*.  Use clues like dates of events mentioned, or relative time references (if resolvable).  If inferring, briefly explain your reasoning (1-2 sentences). If no date can be determined or reasonably estimated, explain why.

8. Construct the JSON Output: Finally, assemble all the extracted information into the specified JSON string:

    {{
        "abstract": "...",
        "estimate_doc_create_date": "YYYY-MM-DDTHH:MM:SS"
    }}

-Examples-

Example 1:
<NEWS_ARTICLE>
Banco Bradesco Sa ( BBD ) will begin trading ex-dividend on October 03, 2016. A cash dividend payment of $0.196366 per share is scheduled to be paid on March 15, 2017. Shareholders who purchased BBD prior to the ex-dividend date are eligible for the cash dividend payment.
</NEWS_ARTICLE>
Output:
```json
{{
  "abstract": "This news article announces the upcoming ex-dividend date for Banco Bradesco Sa (BBD) stock.",
  "estimate_doc_create_date": "2016-09-30T00:00:00"
}}
```

Example 2:
<NEWS_ARTICLE>
Bid the lovely summer calm adieu. Most ETF investors watched their portfolios coast along for roughly two months until the financial markets pulled back and volatility spiked, all over again.
Stocks, bonds and even traditional safe havens came under selling pressure from an enigmatic Fed in recent trading sessions. As the Federal Reserve kept the market guessing about its next move on interest rates, portfolios hit a squall.
SPDR S&P 500 ( SPY ), a proxy for the broad U.S. market, ended 2.4% lower in the month ended Sept. 13 while setting fresh all-time highs along the way. The largest ETFs that invest in foreign-developed and emerging markets posted losses of 1.7% and 3.6% over the same period, respectively.
IShares Core U.S. Aggregate Bond ( AGG ) gave up 0.8%. SPDR Gold Shares ( GLD ), a commodity ETF, lost 1.3%.
And the shine came right off PureFunds ISE Junior Silver ( SILJ ), whose 210.7% gain year to date is unrivaled among nonleveraged exchange traded funds. It crumbled 17.9% in the past month, a laggard among ETFs.
</NEWS_ARTICLE>
Output:
```json
{{
  "abstract": "This news article reports on a recent pullback and volatility spike in the financial markets impacting various ETFs, attributed to uncertainty surrounding Federal Reserve interest rate decisions. The report includes performance data for the period ending September 13th, likely published shortly after that date in a US financial context.",
  "estimate_doc_create_date": "2016-09-16T00:00:00"
}}
```
"""
    
    user_prompt = f"""
-Real Data-
<NEWS_ARTICLE>
{news_article}
</NEW_ARTICLE>
Output:
"""


def time_int_extract(news_background: dict, chunk_text: str):
    system_prompt = f"""
You are a helpful assistant tasked with extracting time intervals, associated event descriptions, and all related entity involved in those events from news text chunks. You will receive the news's background (`news_background`) and a text chunk (`text_chunk`).

-Goal-
Given a text chunk with an estimated news background, extract all relevant events, their time intervals, and key entities involved in those events.

-Steps-
1.  Understand the Context:
    - Carefully read the `news_background` object. Note the `estimate_doc_create_date`. This is your reference date for later calculations.
    - Carefully read the `text_chunk`. Focus on understanding the main topic and the sequence of events described. *Do not extract information yet.*

2.  Ruminate on Events, Times, and Entities:
    - Now, think back on the `text_chunk` you just read. *Mentally* list the key events that were mentioned. For each event, ask yourself:
        - Who or what is the main actor or subject performing or involved in this event?* (This is the potential entity).
        - Was a specific time or date mentioned for this event?*
        - Was a relative time mentioned (e.g., "last week", "next month")?*
        - Can I infer a time period, even if it's broad (e.g., "during the summer of 2023")?*
    - Do not convert to ISO 8601 yet.* This step is about recalling and associating events, entities, and potential timeframes.
    - If you cannot associate a time *and* a clear entity with an event, it likely won't be included in the final output, but keep it in mind for now.

3.  Initial Time, Event, and Entities Scan (Detailed):
    - Now, go back to the `text_chunk` and systematically scan for *any* phrases that indicate a date or time. These can be:
        - Explicit Dates/Times: (e.g., "2024-03-15", "10:30 AM", "December 1st, 2023")
        - Relative Dates/Times: (e.g., "last Tuesday", "next year", "two weeks ago")
        - Partial Dates: (e.g., "January 2024", "2022")
    - For *each* time phrase found:
        - Identify the corresponding event or activity directly associated with it in the text. The event should be described *in the same sentence or an immediately adjacent sentence*.
        - Identify the *related entities* (person, organization, subject) related to that specific event, as stated in the surrounding text.
    - Create a *preliminary* list of potential `(event_description_snippet, entity_names, time_phrase)` tuples.

4.  Convert to ISO 8601 Time Intervals and Refine Entities:
    - For *each* potential `(event_description_snippet, entity_names, time_phrase)` tuple from Step 3:
        - Convert the `time_phrase` to an ISO 8601 time interval object: `{{"begin": "YYYY-MM-DDTHH:MM:SS", "end": "YYYY-MM-DDTHH:MM:SS"}}`.
        - Explicit Dates/Times: Use the exact date and time provided.
        - Relative Dates/Times: Calculate the date and time *relative to the `estimate_doc_create_date` from Step 1*.
        - Partial Dates:
            - For the `begin` value, use the *start* of the year/month (e.g., "2023" becomes "2023-01-01T00:00:00").
            - For the `end` value, use the *end* of the year/month (e.g., "2023" becomes "2023-12-31T23:59:59"). Always use "23:59:59" for the end of a day, month, or year.
        - Refine the `entity_name` to be the most specific identifier available in the text (e.g., "The CEO" might become "Jane Doe" if her name is mentioned in connection). Use the proper name if available.

5.  Final Event Description, Entities, and Output:
    - For each refined `(event_description_snippet, entity_names, iso_interval)` tuple (potentially after merging):
        - Write a short, descriptive sentence summarizing the event (`event`). *Using only information found in the `text_chunk`*. Ensure the sentence clearly links the entity to the action.
        - Finalize the `entities` list. This should be the name or identifier identified in Step 4.
        - Create the final JSON object for the event:
            ```json
            {{
              "event": "...", // Your concise sentence
              "entity": ["...", "..."], // The identified entities
              "event_time_interval": {{ "begin": "...", "end": "..." }} // The ISO 8601 interval
            }}
            ```
    - Combine all event JSON objects into a single JSON array named `events`.

6.  JSON Output:
    - Return *only* the JSON string representing the `events` array. Do not include any additional text or explanation. The JSON should be valid and parsable. If no events with associated time intervals and entities were found, return an empty JSON array: `{{ "events": [] }}`.


-Variables-

- `news_background` (object): Contains `"abstract"`, `"estimate_doc_create_date"` (string, YYYY-MM-DDTHH:MM:SSZ). This provides a reference point for relative time expressions. It is *not* a bounding interval.
- `text_chunk` (string): The text excerpt to analyze.


-Examples-
Example 1:
news_background: {{
    "abstract": "This news article announces the upcoming ex-dividend date for Banco Bradesco Sa (BBD) stock.",
    "estimate_doc_create_date": "2016-10-02T00:00:00"
}}
text_chunk:
<TEXT_CHUNK>
Banco Bradesco Sa ( BBD ) will begin trading ex-dividend on October 03, 2016. A cash dividend payment of $0.196366 per share is scheduled to be paid on March 15, 2017. Shareholders who purchased BBD prior to the ex-dividend date are eligible for the cash dividend payment.
</TEXT_CHUNK>

Output:
```json
{{
  "events": [
    {{
      "event": "Banco Bradesco Sa (BBD) will begin trading ex-dividend.",
      "entity": ["Banco Bradesco Sa (BBD)"],
      "event_time_interval": {{
        "begin": "2016-10-03T00:00:00",
        "end": "2016-10-03T23:59:59"
      }}
    }},
    {{
      "event": "A cash dividend payment of $0.196366 per share from Banco Bradesco Sa (BBD) is scheduled to be paid.",
      "entity": ["Banco Bradesco Sa (BBD)"],
      "event_time_interval": {{
        "begin": "2017-03-15T00:00:00",
        "end": "2017-03-15T23:59:59"
      }}
    }}
  ]
}}
```

Example 2:
news_background: {{
  "abstract": "This news article reports on the performance of stock index ETFs (like SPY) and commodity ETFs (like DBC) in the US market on Friday, December 30, 2016, the final trading day of the year, and summarizes their performance for 2016. The article was likely published on or shortly after December 30, 2016.",
  "estimate_doc_create_date": "2016-12-30T00:00:00"
}}
text_chunk:
<TEXT_CHUNK>
SPDR S&P 500 ( SPY ) nudged 0.2% lower on the stock market today in morning trade. This exchange traded fund, a proxy for the broad U.S. market, set an all-time high of 228.34 on Dec. 13.
Amid higher prices in precious and industrial metals, as well as corn, wheat and cotton, PowerShares DB Commodity Tracking ( DBC ) added 0.2% in morning trade as it rose for a fifth consecutive day.
This $2.55 billion, broadly diversified commodity ETF has a 18.34% gain so far in 2016, a period over which crude oil, silver, gold, and copper staged impressive rallies at various times.
By comparison, SPY posted a 12.4% gain this year through Dec. 29 and iShares Core U.S. Aggregate Bond ( AGG ) scored a 2.2% gain.
DBC follows a rules-based index composed of futures contracts on 14 of the most heavily traded and important physical commodities in the world. They include gasoline, heating oil, Brent crude oil, WTI crude oil, gold, wheat, corn, soybeans, sugar, natural gas, zinc, copper, aluminum and silver.
</TEXT_CHUNK>
Output:
```json
{{
  "events": [
    {{
      "event": "SPDR S&P 500 (SPY) moved lower while PowerShares DB Commodity Tracking (DBC) moved higher in morning trade.",
      "entity": [
        "SPDR S&P 500 (SPY)",
        "PowerShares DB Commodity Tracking (DBC)"
      ],
      "event_time_interval": {{
        "begin": "2016-12-30T08:00:00",
        "end": "2016-12-30T12:00:00"
      }}
    }},
    {{
      "event": "SPDR S&P 500 (SPY) set an all-time high.",
      "entity": [
        "SPDR S&P 500 (SPY)"
      ],
      "event_time_interval": {{
        "begin": "2016-12-13T00:00:00",
        "end": "2016-12-13T23:59:59"
      }}
    }},
    {{
      "event": "PowerShares DB Commodity Tracking (DBC) rose for a fifth consecutive day.",
      "entity": [
        "PowerShares DB Commodity Tracking (DBC)"
      ],
      "event_time_interval": {{
        "begin": "2016-12-26T00:00:00",
        "end": "2016-12-30T23:59:59"
      }}
    }},
    {{
      "event": "PowerShares DB Commodity Tracking (DBC) achieved a gain in 2016.",
      "entity": [
        "PowerShares DB Commodity Tracking (DBC)"
      ],
      "event_time_interval": {{
        "begin": "2016-01-01T00:00:00",
        "end": "2016-12-30T23:59:59"
      }}
    }},
    {{
      "event": "SPDR S&P 500 (SPY) and iShares Core U.S. Aggregate Bond (AGG) scored gain in 2016.",
      "entity": [
        "SPDR S&P 500 (SPY)",
        "iShares Core U.S. Aggregate Bond (AGG)"
      ],
      "event_time_interval": {{
        "begin": "2016-01-01T00:00:00",
        "end": "2016-12-29T23:59:59"
      }}
    }}
  ]
}}
```
"""
    if news_background:
        news_background_str = json.dumps(news_background, indent=4)
        user_prompt = f"""
-Real Data-
news_background:
{news_background_str}
text_chunk:
<TEXT_CHUNK>
{chunk_text}
</TEXT_CHUNK>
Output:
"""
    else: 
        user_prompt = f"""
-Real Data-
news_background:
{{ }} // no background
text_chunk:
<TEXT_CHUNK>
{chunk_text}
</TEXT_CHUNK>
Output:
"""