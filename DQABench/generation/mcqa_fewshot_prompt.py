GEN_QA_PROMPT_TEMPLATE = """
Here are a couple of examples of how to analyze the chart:

--- Example 1 ---
{example_1}
--- End Example 1 ---

--- Example 2 ---
{example_2}
--- End Example 2 ---

--- Example 3 ---
{example_3}
--- End Example 3 ---

Analyze the provided stock price chart for {stock_name}{ticker} covering {time_range_str}.
Based *Only* on the visual trends in this chart, generate one open-ended question and a concise, factual answer related to the '{query_type}' category.

For the question statement, please do not mention "base on the provided image". For the answer, describe the trend instead of detail stock price.

[Your actual image to be analyzed would be processed here by the VLM]
ticker: {ticker}
range_str: {range_str}
query_type: {query_type}

Output with following format with any additional text/symbols:
Question: [Your generated question]
Answer: [Your generated answer]
"""


TIME_PERIOD_EXAMPLE = [
"""
Hypothetical Chart Scenario: The stock price chart for "HLTHCO" for the entire year 2022. The price is relatively stable for the first quarter. Then, a steep, nearly vertical drop is visible over a short period, traversing a considerable portion of the Y-axis range. Following this sharp decline, the price line shows a very slow, almost flat, marginal upward drift for the remaining months, with this recovery being visually very small compared to the preceding drop.
ticker: HLTHCO
range_str: 2022
query_type: Specific Time Period (Year)

Question: Describe the dominant stock price behavior observed for HLTHCO in 2022.
Answer: The stock experienced a visually sharp and substantial decline early in the second quarter of 2022. Subsequently, the price showed a very slight and gradual upward tendency for the remainder of the year, recovering only a minor fraction of the earlier loss.
""",
"""
Hypothetical Chart Scenario: The chart for ticker GROWTHX, a smaller cap stock, covers its performance in 2022. It started the year valued around \$15. The first quarter saw a rapid surge to nearly \$30. The second quarter experienced a sharp pullback to \$20, followed by a volatile but generally upward trajectory, closing the year near \$40.
ticker: GROWTHX
range_str: 2022
query_type: Specific Time Period (Year)

Question: How would you describe the overall stock price trend for GROWTHX throughout 2022?
Answer: GROWTHX's stock price exhibited significant overall growth during 2022, characterized by an initial rapid surge followed by a period of volatility that eventually resolved into a strong upward movement.
"""
,
"""
Hypothetical Chart Scenario: The chart for ticker GREENCO throughout 2015 shows a consistent upward incline. While the line appears to slope upwards clearly, the actual price movement for the entire year was from approximately $150 to $165, indicating a relatively contained overall increase.
ticker: GREENCO
range_str: 2015
query_type: Specific Time Period (Year)

Question: What was the general stock price progression for GREENCO during 2015?
Answer: The stock price for GREENCO demonstrated a slow and continuous rise throughout 2015, reflecting a modest overall increase in its value.
"""
]

BEFORE_YEAR = [
"""
Hypothetical Chart Scenario: GAMMA Innovations experienced a period of high volatility from 2015 to 2016, where its stock price saw significant fluctuations but ended these two years with no clear overall directional change from its starting point. Throughout 2017, there was an almost imperceptible, very slight upward drift in its price. This was then followed by a consistent and moderate upward movement from the beginning of 2018 through to the end of 2019.
ticker: GMIN
range_str: 2015 to 2019, before 2020
query_type: Before (Year Anchor)

Question: What was the predominant stock price trend for GAMMA Innovations before 2020?
Answer: GAMMA Innovations' stock was marked by high volatility without a clear net directional change during 2015-2016. This was followed by a phase of almost imperceptible, very slight growth in 2017, transitioning into a consistent and moderate upward trend throughout 2018 and 2019.
"""
,
"""
Hypothetical Chart Scenario: AGROPLUS Solutions (AGPS) stock remained largely flat with only minor fluctuations and a very slight overall price increase from 2012 to mid-2014, indicating a period of stagnation. Starting in late 2014, the stock began a significant and sustained upward trajectory. This strong growth phase was characterized by consistent, substantial price appreciation that continued robustly through to the end of 2018, resulting in a manifold increase in its value over this latter part of the period.
ticker: AGPS
range_str: 2012 to 2018, before 2019
query_type: Before (Year Anchor)

Question: Describe the dominant stock price behavior observed for AGROPLUS Solutions (AGPS) for the timeframe before 2019.
Answer: AGROPLUS Solutions' stock price (2012-2018) initially showed a period of stagnation with minimal change (2012-mid 2014). This was then followed by a significant and sustained upward trend, marked by strong and consistent price appreciation from late 2014 through the end of 2018.
"""
,
"""
Hypothetical Chart Scenario: For AEROSTREAM Dynamics, the stock price from 2018 to 2019 was characterized by high volatility, with frequent peaks and troughs but no clear overall upward or downward movement. In 2020, the stock price stabilized and began a slow but steady incline.
ticker: ASD
range_str: 2018 to 2020, before 2021
query_type: Before (Year Anchor)

Question: What was the general stock price behavior for AEROSTREAM Dynamics prior to the year 2021?
Answer: Prior to 2021, AEROSTREAM Dynamics' stock price exhibited high volatility without a clear directional trend during 2018 and 2019, followed by a stabilization and a slow, steady incline throughout 2020.
"""
]

AFTER_YEAR = [
"""
Hypothetical Chart Scenario: The stock price for INNOVATECH (ticker: INVT) was relatively flat throughout 2019 and 2020, hovering around $50. Starting in early 2021, the stock price began a steady and substantial increase, reaching $120 by the end of 2021 and continuing its upward trend throughout 2022, closing the year at $180.
ticker: INVT
range_str: 2019 to 2022, after 2018
query_type: After (Year Anchor)

Question: What was the predominant stock price trend for INNOVATECH (INVT) in the period after 2018?
Answer: After being relatively flat in 2019 and 2020, INNOVATECH (INVT) stock experienced a significant and sustained uptrend in 2021, more than tripling in value by the end of 2022.
""",
"""
Hypothetical Chart Scenario: CONSOLIDATED MOTORS (ticker: CNMO) maintained a strong stock price, generally above $150, through 2019. After 2019, its stock price experienced a sharp and significant decline throughout 2020, bottoming out near $80. In 2021 and 2022, the price showed some recovery, often with notable fluctuations, to trade in the $100-$120 range, but it did not approach its pre-2020 levels.
ticker: CNMO
range_str: 2020 to 2022, after 2019
query_type: After (Year Anchor)

Question: Describe the predominant stock price trend for CONSOLIDATED MOTORS (CNMO) after 2019.
Answer: In the period after 2019, CONSOLIDATED MOTORS (CNMO) stock underwent a substantial initial downturn, followed by a partial and somewhat volatile recovery that failed to reclaim its previous price levels.
""",
"""
Hypothetical Chart Scenario: "TECHNOVA Solutions" (ticker: TCVA) was trading at approximately $40 at the end of 2015. Beginning in early 2016, the stock embarked on a strong and sustained upward trajectory, reaching $150 by mid-2018. Following this peak, TCVA entered a phase of gradual decline with some volatility, falling to around $90 by late 2020. From 2021 to 2022, the stock price stabilized, trading mostly sideways in the $85-$100 range.
ticker: TCVA
range_str: 2016 to 2022, after 2015
query_type: After (Year Anchor)

Question: Describe the dominant stock price behavior observed for TECHNOVA Solutions (TCVA) after 2015.
Answer: Start from 2016, TECHNOVA Solutions (TCVA) initially underwent a significant upward surge, more than tripling its value by mid-2018. Subsequently, it experienced a period of gradual decline with notable fluctuations until late 2020, after which the price movement largely stabilized and traded sideways through 2022.
"""
]

BEFORE_MONTH = [
"""
Hypothetical Chart Scenario: GreenLeaf Renewables (GRLF) stock price exhibited a slow but steady appreciation from 2016 until mid-2018. From mid-2018 through 2019, it entered a period of high volatility, with several sharp peaks and troughs but no clear overall directional trend. This volatility persisted into early 2020. However, from March 2020, the stock began a consistent and strong downward trend that persisted month over month.
ticker: GRLF
range_str: 2016-01 to 2020-08, before September 2020
query_type: Before (Month Anchor)

Question: What was the predominant stock price trend for GreenLeaf Renewables before September 2020?
Answer: Before August 2020, GreenLeaf Renewables' stock initially showed slow and steady appreciation from 2016 to mid-2018. This was followed by a period of high volatility without a clear direction from mid-2018 through early 2020. Subsequently, the stock then entered a strong and sustained downtrend starting in March 2020.
"""
,
"""
Hypothetical Chart Scenario: AlphaBuild Construction materials (ABCM) saw its stock price generally increase from 2019 to 2021, though with some notable short-term dips and recoveries, indicating moderate volatility. Throughout 2022, the price plateaued, maintaining its gains from the previous years but without significant further growth. In 2023, a steady, moderate decline began and continued into the first quarter of 2024 (January-March). In April 2024, the stock price stabilized, showing very little movement. This stability persisted through May 2024, but the price began to slowly increase in June 2024.
ticker: ABCM
range_str: 2019-01 to 2024-06, before July 2024
query_type: Before (Month Anchor)

Question: Characterize the stock price trend for AlphaBuild Construction materials prior to June 2024.
Answer: AlphaBuild Construction materials' stock generally increased from 2019 to 2021 with moderate volatility. It then plateaued throughout 2022. A steady, moderate decline commenced in 2023 and persisted through the first quarter of 2024, followed by price stabilization with minimal movement in April and May 2024.
"""
,
"""
Hypothetical Chart Scenario: OLDWORLD UTILITIES (ticker: OWU) had a relatively stable stock price, trading in a narrow band between $40 and $45 for several years. Starting in late 2018, the stock began a gradual but consistent decline. This downward trajectory continued through the first quarter of 2019, with the price dipping below $35 by March 2019.
ticker: OWU
range_str: 2015-01 to 2019-03
query_type: Before (Month Anchor)

Question: What was the predominant stock price trend for OLDWORLD UTILITIES (OWU) in the period before April 2019?
Answer: OLDWORLD UTILITIES (OWU) initially maintained a stable stock price from 2015 to 2018, followed by a clear and consistent downward trend in the months leading up to April 2019.
"""
]

AFTER_MONTH = [
"""
Hypothetical Chart Scenario: The stock price of ALPHACORE DYNAMICS (ACD) was relatively flat in the first few months of 2018. After March 2018, the price initiated a consistent upward movement from around $50, steadily climbing to reach $85 by mid-2020. Following this peak, the stock price entered a phase of gradual decline, falling to $70 by early 2022, and then continued a slower downward drift to around $60 by the end of 2023.
range_str: 2018-04 to 2023-12, after March 2018
query_type: After (Month Anchor)

Question: What was the prevailing trend of the ALPHACORE DYNAMICS (ACD) stock price after March 2018?
Answer: After March 2018, ALPHACORE DYNAMICS (ACD) stock price initially showed a consistent upward trend for over two years, peaking in mid-2020. Subsequently, the price entered a multi-year period of gradual decline that continued through December 2023.
"""
,
"""
Hypothetical Chart Scenario: After July 2019, the stock, starting from approximately $210, began a steep and rapid appreciation, reaching $350 by early 2021. It then traded in a wide range, generally between $320 and $370, for about a year. From early 2022, a distinct downward trend emerged, with the price falling consistently to $250 by December 2024.
range_str: 2019-08 to 2024-12, after July 2019
query_type: After (Month Anchor)

Question: What was the prevailing trend of the QUANTUM SYSTEMS (QSYS) stock price after July 2019?
Answer: After July 2019, QUANTUM SYSTEMS (QSYS) stock price first exhibited a steep and rapid appreciation until early 2021. This was followed by approximately a year of trading within a wide range. Starting in early 2022, the stock entered a distinct and consistent downward trend that persisted through December 2024.
"""
,
"""
Hypothetical Chart Scenario: Start from November 2017, starting from a low of $15, the stock price reversed and began a sustained, multi-year upward trend. This climb was mostly steady, with the price reaching $45 by late 2020. From early 2021 through December 2022, the stock price largely consolidated, moving sideways with fluctuations between $40 and $50.
range_str: 2017-11 to 2022-12, after October 2017
query_type: After (Month Anchor)

Question: What was the prevailing trend of the SUMMIT MINERALS (SMN) stock price after October 2017?
Answer: SUMMIT MINERALS (SMN) stock price entered a sustained upward trend start from November 2017,  for approximately three years, peaking in late 2020. Following this, the stock price largely consolidated its gains, exhibiting a sideways movement with fluctuations through December 2022.
"""
]

YEAR_INTERVAL = [
"""
Hypothetical Chart Scenario: From 2016 to 2018, TECHSOLUTIONS Inc. saw its stock price initially surge dramatically throughout 2016. This was followed by a period of high volatility with significant peaks and troughs but no clear upward or downward direction during 2017. Finally, in 2018, the stock price entered a consistent downward trend, losing much of the earlier gains.
ticker: TSLI
range_str: 2016 to 2018
query_type: Time Interval (Years)

Question: Analyze the overall stock price trend of TECHSOLUTIONS Inc. from 2016 to 2018.
Answer: From 2016 to 2018, TECHSOLUTIONS Inc.'s stock first experienced a dramatic surge in 2016, then underwent a period of high volatility without a clear trend in 2017, and finally entered a consistent downward trend in 2018.
"""
,
"""
Hypothetical Chart Scenario: From 2019 to 2021, GHMCorp's stock showed significant growth in 2019, followed by a year of high volatility in 2020 where it experienced sharp ups and downs but ultimately ended the year slightly above its start. In 2021, the stock resumed a path of steady, moderate appreciation.
ticker: GHMC
range_str: 2019 to 2021
query_type: Time Interval (Years)

Question: Analyze the overall stock price trend of GHMCorp from 2019 to 2021.
Answer: GHMCorp's stock initially experienced significant growth in 2019. This was followed by a period of high volatility with notable fluctuations in 2020, though it concluded that year slightly higher. The trend then shifted to steady, moderate appreciation throughout 2021, indicating an overall positive trajectory across the three-year period despite the interim volatility.
"""
,
"""
Hypothetical Chart Scenario: PIONEER PHARMACEUTICALS (ticker: PPH) was trading at approximately $200 per share at the beginning of 2020. Throughout 2020, the stock price showed a consistent, albeit slow, decline, reaching around $180 by the end of the year. This downward trajectory continued through 2021, with the price falling more steeply in the second half of the year to close near $140. In 2022, the stock price stabilized in the first quarter, fluctuating between $135 and $145, but then resumed its decline, ending the year around $110.
ticker: PPH
range_str: 2020 to 2022
query_type: Time Interval (Years)

Question: Analyze the overall stock price trend of PIONEER PHARMACEUTICALS (PPH) from 2020 to 2022.
Answer: Between 2020 and 2022, PIONEER PHARMACEUTICALS (PPH) stock exhibited a consistent overall downward trend, starting with a slow decline in 2020, accelerating its fall in 2021, and continuing the decline in 2022 despite a brief period of stabilization.
"""
]

MONTH_INTERVAL = [
"""
Hypothetical Chart Scenario: BRIGHTSTAR RENEWABLES (ticker: BSR) stock price exhibited a consistent upward movement from September 2021, starting at $110 and climbing to $145 by the end of January 2022. In February 2022, the stock experienced a minor pullback, with prices drifting down to the $135-$140 range but still maintaining most of its recent gains.
ticker: BSR
range_str: 2021-09 to 2022-02
query_type: Time Interval (Months)

Question: Analyze the overall stock price trend for BRIGHTSTAR RENEWABLES (BSR) from September 2021 to February 2022.
Answer: During the period from September 2021 to February 2022, BRIGHTSTAR RENEWABLES (BSR) predominantly showed a strong upward trend, followed by a slight moderation or minor pullback in the final month.
""",
"""
Hypothetical Chart Scenario: BioScaffold Dynamics (BSDN) stock experienced a strong bull run throughout 2019. This continued into early 2020. From March 2020 to October 2020, the stock price entered a pronounced and consistent downward trend, losing value month over month. There were minor upward corrections within this period, but the overarching trend was clearly negative. From November 2020, the stock price started to bottom out and showed signs of stabilization.
ticker: BSDN
range_str: 2020-03 to 2020-10
query_type: Time Interval (Months)

Question: Describe the stock price trend for BioScaffold Dynamics from March 2020 to October 2020.
Answer: From March 2020 to October 2020, BioScaffold Dynamics' stock was in a pronounced and consistent downward trend.
"""
,
"""
Hypothetical Chart Scenario: Quantum Dynamics (QDY) saw varied performance in late 2019 and early 2020. After a period of stability in mid-2019, its stock price began a noticeable downward slide starting October 2019, which continued through December 2019. In January 2020, the trend reversed, and QDY embarked on a sharp recovery, with significant upward momentum that lasted until April 2020. Following April 2020, the price stabilized at this new higher level for a few months.
ticker: QDY
range_str: 2019-05 to 2020-06
query_type: Time Interval (Months)

Question: Analyze the overall stock price trend of Quantum Dynamics from October 2019 to June 2020.
Answer: From October 2019 to June 2020, Quantum Dynamics' stock first experienced a noticeable downward slide from October to December 2019, followed by a sharp recovery with significant upward momentum from January 2020 to June 2020.
"""
]