# Day 22 debug notes

Source: `logs/query_logs.jsonl` for document `ea44a068-352a-4ed2-b9d5-51dd629f58cf`, plus a wrong-chunk run in `debug_failures.py`. Pipeline code was not changed.

## 1. Khadija / Abyssinia — weak grounding

Query: "What was Khadija's role in Muhammad's life, and did she go to Abyssinia?"

Logged answer: "Khadija was Muhammad's wife and a crucial source of his financial and emotional support; she did not go to Abyssinia."

Retrieved context (3 chunks):

- Migration to Abyssinia, chunk 1 — the 615 migration and persecution. Does not mention Khadija.
- Attempt to establish himself in Ta'if, chunk 1 — says his wife Khadija died in 619 and was a source of financial and emotional support. Says nothing about Abyssinia.
- Migration to Abyssinia, chunk 2 — Satanic Verses / return from Abyssinia. Does not mention Khadija.

The role half matches the Ta'if sentence. The claim that she did not go to Abyssinia is not in any of the three chunks. The migration passages never name her, and the model treated that absence as a negative fact.

## 2. "Muhammad's difficulties" — retrieval precision on a vague query

Query: "Tell me about Muhammad's difficulties."

Top retrieved chunk was Quraysh delegation to Yathrib, chunk 1 (the rabbis' three questions, a 15-day wait, and distress until Gabriel answered). The other two chunks were from Attempt to establish himself in Ta'if (Khadija and Abu Talib dying in 619, then the Ta'if rejection and stoning).

The Yathrib section is not a hardship passage. It was still ranked first, and the logged answer folded the rabbi questions and the 15-day delay into the list of difficulties. Vague query, unrelated section pulled in beside the real hardship chunks.

## 3. Wrong-chunk robustness — 3/3 refused

`debug_failures.py` sent each question to the model with one chunk from a section that had not been retrieved for that question. Same instruction as `ask()`: answer only from the context, otherwise "I don't know". These calls did not go through `ask()`, so nothing was written to the query cache.

| Question | Only context given | Model answer |
| --- | --- | --- |
| What happened when Muhammad tried to preach in Taif? | Social exclusion of the Banu Hashim, chunk 1 | I don't know |
| How did Abu Talib respond when Muhammad said he would not stop preaching? | Migration to Abyssinia, chunk 1 | I don't know |
| Why did the early Muslims migrate to Abyssinia? | Quraysh delegation to Yathrib, chunk 1 | I don't know |

3/3 returned "I don't know" with an empty citations list. With a single mismatched chunk, the model did not fill in the real event from outside the context.
