# Day 13 — Retrieval Evaluation

Tested 10 questions against the RAG pipeline, each run twice (with HyDE query rewriting and without), comparing retrieved chunks and generated answers.

## Summary

- 7/10 questions: HyDE and No-HyDE tied — both retrieved the same relevant chunk and gave correct/equivalent answers.
- 2/10 questions: No-HyDE gave a more complete or accurate answer than HyDE, despite both retrieving similar chunks.
- 1/10 question: Both failed identically — a retrieval success but generation failure.
- **HyDE never outright won** on any question in this evaluation.

## Key Findings

1. **On a short, single-topic document, HyDE's hallucinated search text usually didn't matter.** There aren't enough competing chunks in a small doc for a "wrong" hypothetical-answer embedding to actually mislead retrieval — the correct chunk kept getting pulled regardless.

2. **Retrieval succeeding doesn't guarantee correct generation.** On the "one word" question, both HyDE and No-HyDE retrieved the correct chunk, but the LLM still failed to extract the exact quoted word ("Hell") during answer generation — a generation-step failure, not a retrieval-step failure. Worth tracking these separately in future evals.

3. **A wrong guess is arguably worse than an honest "I don't know."** On that same question, HyDE guessed incorrectly ("No") while No-HyDE admitted it didn't know — the second is more useful even though neither was correct.

4. **Ambiguous questions cause both pipelines to struggle equally.** One question was vaguely worded going in, and neither HyDE nor No-HyDE could produce a clean answer — a "bad question" case rather than a system failure, worth flagging separately from genuine retrieval/generation issues.

5. **Identical retrieved context can still produce different quality answers.** On one question, both HyDE and No-HyDE retrieved the same chunk, but one generated a fuller, more detailed answer than the other — pointing to run-to-run generation variance, not just a HyDE effect.

6. **Grounding instructions held up under hallucinated search input.** Even when HyDE's search text was a completely fabricated scenario, the final LLM correctly said "I don't know" for information genuinely absent from the source — confirming the "say I don't know if not found" prompt instruction is robust to noisy retrieval input.

7. **HyDE answers showed more run-to-run variance than No-HyDE when the same questions were re-run.** On a repeat pass, most questions stayed stable, but one HyDE answer dropped previously-included correct details (e.g. omitted the Khadija/Abu Talib/Ta'if causes on the "distress" question, keeping only one cause) while the equivalent No-HyDE answer stayed consistent across both runs. Likely cause: HyDE adds an extra LLM generation step (the hypothetical answer) before retrieval even begins, compounding randomness on top of the final answer generation step. No-HyDE only has one generation step, so it has one less source of variance.

## Conclusion

On this narrow, single-topic test document, HyDE query rewriting provided no measurable retrieval benefit and occasionally introduced pure noise into the search step with no corresponding payoff. Confirms the Day 12 hypothesis with quantified evidence across a 10-question test set, run twice for consistency. The repeat run also surfaced a secondary finding: HyDE's extra generation step introduces more answer-to-answer variance than the simpler No-HyDE path.