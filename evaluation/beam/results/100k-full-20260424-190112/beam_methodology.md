# BEAM Eval Methodology

- Dataset: `Mohammadta/BEAM`
- Split: `100K`
- Conversations evaluated: `20`
- Retrieval: `synapt` transcript-adapter + hybrid recall search
- Answer model: `gpt-4o-mini`
- Judge model: `gpt-4o-mini`
- Max chunks: `5`
- Max tokens: `2000`
- Public BEAM splits currently available: `100K`, `500K`, `1M`
- Public `10M` split is not present in the current Hugging Face release
- Retrieval recall is not reported because BEAM does not expose LOCOMO-style evidence turn IDs

- Started at: `2026-04-25T00:01:13.691183+00:00`
- Finished at: `2026-04-25T01:48:22.450843+00:00`
- Elapsed seconds: `6428.8`

- Average retrieve ms: `6.0`
- Questions evaluated: `400`
- J-score overall: `52.5`
- F1 overall: `22.01`
- Average generate ms: `12354.4`
- Average judge ms: `3675.4`
- Estimated cost USD: `0.1496`
- Prompt tokens: `855070`
- Completion tokens: `35618`