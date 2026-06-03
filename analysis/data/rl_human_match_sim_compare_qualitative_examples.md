# Qualitative Examples

## Susceptibility Proposition Pair
- Corpus: vanilla_llm_target
- Initial-belief bin: [0.00, 0.10]
- High-movement proposition: The US should have mandatory national service.
- Low-movement proposition: Colleges should consider race as a factor in admissions to ensure diversity.
- Mean total delta (high/low): +0.4668 / +0.0018
- Mean initial belief (high/low): 0.0632 / 0.0632
- Mean final belief (high/low): 0.5300 / 0.0650
- Rounds (high/low): 2 / 2
- Mean total delta gap: +0.4650

## Naive Bad-Case
- Corpus: vanilla_llm_target
- Proposition: Online learning is a suitable replacement for traditional in-person education.
- Stance: opposes
- Matched rounds: 3
- Mean abs total delta (non-naive / naive): 0.0077 / 0.6096
- Mean initial belief (non-naive / naive): 0.1804 / 0.1804
- Mean final belief (non-naive / naive): 0.1867 / 0.7900
- Naive excess (mean abs total delta): +0.6019
- Relative to non-naive baseline: +7792.5%
