# Evals Summary — core loop (9 iterations)

## Key Metrics
- Total iterations: 9 | Kept: 5 | Reverted: 4 | Revert rate: 44%
- Starting metric: 0.3158 | Final metric: 0.8119 | Improvement: +157%
- Longest keep-streak: 2 | Plateau since iter 7 (2 stagnant)

## Trend Analysis
- Progression (best-so-far): 0.316 → 0.674 → 0.681 → 0.681 → 0.682 → 0.804 → 0.804 → 0.812 → 0.812 → 0.812
- Biggest win: iter 1 (+0.3579) — lower τ 0.55→0.30 (recover low-sim positives)
- Biggest loss: iter 6 (-0.0453) — tighten field_min 0.5→0.7
- Diminishing returns: after iter 5, mean kept-delta 0.0080 (vs +0.24 for the first two wins)

## Patterns
- Succeeded: lowering the text-similarity floor (τ 0.55→0.30→0.10) + conjunctive structured gates (date-overlap, field-agreement ≥0.5). Trust structure over text.
- Failed: every precision-over-recall tightening — strict location gate (location drift), field_min 0.6/0.7, high-sim bypass.
- Hotspot: all kept changes in geo_dedup.py::PREDICT_PARAMS (scope held).
- Effectiveness: τ reductions ≫ date gate > field gate ≫ location gate (negative).

## Recommendation
STOP threshold loop (converged, plateau ≥3 iters). Change strategy: swap lexical sim→embedding cosine and grow gold labels over the UNKNOWN band to attack the residual same-day flap FPs (LLM/semantic tier).
