# Flag-similar-events autoresearch — findings

Baseline F1 0.3158 -> final 0.8119 (+157%).

## Winning rule (committed)
tau=0.10, date-overlap gate (t_max=0), field-agreement>=0.5, location gate OFF.

## Why it works
- Lowering tau (not raising it) drove the gains — true dups are low-text-sim; the LLM identifies them by date/location/witness agreement.
- Date gate + field gate restore the precision that low tau gives up.
- Strict location matching backfires (location drift).
- Residual false positives are same-day flap pairs -> require the LLM/semantic tier.
