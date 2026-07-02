import { useState } from 'react';
import { Info, ChevronDown, ChevronRight, AlertTriangle } from 'lucide-react';

/**
 * Shared interpretive note for the Cramér's V explorer and the XGBoost feature
 * importance view: how to read agreement/disagreement between the two, so a low
 * importance score isn't misread as "no correlation". Collapsed by default.
 */
export function EvidenceGuide() {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-md border border-border/60 bg-raised/40 text-xs">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 px-3 py-2 text-left font-medium text-text-secondary transition-colors hover:text-text-primary"
      >
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        <Info className="h-3.5 w-3.5 text-accent" />
        How to read Cramér's V vs. feature importance
      </button>

      {open && (
        <div className="space-y-3 border-t border-border/60 px-3 py-3 text-text-muted">
          <p>
            They answer <span className="text-text-secondary">different questions.</span> Cramér's V is{' '}
            <span className="text-text-secondary">marginal</span> — do two fields co-occur, pairwise?
            XGBoost importance is <span className="text-text-secondary">conditional</span> — does a field
            predict <em>this</em> target, given all the others? Read a clash as a clue, not a contradiction:
          </p>

          {/* 2×2 quadrant — V (rows) × importance (cols) */}
          <div className="grid grid-cols-[auto_1fr_1fr] gap-px overflow-hidden rounded border border-border/60 bg-border/60 text-[11px]">
            <div className="bg-deep p-2" />
            <div className="bg-deep p-2 font-medium text-text-secondary">High importance</div>
            <div className="bg-deep p-2 font-medium text-text-secondary">Low importance</div>

            <div className="bg-deep p-2 font-medium text-text-secondary">High&nbsp;V</div>
            <div className="bg-surface p-2 text-text-secondary">
              <span className="text-success">Robust, real, useful</span> — strongest evidence.
            </div>
            <div className="bg-surface p-2 text-text-secondary">
              <span className="text-warning">Redundancy / confounding</span> — a correlated feature already
              supplies the signal, so the tree never splits on this one.{' '}
              <span className="inline-flex items-center gap-1 font-medium text-warning">
                <AlertTriangle className="h-3 w-3" /> not “no relationship”.
              </span>
            </div>

            <div className="bg-deep p-2 font-medium text-text-secondary">Low&nbsp;V</div>
            <div className="bg-surface p-2 text-text-secondary">
              <span className="text-accent-bright">Interaction / nonlinear</span> — a conditional effect a
              pairwise contingency table structurally can’t see.
            </div>
            <div className="bg-surface p-2 text-text-secondary">
              Likely no association (with the sparse-cell caveat for&nbsp;V).
            </div>
          </div>

          <p>
            V maps the <span className="text-text-secondary">association structure</span>; importance asks{' '}
            <span className="text-text-secondary">what predicts a target.</span> Both are associational, not
            causal.
          </p>
        </div>
      )}
    </div>
  );
}
