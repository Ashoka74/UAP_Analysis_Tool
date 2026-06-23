"""Staircase chart of the flag-similar-events autoresearch loop: running-best F1
with labels explaining the feature change behind each improvement."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TSV = sys.argv[1] if len(sys.argv) > 1 else "autoresearch/loop-260618-2053/results.tsv"
OUT = TSV.replace("results.tsv", "improvement_stairs.png")

df = pd.read_csv(TSV, sep="\t", comment="#")
it = df["iteration"].tolist()
metric = df["metric"].tolist()
status = df["status"].tolist()

# running best-so-far (the staircase)
best, b = [], -1.0
for m in metric:
    b = max(b, m)
    best.append(b)

# short feature label for each KEPT improving step
STEP_LABEL = {
    0: "baseline\nτ=0.55, text-only\n→ F1 0.316",
    1: "↓ τ 0.55→0.30\ntrust structure, not text",
    2: "+ date gate",
    4: "+ field-agreement ≥0.5",
    5: "↓ τ 0.30→0.10\ngates now protect precision",
    7: "date window → same-day",
}

ORANGE, GREEN, RED, INK = "#f97316", "#22c55e", "#ef4444", "#e5e7eb"
fig, ax = plt.subplots(figsize=(12.5, 7))
fig.patch.set_facecolor("#0b0b0f")
ax.set_facecolor("#0b0b0f")

# staircase of best-so-far
ax.step(it, best, where="post", color=ORANGE, lw=3, zorder=3, label="best-so-far (kept)")
ax.fill_between(it, best, step="post", color=ORANGE, alpha=0.10, zorder=1)

# measured points: keep vs discard
for i, m, s in zip(it, metric, status):
    if s in ("keep", "baseline"):
        ax.scatter(i, m, s=90, marker="o", color=GREEN, edgecolor="white",
                   zorder=5, label="kept" if i == 1 else None)
    else:
        ax.scatter(i, m, s=110, marker="X", color=RED, edgecolor="white",
                   zorder=5, label="discarded probe" if s == "discard" and i == 3 else None)
        ax.annotate(df["description"][i].split("(")[0].strip(),
                    (i, m), textcoords="offset points", xytext=(6, -16),
                    color=RED, fontsize=7.5, alpha=0.9)

# annotate each rising step with the feature that caused it
prev = -1.0
for i in it:
    if best[i] > prev + 1e-9 and i in STEP_LABEL:
        big = i in (1, 5)
        ax.annotate(
            STEP_LABEL[i] + (f"\n→ F1 {best[i]:.3f}" if big else ""),
            (i, best[i]), textcoords="offset points",
            xytext=(12, 28) if i == 0 else (8, 18 if not big else 26),
            color="white", fontsize=9 if big else 8,
            fontweight="bold" if big else "normal",
            bbox=dict(boxstyle="round,pad=0.35", fc="#1f2937",
                      ec=ORANGE, lw=1.2 if big else 0.7, alpha=0.95),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    prev = best[i]

ax.axhline(best[-1], color=ORANGE, ls=":", lw=1, alpha=0.5)
ax.text(9.05, best[-1], f" final {best[-1]:.3f}", color=ORANGE, va="center", fontsize=10, fontweight="bold")

ax.set_title("Flag-similar-events dedupe — F1 vs LLM gold\nstaircase of kept improvements (+157%)",
             color="white", fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("iteration", color=INK)
ax.set_ylabel("F1 (programmatic full-row  vs  LLM same-event)", color=INK)
ax.set_ylim(0.25, 0.92)
ax.set_xlim(-0.4, 9.9)
ax.set_xticks(it)
for sp in ax.spines.values():
    sp.set_color("#374151")
ax.tick_params(colors=INK)
ax.grid(axis="y", color="#1f2937", lw=0.7)
leg = ax.legend(loc="lower right", facecolor="#111827", edgecolor="#374151", labelcolor=INK, fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150, facecolor=fig.get_facecolor())
print("wrote", OUT)
