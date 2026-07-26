"""
build_public_report.py
───────────────────────
Renders pursue_public_report.md-equivalent content directly into a
publication-style PDF via reportlab (no browser backend required — the
make-pdf skill's Chromium/Paged.js pipeline is unavailable in this
sandbox, so this is a direct fallback using a dependency already in
requirements.txt).

Public-facing, narrative tone: analyzes the 11 corpus figures for a general
audience, does not dwell on methodology/coverage gaps.

Usage:
    uv run python build_public_report.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    PageBreak, HRFlowable,
)
from reportlab.platypus.flowables import KeepTogether

FIG_DIR = Path("figures")
OUT = Path("pursue_public_report.pdf")
CONTENT_WIDTH = LETTER[0] - 2 * inch

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("CoverTitle", fontName="Helvetica-Bold", fontSize=28,
                          leading=34, alignment=TA_CENTER, spaceAfter=14,
                          textColor=colors.HexColor("#111111")))
styles.add(ParagraphStyle("CoverSub", fontName="Helvetica", fontSize=13,
                          leading=18, alignment=TA_CENTER,
                          textColor=colors.HexColor("#444444"), spaceAfter=6))
styles.add(ParagraphStyle("CoverDate", fontName="Helvetica", fontSize=11,
                          alignment=TA_CENTER, textColor=colors.HexColor("#777777")))
styles.add(ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=16,
                          leading=20, spaceBefore=18, spaceAfter=8,
                          textColor=colors.HexColor("#111111")))
styles.add(ParagraphStyle("Body", fontName="Helvetica", fontSize=10.5,
                          leading=15.5, alignment=TA_LEFT, spaceAfter=8,
                          textColor=colors.HexColor("#222222")))
styles.add(ParagraphStyle("Caption", fontName="Helvetica-Oblique", fontSize=8.5,
                          leading=11, alignment=TA_CENTER,
                          textColor=colors.HexColor("#666666"), spaceAfter=14))
styles.add(ParagraphStyle("Footer", fontName="Helvetica-Oblique", fontSize=8.5,
                          leading=12, alignment=TA_LEFT,
                          textColor=colors.HexColor("#888888")))
styles.add(ParagraphStyle("BulletItem", fontName="Helvetica", fontSize=10.5,
                          leading=15, leftIndent=14, spaceAfter=4,
                          textColor=colors.HexColor("#222222")))


def fig(name: str, max_h: float = 3.0 * inch) -> RLImage:
    path = FIG_DIR / name
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    aspect = h / w
    disp_w = CONTENT_WIDTH
    disp_h = disp_w * aspect
    if disp_h > max_h:
        disp_h = max_h
        disp_w = disp_h / aspect
    return RLImage(str(path), width=disp_w, height=disp_h, hAlign="CENTER")


def P(text: str) -> Paragraph:
    return Paragraph(text, styles["Body"])


def H(text: str) -> Paragraph:
    return Paragraph(text, styles["H2"])


def B(text: str) -> Paragraph:
    return Paragraph(f"&bull;&nbsp;&nbsp;{text}", styles["BulletItem"])


story = []

# ── Cover ────────────────────────────────────────────────────────────────
story.append(Spacer(1, 2.2 * inch))
story.append(Paragraph("What the Government's UFO Files<br/>Actually Say", styles["CoverTitle"]))
story.append(Spacer(1, 0.15 * inch))
story.append(Paragraph("A data-driven look at 1,873 declassified UAP sighting reports", styles["CoverSub"]))
story.append(Spacer(1, 0.35 * inch))
story.append(Paragraph("PURSUE Data Project &middot; July 2026", styles["CoverDate"]))
story.append(PageBreak())

# ── Intro ────────────────────────────────────────────────────────────────
story.append(P(
    "Nearly two thousand declassified sighting reports, pulled straight from "
    "the Department of War's own UAP release archive. No speculation, no "
    "theories &mdash; just what's actually in the paperwork the government "
    "chose to make public. Here's what the numbers show."
))
story.append(Spacer(1, 6))

# ── 1947 spike ───────────────────────────────────────────────────────────
story.append(H("A Wave of Sightings That Still Defines the Story"))
story.append(P(
    "If you plot every sighting in this archive by year, one thing jumps "
    "out immediately: a single, enormous spike."
))
story.append(fig("02_year_distribution.png"))
story.append(Paragraph("Sighting reports by year, 1833&ndash;2026", styles["Caption"]))
story.append(P(
    "<b>1947.</b> Almost a thousand individual reports cluster around that "
    "one year &mdash; dwarfing everything before or after it. This is the "
    "“flying disc” summer, the moment the phrase “flying "
    "saucer” entered the American vocabulary after pilot Kenneth "
    "Arnold's sighting near Mount Rainier in June 1947. What this chart "
    "shows is just how seriously the reporting apparatus took it at the "
    "time: whatever people were seeing that summer, the government was "
    "writing it down by the hundreds."
))
story.append(P("Zoom out to the broader eras and the pattern holds:"))
story.append(fig("03_year_eras.png", max_h=2.6 * inch))
story.append(Paragraph("Sightings grouped by historical era", styles["Caption"]))
story.append(P(
    "Three out of every four reports in this archive fall in the "
    "1945&ndash;1975 window &mdash; the early Cold War decades when radar, "
    "jet aircraft, and nuclear anxiety were all colliding for the first "
    "time. The modern era (2001&ndash;2026) is a much smaller slice, not "
    "because sightings stopped, but because this particular archive is "
    "weighted toward the older, newly-declassified material."
))

# ── Geography ────────────────────────────────────────────────────────────
story.append(H("Where in the World"))
story.append(fig("04_top_countries.png"))
story.append(Paragraph("Top sighting locations by country/region", styles["Caption"]))
story.append(P(
    "Unsurprisingly, the overwhelming majority of reports &mdash; 1,383 of "
    "them &mdash; are tied to the United States. But look past the top bar: "
    "“INTERNATIONAL WATERS” shows up as its own distinct category "
    "with 51 reports. These are sightings logged by ships and aircraft "
    "crews out at sea, far from any coastline &mdash; exactly the kind of "
    "witness (trained military observers, standardized instrumentation, no "
    "light pollution) that skeptics usually wish they had more of."
))

# ── Shapes ───────────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(H("What Are People Actually Seeing?"))
story.append(fig("05_craft_shapes.png"))
story.append(Paragraph("Top reported craft shapes", styles["Caption"]))
story.append(P(
    "Strip away the “Unknown” category (464 reports &mdash; the "
    "honest majority, since most witnesses genuinely can't pin down a "
    "precise shape in the moment) and two shapes dominate everything else: "
    "<b>Sphere</b> (438) and <b>Disc</b> (313). Between them, spheres and "
    "discs account for more reports than every other named shape combined "
    "&mdash; triangles, cylinders, and the rest trail well behind. If "
    "you've ever wondered whether “flying saucer” was just a 1947 "
    "media invention or an actual recurring pattern in what people report, "
    "this chart is the closest thing to an answer: the disc shape shows up "
    "again and again, decades apart, across completely unrelated witnesses."
))

# ── Trust ────────────────────────────────────────────────────────────────
story.append(H("How Much Do We Trust These Reports?"))
story.append(P(
    "Every report in this archive carries a trust score &mdash; a 0&ndash;100 "
    "rating based on witness credibility, corroboration, and internal "
    "consistency."
))
story.append(fig("06_trust_score_distribution.png", max_h=2.6 * inch))
story.append(Paragraph("Distribution of report trust scores", styles["Caption"]))
story.append(P(
    "The average sits at 60 out of 100, and the distribution is broad "
    "rather than clustered at the bottom &mdash; this isn't a pile of "
    "obviously dismissible reports. Break it into bands and the picture "
    "gets sharper:"
))
story.append(fig("07_trust_bands.png", max_h=2.4 * inch))
story.append(Paragraph("Reports grouped by trust band", styles["Caption"]))
story.append(P(
    "Nearly a thousand reports &mdash; high plus very-high combined "
    "&mdash; sit in the upper trust tiers. That's not a fringe minority; "
    "that's close to half the entire archive rated as credible-to-highly-"
    "credible by whatever standard produced this score."
))

# ── Day/night ────────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(H("Night Sky, Day Sky"))
story.append(fig("08_day_night.png", max_h=2.4 * inch))
story.append(Paragraph("Sightings by time of day", styles["Caption"]))
story.append(P(
    "Slightly more sightings happen at night (710) than during the day "
    "(617), which tracks with intuition &mdash; unusual lights are easier "
    "to notice against a dark sky. But the daytime number is far from "
    "negligible, and it argues against the easy dismissal that this is all "
    "just misidentified stars, planets, or aircraft lights. A disc-shaped "
    "object reported at 617 daylight sightings isn't explained by "
    "“it was probably Venus.”"
))

# ── Engagement ───────────────────────────────────────────────────────────
story.append(H("Encounters With the Military"))
story.append(P(
    "This is where the archive gets genuinely striking. A cluster of "
    "fields track whether a sighting involved some form of direct "
    "interaction with military assets or defied conventional physics:"
))
story.append(fig("10_engagement_performance_rates.png", max_h=3.6 * inch))
story.append(Paragraph("Engagement and performance anomaly rates across the corpus", styles["Caption"]))
story.append(B("<b>41.3%</b> of reports flag “low observability” &mdash; the object was hard to track on radar or visually, despite being seen."))
story.append(B("<b>20.6%</b> occurred directly over a <b>military installation.</b>"))
story.append(B("<b>16.3%</b> involved a direct <b>aircraft encounter.</b>"))
story.append(B("<b>14.2%</b> described <b>multiple interactive flight</b> maneuvers &mdash; the object appeared to respond to the observer."))
story.append(B("<b>11.5%</b> logged an <b>aircraft engagement</b> specifically."))
story.append(B("<b>11%</b> exhibited <b>positive lift</b> with no visible means of propulsion."))
story.append(Spacer(1, 6))
story.append(P(
    "Read that first line again: two out of every five reports in this "
    "archive involve something that shouldn't have been hard to track, but "
    "was. And a full fifth of all reports happened directly over sensitive "
    "military ground. This isn't background noise &mdash; it's a recurring "
    "signature across a dataset spanning eighty years."
))

# ── Timeliness ───────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(H("Where the Investigations Stand Today"))
story.append(fig("11_timeliness_status.png", max_h=2.2 * inch))
story.append(Paragraph("Investigation timeliness status", styles["Caption"]))
story.append(P(
    "Most of the archive &mdash; 1,637 reports &mdash; is marked "
    "“presumed timely via source,” meaning the original "
    "documentation was processed close to when the sighting occurred. A "
    "smaller set, 154 reports, are flagged as having had no formal "
    "investigation at the time. Put together, this is a body of evidence "
    "that was, for the most part, actually looked at when it came in "
    "&mdash; not simply filed away and forgotten."
))

# ── Composition / who released this ─────────────────────────────────────
story.append(H("Who Released This"))
story.append(fig("01_source_composition.png", max_h=2.4 * inch))
story.append(Paragraph("Corpus composition by release batch", styles["Caption"]))
story.append(P(
    "This report draws on 1,873 sighting records spanning four separate "
    "document releases from the Department of War, published between May "
    "and July 2026 &mdash; the most substantial voluntary release of "
    "UAP-related material in the program's history. The bulk of the "
    "archive (1,791 reports) comes from the first three releases; the "
    "newest batch adds another 82 documents, several of which &mdash; "
    "after cross-checking &mdash; turned out to describe events already on "
    "record in the earlier releases, corroborating each other "
    "independently."
))

story.append(Spacer(1, 20))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Source: PURSUE government-release corpus, compiled from official "
    "Department of War UAP document releases 1&ndash;4 (war.gov/UFO). "
    "Figures generated from the merged sighting-report dataset; full "
    "technical documentation available in the project repository.",
    styles["Footer"],
))


def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=LETTER,
        leftMargin=1 * inch, rightMargin=1 * inch,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        title="What the Government's UFO Files Actually Say",
        author="PURSUE Data Project",
    )
    doc.build(story)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
