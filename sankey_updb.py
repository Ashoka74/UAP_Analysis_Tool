import pandas as pd
import plotly.graph_objects as go

df = pd.read_excel("/mnt/d/divided/updb_with_agency.xlsx")

# Final source: updb_source if present, else overmeire_Reference
df["final_source"] = df["updb_source"].where(
    df["updb_source"].notna(), df["overmeire_Reference"]
)

df["agency"]                 = df["agency"].fillna("FBI")
df["collection"]             = df["collection"].fillna("")
df["may_release_source_file"]= df["may_release_source_file"].fillna("(unknown file)")
df["final_source"]           = df["final_source"].fillna("(no source)")

# Build ordered node list, deduplicated
def make_nodes(*cols):
    seen, nodes = set(), []
    for col in cols:
        for v in df[col].unique():
            if v not in seen:
                seen.add(v)
                nodes.append(v)
    return nodes

def wrap_label(text, max_len=40):
    if len(text) <= max_len:
        return text
    mid = len(text) // 2
    left  = text.rfind(" ", 0, mid)
    right = text.find(" ", mid)
    split = left if left != -1 else (right if right != -1 else mid)
    return text[:split] + "<br>" + text[split + 1:]

df["final_source"] = df["final_source"].apply(wrap_label)

nodes = make_nodes("agency", "collection", "may_release_source_file", "final_source")
idx   = {n: i for i, n in enumerate(nodes)}

# Count flows across each pair of consecutive levels
def flow_counts(src_col, tgt_col):
    counts = df.groupby([src_col, tgt_col]).size().reset_index(name="value")
    counts.columns = ["src", "tgt", "value"]
    return counts

flows = pd.concat([
    flow_counts("agency",                  "collection"),
    flow_counts("collection",              "may_release_source_file"),
    flow_counts("may_release_source_file", "final_source"),
], ignore_index=True)

PALETTE     = ["#f97316", "#22c55e", "#3b82f6"]   # orange, green, blue
LEVEL_COLS  = ["agency", "collection", "may_release_source_file", "final_source"]
LEVEL_X     = [0.01, 0.18, 0.36, 0.99]

# Distinct colors for each node in col 3 (may_release_source_file)
COL3_PALETTE = [
    "#e74c3c","#9b59b6","#2ecc71","#1abc9c","#f39c12",
    "#e67e22","#3498db","#16a085","#d35400","#8e44ad",
    "#27ae60","#c0392b","#2980b9",
]
srcfile_nodes = list(df["may_release_source_file"].unique())
col3_color = {n: COL3_PALETTE[i % len(COL3_PALETTE)] for i, n in enumerate(srcfile_nodes)}

def _hex_rgba(hex_color, alpha=0.4):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

node_colors = []
node_x      = []
level_node_lists = [[], [], [], []]

for n in nodes:
    for level_i, col in enumerate(LEVEL_COLS):
        if n in df[col].values:
            if level_i == 2:
                node_colors.append(col3_color[n])
            else:
                node_colors.append(PALETTE[min(level_i, len(PALETTE) - 1)])
            node_x.append(LEVEL_X[level_i])
            level_node_lists[level_i].append(n)
            break
    else:
        node_colors.append(PALETTE[-1])
        node_x.append(LEVEL_X[-1])
        level_node_lists[-1].append(n)

# Distribute y evenly within each level
node_y = [0.5] * len(nodes)
for level_i, level_nodes in enumerate(level_node_lists):
    n_nodes = len(level_nodes)
    for rank, n in enumerate(level_nodes):
        node_y[idx[n]] = (rank + 1) / (n_nodes + 1)

# Bridge colors: col3→final bridges inherit the col3 source node color
agency_vals = set(df["agency"].unique())
coll_vals   = set(df["collection"].unique())

def link_color(src_label):
    if src_label in agency_vals:
        return _hex_rgba(PALETTE[0])          # orange — agency→collection
    if src_label in coll_vals:
        return _hex_rgba(PALETTE[1])          # green  — collection→source_file
    return _hex_rgba(col3_color.get(src_label, PALETTE[2]))  # per-file color

link_colors = [link_color(s) for s in flows["src"]]

fig = go.Figure(go.Sankey(
    arrangement="fixed",
    node=dict(
        label=nodes,
        pad=15,
        thickness=20,
        color=node_colors,
        x=node_x,
        y=node_y,
    ),
    link=dict(
        source=[idx[s] for s in flows["src"]],
        target=[idx[t] for t in flows["tgt"]],
        value =flows["value"].tolist(),
        color=link_colors,
        label=[str(v) for v in flows["value"]],
    ),
))

fig.update_layout(
    title=dict(
        text="References in the literature prior to the May Release"
             "<br><sup>Agency → Collection → Source File → UPDB / Overmeire Reference</sup>",
    ),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_size=12,
    height=700,
)

out = "/mnt/d/divided/sankey_updb.png"
fig.write_image(out, width=1400, height=800, scale=200/72)
print("Saved:", out)
