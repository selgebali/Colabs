import math
import os
import time
import urllib.parse
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import requests

# ============================================================
# CONFIGURATION
# ============================================================
CLIENT_ID = "ZBW.ZBW-JDA"
PAGE_SIZE = 10
MAX_ITEMS = 10

HTML_FILE = "merged_knowledge_graphs.html"
NODES_CSV = "merged_nodes.csv"
EDGES_CSV = "merged_edges.csv"

TIMEOUT = 10
RETRY = 3
SLEEP_BETWEEN_CALLS = 0.25

FIG_W = 1000
FIG_H = 900

RADIUS = 1.8          # circle radius for related nodes (controls edge length)
CURVATURE = 0.22      # base curvature for edges (0 = straight line)
CURVE_JITTER = 0.06   # small +/- added to curvature to reduce overlaps
EDGE_SAMPLES = 80     # points per edge curve
EDGE_WIDTH = 2
EDGE_COLOR = "#888"
LABEL_OFFSET = 0.08

custom_colors: Dict[str, str] = {
    "Central": "#243B54",
    "RelatedItem": "#90D7CD",
    "Fallback": "#7f7f7f",
}


# ============================================================
# UTILS: text wrapping, safe requests, small helpers
# ============================================================

def wrap_text(text: str, width: int = 24) -> str:
    """Wrap long labels for nicer Plotly rendering."""
    return "<br>".join(textwrap.wrap(text, width=width))


def safe_get(url: str, timeout: int = TIMEOUT, retry: int = RETRY) -> requests.Response:
    """
    GET with simple retries and a short backoff.
    Raises the last exception if all retries fail.
    """
    last_err: Optional[Exception] = None
    for _ in range(retry):
        try:
            resp = requests.get(url, timeout=timeout)
            if 200 <= resp.status_code < 300:
                return resp
            last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        except Exception as exc:  # noqa: BLE001 - simple retry loop
            last_err = exc
        time.sleep(SLEEP_BETWEEN_CALLS)
    if last_err:
        raise last_err
    raise RuntimeError("safe_get failed unexpectedly without an error object")


def url_for_related_doi(doi: str) -> str:
    """
    DataCite endpoint for a single DOI resource.
    DOI must be URL-encoded (slashes included).
    """
    encoded = urllib.parse.quote(doi, safe="")
    return f"https://api.datacite.org/dois/{encoded}"


def normalize_string(value: Optional[str]) -> str:
    """None-safe strip."""
    return (value or "").strip()


# ============================================================
# SORTING: prioritize DOIs with the most related identifiers
# ============================================================

def sort_items_by_related_count(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort API items in descending order by number of relatedIdentifiers."""

    def related_count(it: Dict[str, Any]) -> int:
        attributes = it.get("attributes", {}) or {}
        return len(attributes.get("relatedIdentifiers") or [])

    return sorted(items, key=related_count, reverse=True)


# ============================================================
# DATA HELPERS
# ============================================================

def resolve_related_target(related: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Build a display label for a related identifier and optionally enrich with DOI data.

    Returns:
        Tuple of (display_label, target_type_general, title)
    """
    identifier = normalize_string(related.get("relatedIdentifier"))
    identifier_type = normalize_string(related.get("relatedIdentifierType"))
    target_type_general = ""
    title = ""

    if identifier_type.upper() == "DOI" and identifier:
        try:
            response = safe_get(url_for_related_doi(identifier))
            data = response.json().get("data") or {}
            attributes = data.get("attributes", {}) or {}
            target_type_general = normalize_string(
                (attributes.get("types") or {}).get("resourceTypeGeneral")
            )
            titles = attributes.get("titles") or []
            title = normalize_string(titles[0].get("title")) if titles else ""
            if title:
                label = f"{title}\n{identifier}"
            else:
                label = f"DOI: {identifier}"
        except Exception:
            label = f"DOI: {identifier}"
    else:
        human_readable_type = identifier_type or "Identifier"
        label = f"{human_readable_type}: {identifier or 'Unknown'}"

    return wrap_text(label, width=24), target_type_general, title


def append_df_to_csv(df: pd.DataFrame, path: str) -> None:
    """Append DataFrame content to CSV (create header on first write)."""
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path), encoding="utf-8")


# ============================================================
# VIZ HELPERS: radial layout and quadratic Bézier curves
# ============================================================

def radial_layout(
    graph: nx.Graph, center_node: str, radius: float = RADIUS
) -> Dict[str, Tuple[float, float]]:
    """Place center at (0,0) and others on a circle -> uniform edge lengths/clarity."""
    nodes = [n for n in graph.nodes() if n != center_node]
    if not nodes:
        return {center_node: (0.0, 0.0)}
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos: Dict[str, Tuple[float, float]] = {center_node: (0.0, 0.0)}
    for node, theta in zip(nodes, angles):
        pos[node] = (radius * np.cos(theta), radius * np.sin(theta))
    return pos


def quad_bezier_curve(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    bend: float = CURVATURE,
    samples: int = EDGE_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quadratic Bézier between p0 and p1.
    Control point is offset perpendicular to the segment by 'bend' * distance.
    """
    x0, y0 = p0
    x1, y1 = p1

    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = (x1 - x0), (y1 - y0)
    length = math.hypot(dx, dy) or 1.0
    perp_x, perp_y = -dy / length, dx / length

    cx = mx + bend * length * perp_x
    cy = my + bend * length * perp_y

    t = np.linspace(0, 1, samples)
    x_vals = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
    y_vals = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
    return x_vals, y_vals


# ============================================================
# VIZ: build a per-DOI knowledge graph (central + related items)
# ============================================================

def visualize_item(
    item: Dict[str, Any],
    html_file: str = HTML_FILE,
    nodes_csv: str = NODES_CSV,
    edges_csv: str = EDGES_CSV,
) -> None:
    """
    Render one DOI as a star/radial graph:
      - central node = DOI
      - ring nodes   = related identifiers (optionally enriched by resolving DOIs)
    """
    attributes = item.get("attributes", {}) or {}
    doi = normalize_string(attributes.get("doi")) or "Unknown DOI"
    related_identifiers = attributes.get("relatedIdentifiers") or []

    graph: nx.Graph = nx.Graph()

    node_size_map = {"Central": 320, "RelatedItem": 200}

    central_info = wrap_text(f"DOI: {doi}\nRelated IDs: {len(related_identifiers)}", width=24)
    graph.add_node(
        central_info,
        label="Central",
        size=node_size_map["Central"],
        color=custom_colors.get("Central", custom_colors["Fallback"]),
    )

    for related in related_identifiers:
        display_label, target_type_general, title = resolve_related_target(related)
        graph.add_node(
            display_label,
            label="RelatedItem",
            size=node_size_map["RelatedItem"],
            color=custom_colors.get("RelatedItem", custom_colors["Fallback"]),
        )

        relation_type = normalize_string(related.get("relationType"))
        identifier_type = normalize_string(related.get("relatedIdentifierType"))

        graph.add_edge(
            central_info,
            display_label,
            relationType=relation_type,
            identifierType=identifier_type,
            targetTypeGeneral=target_type_general,
            title=title,
        )

        if identifier_type.upper() == "DOI":
            time.sleep(SLEEP_BETWEEN_CALLS)

    for u, v in graph.edges():
        graph.edges[u, v]["weight"] = 0.1

    positions = radial_layout(graph, central_info, radius=RADIUS)

    node_x = [positions[node][0] for node in graph.nodes()]
    node_y = [positions[node][1] for node in graph.nodes()]
    node_size = [graph.nodes[node]["size"] for node in graph.nodes()]
    node_color = [graph.nodes[node]["color"] for node in graph.nodes()]
    node_text = list(graph.nodes())

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2,
            opacity=1.0,
        ),
        textposition="middle center",
        textfont=dict(size=12, family="Arial", color="white"),
    )

    edge_traces: List[go.Scatter] = []
    annotations: List[Dict[str, Any]] = []

    sign = 1
    for source, target in graph.edges():
        p0, p1 = positions[source], positions[target]
        jitter = CURVE_JITTER if sign > 0 else -CURVE_JITTER
        bend = sign * (CURVATURE + jitter)
        sign *= -1

        x_vals, y_vals = quad_bezier_curve(p0, p1, bend=bend, samples=EDGE_SAMPLES)

        relation_type = graph.edges[source, target].get("relationType", "")
        identifier_type = graph.edges[source, target].get("identifierType", "")
        target_type_general = graph.edges[source, target].get("targetTypeGeneral", "")

        hover_text = (
            f"relationType: {relation_type}<br>"
            f"identifierType: {identifier_type}<br>"
            f"target RTG: {target_type_general}"
        )

        edge_traces.append(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(width=EDGE_WIDTH, color=EDGE_COLOR),
                hoverinfo="text",
                text=[hover_text] * len(x_vals),
            )
        )

        mid_index = len(x_vals) // 2
        dx = x_vals[min(mid_index + 1, len(x_vals) - 1)] - x_vals[max(mid_index - 1, 0)]
        dy = y_vals[min(mid_index + 1, len(y_vals) - 1)] - y_vals[max(mid_index - 1, 0)]
        angle = math.degrees(math.atan2(dy, dx))

        length = math.hypot(dx, dy) or 1.0
        normal_x, normal_y = -dy / length, dx / length

        label_x = x_vals[mid_index] + LABEL_OFFSET * normal_x
        label_y = y_vals[mid_index] + LABEL_OFFSET * normal_y

        if relation_type:
            annotations.append(
                dict(
                    x=label_x,
                    y=label_y,
                    xref="x",
                    yref="y",
                    text=relation_type,
                    showarrow=False,
                    align="center",
                    textangle=angle,
                    font=dict(size=12, family="Arial", color="#111"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    borderpad=2,
                    opacity=1,
                    captureevents=False,
                )
            )

    figure = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(text=f"Related Identifiers Graph — {doi}", x=0.5),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=FIG_W,
            height=FIG_H,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=annotations,
        ),
    )

    figure.show()

    with open(html_file, "a", encoding="utf-8") as handle:
        handle.write(pio.to_html(figure, include_plotlyjs="cdn"))

    node_rows = [
        {
            "Node": node,
            "Label": graph.nodes[node]["label"],
            "Color": graph.nodes[node]["color"],
            "Size": graph.nodes[node]["size"],
        }
        for node in graph.nodes()
    ]
    append_df_to_csv(pd.DataFrame(node_rows), nodes_csv)

    edge_rows = [
        {
            "Source": source,
            "Target": target,
            "relationType": graph.edges[source, target].get("relationType", ""),
            "identifierType": graph.edges[source, target].get("identifierType", ""),
            "targetTypeGeneral": graph.edges[source, target].get("targetTypeGeneral", ""),
            "title": graph.edges[source, target].get("title", ""),
        }
        for source, target in graph.edges()
    ]
    append_df_to_csv(pd.DataFrame(edge_rows), edges_csv)


# ============================================================
# API FETCHING
# ============================================================

def fetch_items_for_client(
    client_id: str,
    page_size: int = PAGE_SIZE,
    limit: Optional[int] = MAX_ITEMS,
) -> List[Dict[str, Any]]:
    """Fetch DOIs for a client and sort them by related identifier counts."""
    url = f"https://api.datacite.org/dois?client-id={client_id}&page[size]={page_size}"
    response = safe_get(url)
    payload = response.json()
    items = payload.get("data") or []
    total = payload.get("meta", {}).get("total")
    print(f"Total number of resources available: {total}")

    sorted_items = sort_items_by_related_count(items)
    if limit is not None:
        return sorted_items[:limit]
    return sorted_items


# ============================================================
# MAIN EXECUTION
# ============================================================

def main() -> None:
    for path in (HTML_FILE, NODES_CSV, EDGES_CSV):
        if os.path.exists(path):
            os.remove(path)

    with open(HTML_FILE, "w", encoding="utf-8") as handle:
        handle.write("<html><head><title>Knowledge Graphs</title></head><body>")

    items = fetch_items_for_client(CLIENT_ID, page_size=PAGE_SIZE, limit=MAX_ITEMS)
    for item in items:
        visualize_item(item)

    with open(HTML_FILE, "a", encoding="utf-8") as handle:
        handle.write("</body></html>")

    print(
        f"Graphs saved in {HTML_FILE}, nodes in {NODES_CSV}, and edges in {EDGES_CSV}"
    )


if __name__ == "__main__":
    main()
