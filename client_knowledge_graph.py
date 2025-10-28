import os
import textwrap
from typing import List, Dict, Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import requests

# Custom color palette
custom_colors = {
    "Resource": "#243B54",
    "Creator": "#00B1E2",
    "Contributor": "#5B88B9",
    "Publisher": "#46BCAB",
    "RelatedItem": "#90D7CD",
}


# Helper function to wrap text
def wrap_text(text: str, width: int = 20) -> str:
    return "<br>".join(textwrap.wrap(text, width=width))


# Function to fetch data from the API endpoint using a client filter
def fetch_api_data(url: str, limit: int = 100) -> List[Dict[str, Any]]:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    resources = parse_resources(data.get("data", []))
    total_resources = data.get("meta", {}).get("total", "Unknown")
    print(f"Total number of resources available: {total_resources}")
    return resources[:limit]


# Function to parse resources
def parse_resources(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    resources = []
    for item in data:
        attributes = item.get("attributes", {})
        types = attributes.get("types", {})

        resource_data = {
            "doi": attributes.get("doi", "No DOI available"),
            "resourceTypeGeneral": types.get("resourceTypeGeneral", "No resourceTypeGeneral available"),
            "resourceType": types.get("resourceType", "No resource type available"),
            "schemaOrg": types.get("schemaOrg", "No schemaOrg available"),
            "creators": [],
            "contributors": [],
            "publishers": attributes.get("publisher", "No publisher information"),
            "relatedItems": [],
        }

        # Add creators
        for creator in attributes.get("creators", []):
            identifiers = creator.get("nameIdentifiers") or []
            identifier = identifiers[0].get("nameIdentifier") if identifiers else "No identifier"
            resource_data["creators"].append(
                {
                    "name": creator.get("name", "No name available"),
                    "identifier": identifier,
                }
            )

        # Add contributors
        for contributor in attributes.get("contributors", []):
            identifiers = contributor.get("nameIdentifiers") or []
            identifier = identifiers[0].get("nameIdentifier") if identifiers else "No identifier"
            resource_data["contributors"].append(
                {
                    "name": contributor.get("name", "No name available"),
                    "type": contributor.get("contributorType", "No type available"),
                    "identifier": identifier,
                }
            )

        # Add related items
        for related in attributes.get("relatedIdentifiers", []):
            resource_data["relatedItems"].append(
                {
                    "identifier": related.get("relatedIdentifier", "No identifier available"),
                    "relationType": related.get("relationType", "No relation type available"),
                }
            )

        resources.append(resource_data)
    return resources


# Function to append DataFrame content to CSV (create header on first write)
def append_df_to_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path), encoding="utf-8")


# Function to visualize the resource network and append results
def visualize_resource_plotly(resource: Dict[str, Any], html_file: str, nodes_csv: str, edges_csv: str) -> None:
    G = nx.Graph()

    # Define node size map
    node_size_map = {
        "Resource": 300,
        "Creator": 200,
        "Contributor": 200,
        "Publisher": 200,
        "RelatedItem": 200,
    }

    resource_type_general = resource.get("resourceTypeGeneral", "No resourceTypeGeneral available")
    central_node_info = f"{resource_type_general}\nDOI: {resource['doi']}"
    resource_node_label = wrap_text(central_node_info, width=20)
    G.add_node(
        resource_node_label,
        label="Resource",
        size=node_size_map.get("Resource", 80),
        color=custom_colors.get("Resource", "#7f7f7f"),
    )

    # Add creators
    for creator in resource["creators"]:
        creator_info = f"Creator: {creator['name']}\nID: {creator['identifier']}"
        creator_label = wrap_text(creator_info, width=20)
        G.add_node(
            creator_label,
            label="Creator",
            size=node_size_map.get("Creator", 60),
            color=custom_colors.get("Creator", "#7f7f7f"),
        )
        G.add_edge(resource_node_label, creator_label)

    # Add contributors
    for contributor in resource["contributors"]:
        contributor_label = wrap_text(
            f"Contributor: {contributor['name']}\n {contributor['identifier']}", width=20
        )
        G.add_node(
            contributor_label,
            label="Contributor",
            size=node_size_map.get("Contributor", 60),
            color=custom_colors.get("Contributor", "#7f7f7f"),
        )
        G.add_edge(resource_node_label, contributor_label)

    # Add publisher
    publisher_label = wrap_text(f"Publisher: {resource['publishers']}", width=20)
    G.add_node(
        publisher_label,
        label="Publisher",
        size=node_size_map.get("Publisher", 60),
        color=custom_colors.get("Publisher", "#7f7f7f"),
    )
    G.add_edge(resource_node_label, publisher_label)

    # Add related items
    for item in resource["relatedItems"]:
        item_node_label = wrap_text(f"{item['relationType']}: {item['identifier']}", width=20)
        G.add_node(
            item_node_label,
            label="RelatedItem",
            size=node_size_map.get("RelatedItem", 60),
            color=custom_colors.get("RelatedItem", "#7f7f7f"),
        )
        G.add_edge(resource_node_label, item_node_label)

    # Define node positions using circular layout for better spacing
    pos = nx.circular_layout(G)

    # Extract node properties
    node_x = [pos[node][0] for node in G.nodes]
    node_y = [pos[node][1] for node in G.nodes]
    node_size = [G.nodes[node]["size"] for node in G.nodes]
    node_color = [G.nodes[node]["color"] for node in G.nodes]
    node_text = list(G.nodes)
    node_hovertext = list(G.nodes)

    # Create node traces with text labels
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        hovertext=node_hovertext,
        hoverinfo="text",
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2,
            opacity=1.0,
        ),
        textposition="middle center",
        textfont=dict(size=12, family="Arial", color="white", weight="bold"),
    )

    # Create edge traces with curved lines
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        width = 2  # default edge width

        # Generate slight curve for edges (using quadratic Bézier curve approximation)
        t = np.linspace(0, 1, 100)
        x_mid = (x0 + x1) / 2
        y_mid = (y0 + y1) / 2 + 0.1  # Adjust for curvature
        x_values = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x_mid + t**2 * x1
        y_values = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y_mid + t**2 * y1

        edge_trace = go.Scatter(
            x=x_values,
            y=y_values,
            line=dict(width=width, color="#888"),
            hoverinfo="none",
            mode="lines",
        )
        edge_traces.append(edge_trace)

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Knowledge Graph of {resource['doi']}",
                font=dict(size=20),
                x=0.5,
                y=0.98,
                xanchor="center",
                yanchor="top",
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=1000,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.show()

    with open(html_file, "a", encoding="utf-8") as f:
        f.write(pio.to_html(fig, include_plotlyjs="cdn"))

    node_data = [
        {
            "Node": node,
            "Label": G.nodes[node]["label"],
            "Size": G.nodes[node]["size"],
            "Color": G.nodes[node]["color"],
        }
        for node in G.nodes()
    ]
    nodes_df = pd.DataFrame(node_data)
    append_df_to_csv(nodes_df, nodes_csv)

    edge_data = [{"Source": edge[0], "Target": edge[1]} for edge in G.edges()]
    edges_df = pd.DataFrame(edge_data)
    append_df_to_csv(edges_df, edges_csv)


if __name__ == "__main__":
    client_id = "ZBW.ZBW-JDA"
    api_url = f"https://api.datacite.org/dois?client-id={client_id}&page[size]=10"

    html_file = "merged_knowledge_graphs.html"
    nodes_csv = "merged_nodes.csv"
    edges_csv = "merged_edges.csv"

    with open(html_file, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Knowledge Graphs</title></head><body>")

    resources = fetch_api_data(api_url)
    for resource in resources:
        visualize_resource_plotly(resource, html_file, nodes_csv, edges_csv)

    with open(html_file, "a", encoding="utf-8") as f:
        f.write("</body></html>")

    print(
        f"Graphs saved in {html_file}, nodes in {nodes_csv}, and edges in {edges_csv}"
    )
