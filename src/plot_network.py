"""
Generates three figures from AlterNet pipeline output:
  1. Network visualization of the combined regulatory network
  2. Degree distribution histogram
  3. Bar chart of top 20 genes by degree
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os.path as op
import os
import sys

EDGE_CATEGORIES = [
    'unique_isoforms',
    'unique_genes',
    'consistent_both',
    'likely_isoform_specific',
    'likely_gene_specific',
    'ambigous',
]

CATEGORY_COLORS = {
    'unique_isoforms': '#e41a1c',
    'unique_genes': '#377eb8',
    'consistent_both': '#4daf4a',
    'likely_isoform_specific': '#ff7f00',
    'likely_gene_specific': '#984ea3',
    'ambigous': '#999999',
}


def load_edges(results_path, prefix):
    frames = []
    for cat in EDGE_CATEGORIES:
        path = op.join(results_path, f"{prefix}_{cat}.tsv")
        if not op.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        if df.empty:
            continue
        df['edge_category'] = cat
        frames.append(df)

    if not frames:
        print("No edge files found. Check results_path and prefix.")
        sys.exit(1)

    edges = pd.concat(frames, ignore_index=True)

    # Resolve the best available source name column, falling back through
    # transcript name -> gene name -> gene ID -> transcript ID
    source_candidates = ['source_transcript_name', 'source_gene_name', 'source_gene', 'source_transcript']
    edges['source_label'] = pd.NA
    for col in source_candidates:
        if col in edges.columns:
            edges['source_label'] = edges['source_label'].fillna(edges[col])

    target_candidates = ['target_gene_name', 'target_gene']
    edges['target_label'] = pd.NA
    for col in target_candidates:
        if col in edges.columns:
            edges['target_label'] = edges['target_label'].fillna(edges[col])

    # Drop any edges where labels could not be resolved
    edges = edges.dropna(subset=['source_label', 'target_label'])

    return edges


def build_graph(edges):
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        src = row['source_label']
        tgt = row['target_label']
        cat = row['edge_category']
        G.add_edge(src, tgt, category=cat)
    return G


def plot_network(G, edges, out_dir):
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Draw edges colored by category
    for cat in EDGE_CATEGORIES:
        cat_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('category') == cat]
        if cat_edges:
            nx.draw_networkx_edges(G, pos, edgelist=cat_edges, ax=ax,
                                   edge_color=CATEGORY_COLORS[cat],
                                   alpha=0.6, arrows=True, arrowsize=12,
                                   label=cat.replace('_', ' '))

    # Node sizes proportional to degree, scaled relative to network size
    degrees = dict(G.degree())
    n_nodes = G.number_of_nodes()
    base_size = max(10, 800 / n_nodes)
    scale_factor = max(1, 200 / n_nodes)
    node_sizes = [base_size + scale_factor * degrees[n] for n in G.nodes()]

    # TF sources vs targets
    sources = set(edges['source_label'])
    targets = set(edges['target_label'])
    node_colors = ['#fdae61' if n in sources else '#abd9e9' for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='#333333', linewidths=0.5)
    max_degree = max(degrees.values()) if degrees else 1
    font_sizes = {n: max(4, base_size / 10 + scale_factor / 5 * degrees[n]) for n in G.nodes()}
    for node, (x, y) in pos.items():
        ax.text(x, y, node, fontsize=font_sizes[node], ha='center', va='center')

    ax.legend(loc='upper left', fontsize=8, title='Edge category')
    ax.set_title('AlterNet Regulatory Network', fontsize=14)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(op.join(out_dir, 'network_visualization.png'), dpi=200)
    plt.close(fig)
    print(f"Saved: {op.join(out_dir, 'network_visualization.png')}")


def plot_degree_distribution(G, out_dir):
    degrees = [d for _, d in G.degree()]

    fig, ax = plt.subplots(figsize=(8, 5))
    max_deg = max(degrees) if degrees else 1
    bins = range(0, max_deg + 2)
    ax.hist(degrees, bins=bins, edgecolor='black', color='#4daf4a', alpha=0.8, align='left')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Number of nodes')
    ax.set_title('Degree Distribution')
    ax.set_xticks(range(0, max_deg + 1))
    fig.tight_layout()
    fig.savefig(op.join(out_dir, 'degree_distribution.png'), dpi=200)
    plt.close(fig)
    print(f"Saved: {op.join(out_dir, 'degree_distribution.png')}")


def _format_bar_label(val):
    if isinstance(val, int):
        return str(val)
    if abs(val) < 0.001:
        return f'{val:.2e}'
    return f'{val:.4f}'


def plot_top_genes(G, out_dir, top_n=20):
    metrics = {
        'Degree': dict(G.degree()),
        'Betweenness': nx.betweenness_centrality(G),
        'PageRank': nx.pagerank(G),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#377eb8', '#e41a1c', '#4daf4a']

    for ax, (name, scores), color in zip(axes, metrics.items(), colors):
        top = pd.Series(scores).nlargest(top_n)
        bars = ax.barh(range(len(top)), top.values, color=color, edgecolor='black')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(name)
        ax.set_title(f'Top {len(top)} Genes — {name}')
        for bar, val in zip(bars, top.values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f' {_format_bar_label(val)}', va='center', fontsize=7)

    fig.tight_layout()
    fig.savefig(op.join(out_dir, 'top_genes_by_centrality.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {op.join(out_dir, 'top_genes_by_centrality.png')}")


def main():
    results_path = op.join(op.dirname(__file__), '..', 'results', 'myc-all-samples')
    prefix = 'myc-all-samples'
    out_dir = op.join(results_path, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    edges = load_edges(results_path, prefix)
    print(f"Loaded {len(edges)} edges across {edges['edge_category'].nunique()} categories")

    G = build_graph(edges)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    plot_network(G, edges, out_dir)
    plot_degree_distribution(G, out_dir)
    plot_top_genes(G, out_dir, top_n=20)


if __name__ == '__main__':
    main()
