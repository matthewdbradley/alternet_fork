"""
Generates network visualizations and downstream analyses from AlterNet pipeline output:
  1. Network visualization (PDF)
  2. Degree distribution histogram
  3. Bar chart of top 20 genes by centrality
  4. Top-200 gene overlap across betweenness, PageRank, and degree
  5. GO + cancer pathway enrichment with Jaccard index
  6. Top-10 enriched term network highlights (PDF per term)
  7. MYC ego-network analysis for TF regulation context
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os.path as op
import os
import sys
import logging
import warnings

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Data loading / graph building
# ---------------------------------------------------------------------------

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

    edges = edges.dropna(subset=['source_label', 'target_label'])
    return edges


def build_graph(edges):
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row['source_label'], row['target_label'], category=row['edge_category'])
    return G


# ---------------------------------------------------------------------------
# 1. Network visualization  (now PDF)
# ---------------------------------------------------------------------------

def plot_network(G, edges, out_dir):
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=2.0)

    for cat in EDGE_CATEGORIES:
        cat_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('category') == cat]
        if cat_edges:
            nx.draw_networkx_edges(G, pos, edgelist=cat_edges, ax=ax,
                                   edge_color=CATEGORY_COLORS[cat],
                                   alpha=0.6, arrows=True, arrowsize=12,
                                   label=cat.replace('_', ' '))

    degrees = dict(G.degree())
    n_nodes = G.number_of_nodes()
    base_size = max(10, 800 / n_nodes)
    scale_factor = max(1, 200 / n_nodes)
    node_sizes = [base_size + scale_factor * degrees[n] for n in G.nodes()]

    sources = set(edges['source_label'])
    node_colors = ['#fdae61' if n in sources else '#abd9e9' for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='#333333', linewidths=0.5)
    font_sizes = {n: max(4, base_size / 10 + scale_factor / 5 * degrees[n]) for n in G.nodes()}
    for node, (x, y) in pos.items():
        ax.text(x, y, node, fontsize=font_sizes[node], ha='center', va='center')

    ax.legend(loc='upper left', fontsize=8, title='Edge category')
    ax.set_title('AlterNet Regulatory Network', fontsize=14)
    ax.axis('off')
    fig.tight_layout()
    out_path = op.join(out_dir, 'network_visualization.pdf')
    fig.savefig(out_path, dpi=300, format='pdf')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Degree distribution
# ---------------------------------------------------------------------------

def plot_degree_distribution(G, out_dir):
    degrees = [d for _, d in G.degree()]
    fig, ax = plt.subplots(figsize=(8, 5))
    max_deg = max(degrees) if degrees else 1
    bins = range(0, max_deg + 2)
    ax.hist(degrees, bins=bins, edgecolor='black', color='#4daf4a', alpha=0.8, align='left')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Number of nodes')
    ax.set_title('Degree Distribution')
    fig.tight_layout()
    fig.savefig(op.join(out_dir, 'degree_distribution.png'), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top genes bar chart
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4. Top-200 overlap across 3 metrics  (requirement 1)
# ---------------------------------------------------------------------------

def get_top200_overlap(G, top_n=200):
    """Return genes appearing in at least 2 of the 3 top-200 lists."""
    metrics = {
        'degree': dict(G.degree()),
        'betweenness': nx.betweenness_centrality(G),
        'pagerank': nx.pagerank(G),
    }
    top_sets = {}
    for name, scores in metrics.items():
        top_sets[name] = set(pd.Series(scores).nlargest(top_n).index)

    from collections import Counter
    counts = Counter()
    for s in top_sets.values():
        counts.update(s)

    overlap_genes = sorted([g for g, c in counts.items() if c >= 2])
    logger.info("Top-200 overlap: %d genes in >=2/3 metrics", len(overlap_genes))
    return overlap_genes, metrics


# ---------------------------------------------------------------------------
# 5. GO + cancer pathway enrichment  (requirement 2)
# ---------------------------------------------------------------------------

def run_enrichment(gene_list, out_dir, alpha=0.1):
    """Run GO Biological Process + cancer pathway (KEGG) enrichment via gseapy.
    Returns a DataFrame of significant terms with Jaccard index."""
    try:
        import gseapy as gp
    except ImportError:
        logger.error("gseapy not installed — skipping enrichment")
        return pd.DataFrame()

    gene_set = set(gene_list)
    all_results = []

    for lib in ['GO_Biological_Process_2023', 'KEGG_2021_Human', 'MSigDB_Hallmark_2020']:
        try:
            enr = gp.enrichr(gene_list=list(gene_list),
                             gene_sets=lib,
                             organism='human',
                             outdir=None,
                             no_plot=True)
            df = enr.results.copy()
            df['Library'] = lib
            all_results.append(df)
        except Exception as e:
            logger.warning("Enrichment failed for %s: %s", lib, e)

    if not all_results:
        return pd.DataFrame()

    results = pd.concat(all_results, ignore_index=True)

    # Compute Jaccard index: |intersection(query, term)| / |union(query, term)|
    def _jaccard(genes_str):
        if pd.isna(genes_str) or not genes_str:
            return 0.0
        term_genes = set(genes_str.split(';'))
        inter = len(gene_set & term_genes)
        union = len(gene_set | term_genes)
        return inter / union if union > 0 else 0.0

    results['Jaccard'] = results['Genes'].apply(_jaccard)
    results = results[results['Adjusted P-value'] < alpha].sort_values('Adjusted P-value')

    # Save table
    results.to_csv(op.join(out_dir, 'enrichment_results.csv'), index=False)
    print(f"Saved: {op.join(out_dir, 'enrichment_results.csv')} ({len(results)} terms)")
    return results


def plot_enrichment(results, out_dir, top_n=30):
    """Dot plot of enriched terms showing -log10(p-value) and Jaccard index."""
    if results.empty:
        return

    df = results.head(top_n).copy()
    df['neg_log_p'] = -np.log10(df['Adjusted P-value'].clip(lower=1e-50))
    df['short_term'] = df['Term'].str[:60]

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))
    scatter = ax.scatter(df['neg_log_p'], range(len(df)),
                         c=df['Jaccard'], cmap='YlOrRd', s=80,
                         edgecolors='black', linewidths=0.5, vmin=0)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['short_term'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('-log10(Adjusted P-value)')
    ax.set_title('Enriched Terms (color = Jaccard index)')
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Jaccard Index')
    fig.tight_layout()
    fig.savefig(op.join(out_dir, 'enrichment_dotplot.pdf'), dpi=300, format='pdf')
    plt.close(fig)
    print(f"Saved: {op.join(out_dir, 'enrichment_dotplot.pdf')}")


# ---------------------------------------------------------------------------
# 6. Network highlights for top enriched terms  (requirement 3)
# ---------------------------------------------------------------------------

def plot_term_networks(G, edges, results, out_dir, top_n=10):
    """For each of the top enriched terms, save a network PDF highlighting member genes."""
    if results.empty:
        return

    pos = nx.spring_layout(G, seed=42, k=2.0)
    all_nodes = list(G.nodes())
    sources = set(edges['source_label'])

    for idx, row in results.head(top_n).iterrows():
        term_genes = set(row['Genes'].split(';')) if pd.notna(row['Genes']) else set()
        highlight = term_genes & set(all_nodes)
        if not highlight:
            continue

        fig, ax = plt.subplots(figsize=(14, 10))

        # Draw all edges lightly
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, arrows=True, arrowsize=8,
                               edge_color='#cccccc')

        # Non-highlighted nodes
        other_nodes = [n for n in all_nodes if n not in highlight]
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, ax=ax,
                               node_size=30, node_color='#dddddd',
                               edgecolors='#aaaaaa', linewidths=0.3)

        # Highlighted nodes
        nx.draw_networkx_nodes(G, pos, nodelist=list(highlight), ax=ax,
                               node_size=200, node_color='#e41a1c',
                               edgecolors='black', linewidths=1.0)
        nx.draw_networkx_labels(G, pos, labels={n: n for n in highlight},
                                font_size=7, ax=ax)

        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in row['Term'])[:80]
        ax.set_title(f"{row['Term'][:80]}\np={row['Adjusted P-value']:.2e}  Jaccard={row['Jaccard']:.3f}",
                     fontsize=10)
        ax.axis('off')
        fig.tight_layout()
        fname = op.join(out_dir, f'term_network_{idx:02d}_{safe_name}.pdf')
        fig.savefig(fname, dpi=300, format='pdf')
        plt.close(fig)
        print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# 7. MYC ego-network analysis  (requirement 5)
# ---------------------------------------------------------------------------

MYC_ALIASES = {'MYC', 'c-Myc', 'C-MYC', 'CMYC', 'c-myc'}


def _find_myc_node(G):
    """Find the MYC node in the graph, trying common aliases."""
    for node in G.nodes():
        if node.upper() in {a.upper() for a in MYC_ALIASES}:
            return node
    # Partial match fallback
    for node in G.nodes():
        if 'MYC' in node.upper() and 'DMYC' not in node.upper():
            return node
    return None


def analyze_myc_regulation(G, edges, out_dir):
    """Extract and visualize the MYC-centric ego subnetwork.

    Shows:
    - TFs that regulate MYC (upstream)
    - Genes regulated by MYC (downstream)
    - Edge categories colored
    Outputs a PDF and a summary CSV.
    """
    myc_node = _find_myc_node(G)
    if myc_node is None:
        logger.warning("MYC node not found in graph — skipping MYC analysis")
        print("MYC node not found in graph — skipping MYC ego-network analysis")
        return

    # Upstream TFs → MYC
    upstream = list(G.predecessors(myc_node))
    # Downstream MYC → targets
    downstream = list(G.successors(myc_node))

    ego_nodes = set(upstream + downstream + [myc_node])
    ego_sub = G.subgraph(ego_nodes).copy()

    # Summary table
    rows = []
    for tf in upstream:
        edata = G.edges[tf, myc_node]
        rows.append({'gene': tf, 'relation': 'TF→MYC', 'edge_category': edata.get('category', '')})
    for tgt in downstream:
        edata = G.edges[myc_node, tgt]
        rows.append({'gene': tgt, 'relation': 'MYC→target', 'edge_category': edata.get('category', '')})

    summary = pd.DataFrame(rows)
    summary.to_csv(op.join(out_dir, 'myc_ego_network_summary.csv'), index=False)
    print(f"Saved: {op.join(out_dir, 'myc_ego_network_summary.csv')} "
          f"({len(upstream)} upstream TFs, {len(downstream)} downstream targets)")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(ego_sub, seed=42, k=2.5)

    # Color edges by category
    for cat in EDGE_CATEGORIES:
        cat_edges = [(u, v) for u, v, d in ego_sub.edges(data=True) if d.get('category') == cat]
        if cat_edges:
            nx.draw_networkx_edges(ego_sub, pos, edgelist=cat_edges, ax=ax,
                                   edge_color=CATEGORY_COLORS[cat], alpha=0.8,
                                   arrows=True, arrowsize=15, width=2,
                                   label=cat.replace('_', ' '))

    # Node colors: MYC=red, upstream TFs=orange, downstream targets=blue
    node_colors = []
    for n in ego_sub.nodes():
        if n == myc_node:
            node_colors.append('#e41a1c')
        elif n in upstream:
            node_colors.append('#fdae61')
        else:
            node_colors.append('#abd9e9')

    nx.draw_networkx_nodes(ego_sub, pos, ax=ax, node_size=400,
                           node_color=node_colors, edgecolors='black', linewidths=1.0)
    nx.draw_networkx_labels(ego_sub, pos, font_size=8, ax=ax)

    legend_elements = [
        mpatches.Patch(facecolor='#e41a1c', edgecolor='black', label='MYC'),
        mpatches.Patch(facecolor='#fdae61', edgecolor='black', label='Upstream TF'),
        mpatches.Patch(facecolor='#abd9e9', edgecolor='black', label='Downstream target'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, title='Node type')
    ax.set_title(f'MYC Ego Network ({len(upstream)} TFs → MYC → {len(downstream)} targets)', fontsize=13)
    ax.axis('off')
    fig.tight_layout()
    out_path = op.join(out_dir, 'myc_ego_network.pdf')
    fig.savefig(out_path, dpi=300, format='pdf')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Public entry point — called from pipeline scripts
# ---------------------------------------------------------------------------

def run_all_visualizations(results_path, prefix, out_dir=None):
    """Run all visualization and analysis steps after the AlterNet pipeline."""
    if out_dir is None:
        out_dir = op.join(results_path, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # Load data and build graph
    edges = load_edges(results_path, prefix)
    print(f"Loaded {len(edges)} edges across {edges['edge_category'].nunique()} categories")

    G = build_graph(edges)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Core plots
    plot_network(G, edges, out_dir)
    plot_degree_distribution(G, out_dir)
    plot_top_genes(G, out_dir, top_n=20)

    # Requirement 1: top-200 overlap
    overlap_genes, _ = get_top200_overlap(G, top_n=200)
    pd.DataFrame({'gene': overlap_genes}).to_csv(
        op.join(out_dir, 'top200_overlap_genes.csv'), index=False)
    print(f"Top-200 overlap: {len(overlap_genes)} genes in >=2/3 metrics")

    # Requirement 2: enrichment
    if overlap_genes:
        enrichment = run_enrichment(overlap_genes, out_dir, alpha=0.1)
        plot_enrichment(enrichment, out_dir)

        # Requirement 3: term-highlighted networks
        plot_term_networks(G, edges, enrichment, out_dir, top_n=10)
    else:
        print("No overlap genes — skipping enrichment")
        enrichment = pd.DataFrame()

    # Requirement 5: MYC ego-network
    analyze_myc_regulation(G, edges, out_dir)

    return G, edges, enrichment


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

def main():
    results_path = op.join(op.dirname(__file__), '..', 'results', 'myc-all-samples')
    prefix = 'myc-all-samples'
    run_all_visualizations(results_path, prefix)


if __name__ == '__main__':
    main()
