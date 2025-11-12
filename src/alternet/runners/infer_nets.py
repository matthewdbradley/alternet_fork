import sys
sys.path.append('/data/bionets/og86asub/alternet-project/alternet/')
import pandas as pd
import alternet.data_preprocessing as preprocessing
from alternet.annotation import map_tf_ids
import alternet.annotation as annotation
import alternet.postprocessing as postprocessing

from alternet.inference import *
import yaml
import time
from typing import Any, Dict
import numpy as np

import seaborn as sns
import os.path as op
import os

def write_dict_to_yaml(data: Dict[str, Any], filepath: str):
    """Writes a Python dictionary to a YAML file."""

    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def add_gene_names_and_save(edge_data, condi, net_type, filepath):
    outfile = op.join(filepath, f"{condi}_{net_type}.tsv" )
    os.makedirs(filepath, exist_ok = True)

    # rename columns to fit a schema
    if 'target' in edge_data.columns and 'target_gene' not in edge_data.columns:
        edge_data = edge_data.rename(columns = {'target': 'target_gene'})


    if 'source_transcript' in edge_data.columns and 'source_gene_name' not in edge_data.columns:
        edge_data = postprocessing.add_transcript_names(edge_data, biomart, transcript_column = 'source_transcript')
    if 'source_gene' in edge_data.columns and 'source_gene_name' not in edge_data.columns:
        edge_data = postprocessing.add_gene_names(edge_data, biomart, gene_column = 'source_gene')
    if 'target_gene' in edge_data.columns and 'target_gene_name' not in edge_data.columns:
        edge_data = postprocessing.add_gene_names(edge_data, biomart, gene_column = 'target_gene')
        
    edge_data.to_csv(outfile)


if __name__ == '__main__':
    config_file = "/data/bionets/og86asub/alternet-project/alternet/configs/MAGNet_NF.yaml"

    results_path = "/data/bionets/og86asub/alternet-project/alternet-manuscript/results/"

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)


    biomart_path = '/data/bionets/og86asub/alternet-project/alternet/data/biomart.txt'
    tf_list_path = '/data/bionets/og86asub/alternet-project/alternet/data/allTFs_hg38.txt'
    appris_df = pd.read_csv(config['appris'], sep='\t')

    biomart = pd.read_csv(biomart_path, sep='\t')
    tf_list = pd.read_csv(tf_list_path, sep='\t', header = None)
    tf_list = map_tf_ids(tf_list, biomart)

    
    tf_database = annotation.create_transcipt_annotation_database(tf_list=tf_list, appris_path= config['appris'], digger_path=config['digger'])
    transcript_mapper = annotation.create_transcript_mapping(biomart)

    transcript_data = pd.read_csv(config['transcript_data'], index_col=0)

    # Subset to protein coding isoforms
    protein_coding_isoforms = list(appris_df[appris_df['Transcript type'] == 'protein_coding']['Transcript ID'])
    transcript_data = transcript_data[transcript_data.transcript_id.isin(protein_coding_isoforms)]



    gene_data = transcript_data.groupby('gene_id').sum()
    gene_data = gene_data.drop(columns={'transcript_id'})

    gene_data.index.name = 'gene_id'
    gene_data = gene_data.reset_index()

    sample_attributes = pd.read_csv(config['sample_attributes'])
    sample_attributes = sample_attributes.loc[:, ['sample_name', 'etiology']]

    conditions = ['DCM', 'NF', 'HCM']

    runs = 10
    ## Subset the samples of interest
    for condi in conditions:

        samples = sample_attributes[sample_attributes['etiology'] == config['tissue']]
        samples = samples['sample_name'].tolist()

        gene_data_cp = gene_data.copy(deep=True)
        transcript_data_cp = transcript_data.copy(deep=True)
        ## Get unified gene and transcript table
        gene_data_cp = gene_data_cp.loc[:, ['gene_id'] + samples ]
        transcript_data_cp = transcript_data_cp.loc[:,['gene_id', 'transcript_id'] + samples]


        gene_target_names = list(tf_list['Gene stable ID'].unique())


        gene_data_cp = gene_data_cp.set_index('gene_id')
        gene_data_cp = gene_data_cp.T

        # scale data!!
        gene_data_cp_scaled = standardize_dataframe(gene_data_cp)

        transcript_data_cp = transcript_data_cp.set_index('transcript_id')
        transcript_data_cp = transcript_data_cp.drop('gene_id', axis=1)
        transcript_data_cp = transcript_data_cp.T

        # scale data!!
        transcript_data_cp_scaled = standardize_dataframe(transcript_data_cp)

        # compute isoform categories
        isoform_categories = postprocessing.isoform_categorization(transcript_data_cp, gene_data_cp, tf_list)
        gene_categories = postprocessing.get_gene_cases(isoform_categories)
        gene_to_transcript_mapping = annotation.create_filtered_gene_to_transcripts_mapping(biomart, gene_list = gene_data_cp_scaled.columns, transcript_list = transcript_data_cp_scaled.columns)


        #start = time.monotonic()
        #canonical_grn = inference(gene_data=gene_data_cp, tf_list=gene_target_names, target_names = 'all', n_runs=runs)
        #end = time.monotonic()
        #runtime = {'canonical': end-start}
        
        #canonical_grn.to_csv(f"/data/bionets/og86asub/alternet-project/alternet-manuscript/results/{condi}_canonical.tsv", header = True)

        #hybrid_data = create_hybrid_data(transcript_data_cp, gene_data_cp, tf_list)
        #hybrid_tf_names = list(tf_list['Transcript stable ID'].unique())
        #target_names = list(gene_data_cp.columns)

        #start = time.monotonic()
        #as_aware_grn = inference(gene_data=hybrid_data, tf_list=hybrid_tf_names, target_names=target_names, n_runs = runs)
        #runtime['as_aware'] = time.monotonic()-start

        #as_aware_grn.to_csv(f"/data/bionets/og86asub/alternet-project/alternet-manuscript/results/{condi}_as_aware.tsv", header = True) 
        #write_dict_to_yaml(runtime, f"/data/bionets/og86asub/alternet-project/alternet-manuscript/results/{condi}_runtime.yaml")

        start_postprocessing = time.monotonic()


        #filter aggregate
        as_aware_grn = pd.read_csv(f"/data/bionets/og86asub/alternet-project/alternet-manuscript/results/{condi}_as_aware.tsv", index_col=0)
        canonical_grn = pd.read_csv(f"/data/bionets/og86asub/alternet-project/alternet-manuscript/results/{condi}_canonical.tsv", index_col=0)


        as_aware_grn = postprocessing.map_transcript_to_gene(as_aware_grn, transcript_mapper)
        as_aware_grn = postprocessing.create_edge_key(as_aware_grn)
        canonical_grn = postprocessing.create_edge_key(canonical_grn, source_column='source', target_column='target')
        canonical_grn = canonical_grn.rename(columns={'source': 'source_gene'})


        # Categorize edges into gene-unique, isoform-unique, common
        common_edges = postprocessing.get_common_edges(canonical_grn, as_aware_grn)
        gene_unique, isoform_unique = postprocessing.get_diff(canonical_grn, as_aware_grn)

        filtering_tracker = {'as_aware_base': as_aware_grn.shape[0], 'canonical_base': canonical_grn.shape[0],  'common_base': common_edges.shape[0], 'gene_unique_base': gene_unique.shape[0], 'isoform_unique_base': isoform_unique.shape[0]}


        # remove low quality edges
        gene_unique, abs_threshold_g, gene_unique_filter_1 = postprocessing.filter_aggregated(gene_unique, threshold_importance=0.2, threshold_frequency=10)
        isoform_unique, abs_threshold_i, isoform_unique_filter_1 = postprocessing.filter_aggregated(isoform_unique, threshold_importance=0.2, threshold_frequency=10)

        # remove implausible edges, i.e. isoform edges where there is a dominant/single isoform which
        # should have been in gene GRN as well.
        isoform_unique, isoform_unique_plausibility_filter  = postprocessing.plausibility_filtering(isoform_unique, isoform_categories)
        # Merge isoform exon/domain information and save
        isoform_unique = annotation.annotate_isoform_exclusive_edges(isoform_unique, tf_database, transcript_column='source_transcript')
        add_gene_names_and_save(isoform_unique, condi, 'unique_isoforms', results_path)


        # remove implausible edges, i.e. edges where there is a single/dominant isoform which should have been in
        # isoform GRN as well.
        gene_unique, gene_unique_plausibility_filter = postprocessing.plausibility_filtering_gene_unique(gene_unique, gene_categories)
        gene_unique = annotation.annotate_gene_exclusive_edges(gene_unique, annotation_database=tf_database, gene_transcript_mapping=gene_to_transcript_mapping)
        add_gene_names_and_save(gene_unique, condi, 'unique_genes', results_path)


        # remainder is in both GRNs. Merge the gene information to the isoform information
        merged_edges = postprocessing.create_common_edge_dataframe(common_edges)
        # Split into dominant/single isoform and ambigous (multiple isoforms)
        consistent, ambigous = postprocessing.split_by_isoform_category(merged_edges, gene_categories)
        # remove edges which have vastly different frequencies and importances from dataframe (edges in common dataframe where there is only one single or
        # a dominant isoform should have a consistent importance and frequency).
        consistent, common_dominant_filter = postprocessing.plausibility_filtering_common_edges_dominant(consistent)
        consistent = annotation.annotate_consistent_edges(consistent, tf_database, transcript_column='source_transcript')
        add_gene_names_and_save(consistent, condi, 'consistent_both', results_path)


        # From remaining ambigous dataframe, select those edges which are sig. more importantn in isoform data frame
        likely_isoform_specific, ambigous, common_isoform_likely_filter = postprocessing.find_likely_isoform_specific(ambigous)
        # add transcript annotation and save
        likely_isoform_specific = annotation.annotate_isoform_exclusive_edges(likely_isoform_specific, tf_database, transcript_column='source_transcript')
        add_gene_names_and_save(likely_isoform_specific, condi, 'likely_isoform_specific', results_path)

        
        # from remaining ambigous dataframe, select those where the gene edge is more important than any isoform edge.
        likely_gene_specific, ambigous, common_gene_likely_filter = postprocessing.find_likely_gene_specific(ambigous)
        likely_gene_specific = annotation.annotate_gene_exclusive_edges(likely_gene_specific, annotation_database=tf_database, gene_transcript_mapping=gene_to_transcript_mapping)
        add_gene_names_and_save(likely_gene_specific, condi, 'likely_gene_specific', results_path)
        
        # remaining edges are ambigous.
        add_gene_names_and_save(ambigous, condi, 'ambigous', results_path)


        correlation_collector = annotation.compute_isoform_gene_correlations(transcript_data_cp_scaled, gene_data_cp_scaled, gene_to_transcript_mapping)


        filtering_tracker['gene_unique_filter_1'] = gene_unique_filter_1
        filtering_tracker['isoform_unique_filter_1'] = isoform_unique_filter_1
        filtering_tracker['isoform_unique_plausibility_filter'] = isoform_unique_plausibility_filter
        filtering_tracker['gene_unique_plausibility_filter'] = gene_unique_plausibility_filter
        filtering_tracker['common_dominant_filter'] = common_dominant_filter
        filtering_tracker['common_isoform_likely_filter'] = common_isoform_likely_filter
        filtering_tracker['common_gene_likely_filter'] = common_gene_likely_filter

        write_dict_to_yaml(filtering_tracker, f"/data/bionets/og86asub/alternet-project/alternet-manuscript/results/{condi}_filtering_tracker.yaml")
        elapsed = {'postprocessing_elapsed': time.monotonic() - start_postprocessing}
        write_dict_to_yaml(elapsed, f"/data/bionets/og86asub/alternet-project/alternet-manuscript/results/{condi}_runtime_postprocessing.yaml")




