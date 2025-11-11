
import pandas as pd
from data_preprocessing import  separate_tf_genes, prepare_for_inference
from tf_utils import map_tf_ids
import db as db
from utils_network import *
from inference import *

import arboreto.algo
import yaml

import pandas as pd
from sklearn.preprocessing import StandardScaler

import time

import yaml
from typing import Any, Dict

def write_dict_to_yaml(data: Dict[str, Any], filepath: str):
    """Writes a Python dictionary to a YAML file."""

    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)






if __name__ == '__main__':

    config_file = "/data/bionets/og86asub/alternet-project/alternet/configs/MAGNet_NF.yaml"

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)


    biomart_path = '/data/bionets/og86asub/alternet-project/alternet/data/biomart.txt'
    tf_list_path = '/data/bionets/og86asub/alternet-project/alternet/data/allTFs_hg38.txt'
    appris_df = pd.read_csv(config['appris'], sep='\t')

    biomart = pd.read_csv(biomart_path, sep='\t')
    tf_list = pd.read_csv(tf_list_path, sep='\t', header = None)
    tf_list = map_tf_ids(tf_list, biomart)


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

    conditions = ['DCM', 'NF','HCM']
    
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
        gene_data_cp = standardize_dataframe(gene_data_cp)

        transcript_data_cp = transcript_data_cp.set_index('transcript_id')
        transcript_data_cp = transcript_data_cp.drop('gene_id', axis=1)
        transcript_data_cp = transcript_data_cp.T

        # scale data!!
        transcript_data_cp = standardize_dataframe(transcript_data_cp)


        start = time.monotonic()
        canonical_grn = inference(gene_data=gene_data_cp, tf_list=gene_target_names, target_names = 'all', n_runs=runs)
        end = time.monotonic()
        runtime = {'canonical': end-start}
        
        canonical_grn.to_csv(f"/data/bionets/og86asub/alternet-project/alternet-manuscript/results/{condi}_canonical.tsv", header = True)

        hybrid_data = create_hybrid_data(transcript_data_cp, gene_data_cp, tf_list)
        hybrid_tf_names = list(tf_list['Transcript stable ID'].unique())
        target_names = list(gene_data_cp.columns)

        start = time.monotonic()
        as_aware_grn = inference(gene_data=hybrid_data, tf_list=hybrid_tf_names, target_names=target_names, n_runs = runs)
        runtime['as_aware'] = time.monotonic()-start

        as_aware_grn.to_csv(f"/data/bionets/og86asub/alternet-project/alternet-manuscript/results/{condi}_as_aware.tsv", header = True) 
        write_dict_to_yaml(runtime, f"/data/bionets/og86asub/alternet-project/alternet-manuscript/results/{condi}_runtime.yaml")