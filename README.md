# AlterNet (Alternative splicing-aware gene regulatory networks)

AlterNet infers alternative-splicing aware gene-regulatory networks by implementing a pipeline based on GRNboost2 with plausibility filtering and annotation steps.
The project can be installed via pip or using pixi. For more information please refer to our preprint:



## Installation


## Minimal working example
GRN inference is a computationally heavy step. The minimal example can be run on a small subset of the data with a couple of target genes. (NOTE: it is possible that
not all files are created)


```{python}
import pandas as pd
from alternet.runners.run_alternet import alternet_pipeline
import yaml
import time
import numpy as np

import seaborn as sns
import os.path as op
import os


# The data folder shipped with the repo
basepath = "../data/"
#your results
results_path = "../results_minimal/"

appris_path = op.join(basepath, "appris_data.appris.txt")
digger_path = op.join(basepath, "digger_data.csv")
biomart_path = op.join(basepath, "biomart.txt")
tf_list_path = op.join(basepath, "allTFs_hg38.txt")

os.makedirs(results_path, exist_ok=True)

appris_df = pd.read_csv(appris_path, sep='\t')
biomart = pd.read_csv(biomart_path, sep='\t')
tf_list = pd.read_csv(tf_list_path, sep='\t', header = None)
digger_df = pd.read_csv(digger_path, sep = ',')

dcm_minimal = op.join(basepath, 'minimal_NF_magnet_prefiltered_tpm.tsv')

# transcript data
dcm_data = pd.read_csv(dcm_minimal, sep='\t')

## RUN the pipeline
alternet_pipeline(dcm_data, appris_df, digger_df, tf_list, biomart, results_path, prefix = 'NF_minimal', runs= 10)
```



## Output description
The pipeline produces six output files. Here, an isoform refers to a specific isoform, whereas a gene refers to the sum of all isoforms (total count)
- isoform-unique (edge between the TF and target has only been found for this specific isoform)
- likely-isoform speicific (the edge between the TF-isoform and the target is way more explanatory than the edge between the TF-gene and the target)
- equivalent (only 1 isoform in the dataset, and is found in both networks)
- ambigous (more thant 1 isoform, but the isoform(s) and gene have about equal explainability)
- likely-gene specific (all isoform-target edges have a lower explainability than the gene-target edge)
- gene specific (no isoform-target edge has been found)