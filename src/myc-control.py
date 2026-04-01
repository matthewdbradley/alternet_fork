import pandas as pd
from alternet.runners.run_alternet import alternet_pipeline
import yaml
import time
import numpy as np

import seaborn as sns
import os.path as op
import os

def main():

    # The data folder shipped with the repo
    basepath = "../data/"
    results_path = "../results/myc-control/"

    appris_path = op.join(basepath, "appris_data.appris.txt")
    digger_path = op.join(basepath, "digger_data.csv")
    biomart_path = op.join(basepath, "biomart.txt")
    tf_list_path = op.join(basepath, "allTFs_hg38.txt")

    os.makedirs(results_path, exist_ok=True)

    appris_df = pd.read_csv(appris_path, sep="\t")
    biomart = pd.read_csv(biomart_path, sep="\t")
    tf_list = pd.read_csv(tf_list_path, sep="\t", header=None)
    digger_df = pd.read_csv(digger_path, sep=",")

    dcm_minimal = op.join(basepath, "myc-yang", "salmon_output", "merged_expression.tsv")

    # transcript data
    dcm_data = pd.read_csv(dcm_minimal, sep="\t")

    # RUN the pipeline
    alternet_pipeline(
        dcm_data,
        appris_df,
        digger_df,
        tf_list,
        biomart,
        results_path,
        prefix="NF_minimal",
        runs=10,
    )


if __name__ == "__main__":
    main()
