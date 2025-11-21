# AlterNet (Alternative splicing-aware gene regulatory networks)

AlterNet infers alternative-splicing aware gene-regulatory networks by implementing a pipeline based on GRNboost2 with plausibility filtering and annotation steps.
The project can be installed via pip or using pixi. For more information please refer to our preprint:



## Installation


## Minimal working example
GRN inference is a computationally heavy step. The minimal example can be run on a small subset of the data with 3 target genes.


## Output description
The pipeline produces six output files. Here, an isoform refers to a specific isoform, whereas a gene refers to the sum of all isoforms (total count)
- isoform-unique (edge between the TF and target has only been found for this specific isoform)
- likely-isoform speicific (the edge between the TF-isoform and the target is way more explanatory than the edge between the TF-gene and the target)
- equivalent (only 1 isoform in the dataset, and is found in both networks)
- ambigous (more thant 1 isoform, but the isoform(s) and gene have about equal explainability)
- likely-gene specific (all isoform-target edges have a lower explainability than the gene-target edge)
- gene specific (no isoform-target edge has been found)