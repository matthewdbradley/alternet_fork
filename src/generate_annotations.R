#!/usr/bin/env Rscript
#
# generate_annotations.R
#
# Pulls the latest hg38 annotation files required by the AlterNet pipeline:
#   1. biomart.txt            – gene/transcript mapping
#   2. digger_data.csv        – exon & Pfam domain structure
#   3. appris_data.appris.txt – APPRIS principal-isoform annotations
#   4. allTFs_hg38.txt        – human transcription factor list
#
# Usage:
#   Rscript generate_annotations.R [output_directory]
#
# Requirements:
#   install.packages(c("biomaRt", "httr"))

suppressPackageStartupMessages({
  library(biomaRt)
  library(httr)
})

output_dir <- if (length(commandArgs(trailingOnly = TRUE)) >= 1) {
  commandArgs(trailingOnly = TRUE)[1]
} else {
  "annotation_output"
}
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl")

# ---- 1. biomart.txt (all human genes/transcripts) ----
cat("Fetching genome-wide biomart mapping...\n")
biomart_results <- getBM(
  attributes = c("ensembl_gene_id", "ensembl_gene_id_version",
                 "ensembl_transcript_id", "ensembl_transcript_id_version",
                 "hgnc_symbol", "external_transcript_name", "gene_biotype"),
  mart = ensembl
)
colnames(biomart_results) <- c("Gene stable ID", "Gene stable ID version",
                                "Transcript stable ID", "Transcript stable ID version",
                                "Gene name", "Transcript name", "Gene type")
write.table(biomart_results, file.path(output_dir, "biomart.txt"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("  biomart.txt: %d rows\n", nrow(biomart_results)))

# ---- 2. digger_data.csv (all exon + Pfam structure) ----
cat("Fetching genome-wide exon/domain structure (chunked)...\n")
all_transcripts <- unique(biomart_results[["Transcript stable ID"]])
chunks <- split(all_transcripts, ceiling(seq_along(all_transcripts) / 500))

digger_results <- do.call(rbind, lapply(seq_along(chunks), function(i) {
  if (i %% 20 == 0) cat(sprintf("  chunk %d / %d\n", i, length(chunks)))
  getBM(attributes = c("chromosome_name", "strand", "ensembl_exon_id", "rank",
                        "genomic_coding_start", "genomic_coding_end",
                        "cds_start", "cds_end", "ensembl_transcript_id",
                        "pfam", "pfam_start", "pfam_end"),
        filters = "ensembl_transcript_id", values = chunks[[i]], mart = ensembl)
}))
colnames(digger_results) <- c("Chromosome/scaffold name", "Strand",
                               "Exon stable ID", "Exon rank in transcript",
                               "Genomic coding start", "Genomic coding end",
                               "CDS start", "CDS end", "Transcript stable ID",
                               "Pfam ID", "Pfam start", "Pfam end")
write.csv(digger_results, file.path(output_dir, "digger_data.csv"), row.names = FALSE)
cat(sprintf("  digger_data.csv: %d rows\n", nrow(digger_results)))

# ---- 3. appris_data.appris.txt ----
cat("Downloading APPRIS annotations...\n")
appris_path <- file.path(output_dir, "appris_data.appris.txt")
resp <- GET("https://apprisws.bioinfo.cnio.es/pub/current_release/datafiles/homo_sapiens/GRCh38/appris_data.appris.txt",
            write_disk(appris_path, overwrite = TRUE), progress(), timeout(300))
if (http_error(resp)) stop("APPRIS download failed (HTTP ", status_code(resp), ")")
cat(sprintf("  appris_data.appris.txt written\n"))

# ---- 4. allTFs_hg38.txt ----
cat("Downloading human TF list...\n")
tf_path <- file.path(output_dir, "allTFs_hg38.txt")
resp <- GET("http://humantfs.ccbr.utoronto.ca/download/v_1.01/TF_names_v_1.01.txt",
            write_disk(tf_path, overwrite = TRUE), timeout(60))
if (http_error(resp)) stop("TF list download failed (HTTP ", status_code(resp), ")")
cat(sprintf("  allTFs_hg38.txt written\n"))

cat("\nDone! All files in:", output_dir, "\n")
