#!/bin/bash

SRR_ID=SRR30101293
prefetch $SRR_ID

fasterq-dump $SRR_ID -O ./myc-yang --split-files
