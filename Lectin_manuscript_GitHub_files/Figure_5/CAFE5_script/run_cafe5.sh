#!/bin/bash

# Define input files with correct Linux paths
GENE_COUNT_FILE="/home/vishnu1988/Lectin_genecount.tsv"
TREE_FILE="/home/vishnu1988/untrametric_tree.nwk"
OUTPUT_DIR="/home/vishnu1988/cafe_output"
CONTROL_FILE="/home/vishnu1988/cafe_control.txt"

# Create the control file
cat <<EOL > "$CONTROL_FILE"
load -i $GENE_COUNT_FILE -t 10 -l $OUTPUT_DIR/report.log
tree $TREE_FILE
lambda -s
report $OUTPUT_DIR/report.cafe
EOL

# Run CAFE5 using the control file
cafe5 -c "$CONTROL_FILE"

echo "CAFE5 analysis completed. Check '$OUTPUT_DIR' for results."



