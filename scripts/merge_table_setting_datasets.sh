#!/bin/bash
# Merge table_setting_dataset_1.hdf5 ... _10.hdf5 into table_setting_dataset.hdf5
# Usage: run from repo root, or pass -o /path/to/output.hdf5

OUTPUT="${1:-./datasets/table_setting_dataset.hdf5}"

DATASETS=(
    "./datasets/table_setting_dataset_1.hdf5"
    "./datasets/table_setting_dataset_2.hdf5"
    "./datasets/table_setting_dataset_3.hdf5"
    "./datasets/table_setting_dataset_4.hdf5"
    "./datasets/table_setting_dataset_5.hdf5"
    "./datasets/table_setting_dataset_6.hdf5"
    "./datasets/table_setting_dataset_7.hdf5"
    "./datasets/table_setting_dataset_8.hdf5"
    "./datasets/table_setting_dataset_9.hdf5"
    "./datasets/table_setting_dataset_10.hdf5"
)

echo "Merging table setting datasets 1-10 -> $OUTPUT"
python scripts/merge_hdf5_datasets.py "${DATASETS[@]}" -o "$OUTPUT"
