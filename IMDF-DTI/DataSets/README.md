# Benchmark inputs

Place `DrugBank.txt`, `BIOSNAP.txt`, and `Davis.txt` in this directory, preserving
their original row order and exact file contents. Each nonempty line ends with:

```text
... SMILES PROTEIN_SEQUENCE BINARY_LABEL
```

Use the processed files from your archived experiments, or obtain the benchmark
files from the subject to the dataset providers' access/reuse terms. Compare their SHA-256 hashes
using `python check_data.py`; a matching dataset name alone does not establish an
identical split. This package does not redistribute raw DrugBank/BioSNAP/Davis data.

For each dataset, reuse your trusted `Preprocessed/{dataset}-protein-new.pkl` and
`Preprocessed/{dataset}-ligand-hi-new.pkl`, or run `python prepare_data.py DATASET`.
Feature generation can require substantial GPU memory and time. Do not load
untrusted pickle/joblib caches or untrusted checkpoints. Large data/weight files
are deliberately excluded from Git; publish approved data separately with a stable
link if permitted, or document the authorized acquisition route.
