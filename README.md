# Benchmark Model for the SNOMED CT Entity Linking Challenge

This repository contains code for training the benchmark entity linking model for the [SNOMED CT Entity Linking Challenge](https://www.drivendata.org/competitions/258/competition-snomed-ct/)

If you'd like to be able to reproduce this notebook or expand upon it for your own submissions, you'll need a few things:

- A GPU machine with at least 24GB of VRAM
    - Note: It's possible to use this notebook on machines with less VRAM, but you may need to use a different base model for the CER like `deberta-v3-base`, use `LoRA` or an equivalent low-rank LLM adaptation, train with mixed precision by setting `fp16=True` in the `TrainingArguments`, and/or decrease the batch size.
- A conda environment that matches the environment provided in [`environment-gpu.yml`](https://github.com/drivendataorg/snomed-ct-entity-linking-runtime/blob/main/runtime/environment-gpu.yml) or [`conda-lock-gpu.yml`](https://github.com/drivendataorg/snomed-ct-entity-linking-runtime/blob/main/runtime/conda-lock-gpu.yml) from the challenge [runtime repository](https://github.com/drivendataorg/snomed-ct-entity-linking-runtime)
- A clone of this repository to install additional requirements (specified in `requirements.txt`) as well as leverage utilities for SNOMED CT (in `snomed_graph.py`)