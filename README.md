# Benchmark Model for the SNOMED CT Entity Linking Challenge

This repository contains code for training the benchmark entity linking model for the [SNOMED CT Entity Linking Challenge](https://www.drivendata.org/competitions/258/competition-snomed-ct/)

If you'd like to be able to reproduce this notebook or expand upon it for your own submissions, you'll need a few things:

- A GPU machine with at least 24GB of VRAM
    - Note: It's possible to use this notebook on machines with less VRAM, but you may need to use a different base model for the CER like `deberta-v3-base`, use `LoRA` or an equivalent low-rank LLM adaptation, train with mixed precision by setting `fp16=True` in the `TrainingArguments`, and/or decrease the batch size.
- A conda environment that matches the environment provided in [`environment-gpu.yml`](https://github.com/drivendataorg/snomed-ct-entity-linking-runtime/blob/main/runtime/environment-gpu.yml) or [`conda-lock-gpu.yml`](https://github.com/drivendataorg/snomed-ct-entity-linking-runtime/blob/main/runtime/conda-lock-gpu.yml) from the challenge [runtime repository](https://github.com/drivendataorg/snomed-ct-entity-linking-runtime)
- A clone of this repository to install additional requirements (specified in `requirements.txt`) as well as leverage utilities for SNOMED CT (in `snomed_graph.py`)

## Submitting the benchmark

To create a valid submission:

1. Re-run the [notebook](entity_linker.ipynb) to generate model assets
2. Clone the [runtime repository](https://github.com/drivendataorg/snomed-ct-entity-linking-runtime/tree/main). 
3. Copy [`main.py`](main.py) as well as the model assets (the `cer_model/` folder and the `linker.pickle` file) generated from re-running the notebook into the `submission_src` folder in the cloned runtime repo
    Your `submission_src` folder should look like this:
    ```
    submission_src
    ├── cer_model
    │   ├── README.md
    │   ├── added_tokens.json
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── special_tokens_map.json
    │   ├── spm.model
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── training_args.bin
    ├── linker.pickle
    └── main.py
    ```
4. Run `make pack-submission` to generate a submission zip file. You could also follow the runtime repo instructions to generate smoke test data (`make smoke-test-data`) so you can test how your submission performs locally (`make test-submission`) before submitting to the platform.
5. Submit to the platform!

Submitting the benchmark will get you in the door, but there's so much more to explore! We hope this helps you get started in the SNOMED CT Entity Linking Challenge.