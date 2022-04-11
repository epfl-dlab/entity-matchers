# Reproducibility Notes
## Preliminary Steps
You need to follow all the "Installation process" steps into our README: https://github.com/epfl-dlab/entity-matchers#installation-process.  
Please use the given environment names, because the reproducibility scripts assume the environments are called in this way.  

### Datasets and word embeddings
You need to download datasets and word embeddings as specified into the "Installation process" steps.  
The dataset path must be provided to the reproduction scripts as a parameter, while the word embedding path (used only by RDGCN) must be overwritten for all the RDGCN args into the [args_best](https://github.com/epfl-dlab/entity-matchers/tree/master/src/experiments/args_best) folder.  
You can use the [replace_word_embeddings_path.py](https://github.com/epfl-dlab/entity-matchers/blob/master/src/preprocess_datasets/replace_word_embeddings_path.py) script to do it easily. You only need to provide the "new_path" parameter.

## Reproducing Table 5
Table 5 can be reproduced by using the following scripts:
- [run_experiments_table5_a.py](https://github.com/epfl-dlab/entity-matchers/blob/master/src/experiments/run_experiments_table5_a.py) to reproduce row a.
- [run_experiments_table5_b.py](https://github.com/epfl-dlab/entity-matchers/blob/master/src/experiments/run_experiments_table5_b.py) to reproduce row b.
- [run_experiments_table5_c.py](https://github.com/epfl-dlab/entity-matchers/blob/master/src/experiments/run_experiments_table5_c.py) to reproduce row c.
- [run_experiments_table5_d_AttRealEA_All.py](https://github.com/epfl-dlab/entity-matchers/blob/master/src/experiments/run_experiments_table5_d_AttRealEA_All.py) to reproduce row d (AttRealEA_All side).
- [run_experiments_table5_d_AttRealEA_None.py](https://github.com/epfl-dlab/entity-matchers/blob/master/src/experiments/run_experiments_table5_d_AttRealEA_None.py) to reproduce row d (AttRealEA_None side).

For all this scripts, you need to provide:
- "root_dataset": Path to the root of the datasets folder (without slash in the end), i.e. where you decompressed the downloaded zip
- "gpu" (Optional): GPU id to use, if not provided the CPU will be used
