#!/usr/bin/env bash
log_folder="/dlabdata1/leone/CAAS_out/BERT-INT/"
export LD_LIBRARY_PATH=""
# Run JA
python3 -u tune_bert_int.py --json_params params_test.json \
                            --dataset /dlabdata1/leone/datasets/sampled_datasets/done/xling/EN_JA_15K \
                            --fold_folder 721_5folds/1 \
                            --dict_path /dlabdata1/leone/datasets/descriptions/description.pkl \
                            2>&1 | tee ${log_folder}BERT_INT_Tuning_EN_JA_15K.log

# Run DE
python3 -u tune_bert_int.py --json_params params_test.json \
                            --dataset /dlabdata1/leone/datasets/sampled_datasets/done/xling/EN_DE_15K \
                            --fold_folder 721_5folds/1 \
                            --dict_path /dlabdata1/leone/datasets/descriptions/description.pkl \
                            2>&1 | tee ${log_folder}BERT_INT_Tuning_EN_DE_15K.log

# Run FR
python3 -u tune_bert_int.py --json_params params_test.json \
                            --dataset /dlabdata1/leone/datasets/sampled_datasets/done/xling/EN_FR_15K \
                            --fold_folder 721_5folds/1 \
                            --dict_path /dlabdata1/leone/datasets/descriptions/description.pkl \
                            2>&1 | tee ${log_folder}BERT_INT_Tuning_EN_FR_15K.log

