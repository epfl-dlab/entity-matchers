#!/usr/bin/env bash
tail --pid=5726 -f /dev/null
log_folder="/dlabdata1/leone/CAAS_out/BERT-INT/"
export LD_LIBRARY_PATH=""
# Run DBP_en_WD_en_15K
python3 -u tune_bert_int.py --json_params params_test.json \
                            --dataset /dlabdata1/leone/datasets/sampled_datasets/done/correct/DBP_en_WD_en_15K_V1_ANON \
                            --fold_folder 721_5folds/1 \
                            2>&1 | tee ${log_folder}BERT_INT_Tuning_DBP_en_WD_en_15K.log

# Run DBP_en_YG_en_15K_V1
python3 -u tune_bert_int.py --json_params params_test.json \
                            --dataset /dlabdata1/leone/datasets/sampled_datasets/done/correct/DBP_en_YG_en_15K_V1_ANON \
                            --fold_folder 721_5folds/1 \
                            2>&1 | tee ${log_folder}BERT_INT_Tuning_DBP_en_YG_en_15K.log

