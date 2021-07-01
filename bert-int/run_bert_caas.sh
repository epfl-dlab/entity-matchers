#!/usr/bin/env bash
log_folder="/dlabdata1/leone/CAAS_out/BERT-INT/"
# Run JA
cd basic_bert_unit/ \
    && python -u main.py 2>&1 | tee ${log_folder}basic_unit_ja.log \
    && cd ../interaction_model/ \
    && ./run.sh 2>&1 | tee ${log_folder}interaction_model_ja.log

# Run ZH
cd ../basic_bert_unit/ \
    && sed -i '0,/ja/{s/ja/zh/}' Param.py \
    && python -u main.py 2>&1 | tee ${log_folder}basic_unit_zh.log \
    && cd ../interaction_model/ \
    && sed -i '0,/ja/{s/ja/zh/}' Param.py \
    && ./run.sh 2>&1 | tee ${log_folder}interaction_model_zh.log

# Run FR
cd ../basic_bert_unit/ \
    && sed -i '0,/zh/{s/zh/fr/}' Param.py \
    && python -u main.py 2>&1 | tee ${log_folder}basic_unit_fr.log \
    && cd ../interaction_model/ \
    && sed -i '0,/zh/{s/zh/fr/}' Param.py \
    && ./run.sh 2>&1 | tee ${log_folder}interaction_model_fr.log

# Reset to ja
cd ../basic_bert_unit/ \
    && sed -i '0,/fr/{s/fr/ja/}' Param.py \
    && cd ../interaction_model/ \
    && sed -i '0,/fr/{s/fr/ja/}' Param.py