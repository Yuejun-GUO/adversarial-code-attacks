#!/bin/bash

tag_models=("mrm8488/codebert-base-finetuned-detect-insecure-code" "claudios/VulBERTa-MLP-ReVeal")
tgt_datasets=("Zaib/java-vulnerability" "google/code_x_glue_cc_defect_detection")
attcks=("greedy" "ga" "codefooler" "codeattack" "alert" "mhm")
max_iter=200
top_k=60

declare -A datasets_lang_map=(
    ["mrm8488/codebert-base-finetuned-detect-insecure-code"]="java"
    ["claudios/VulBERTa-MLP-ReVeal"]="c"
)

declare -A atk_params_map
atk_params_map["greedy"]="--max_iter $max_iter --top_k $top_k"
atk_params_map["ga"]="--max_iter $max_iter --top_k $top_k --max_iter_mutant 10 --cross_probability 0.7"
atk_params_map["codefooler"]="--max_iter $max_iter --top_k $top_k --sim_score_threshold 0.8 --synonym_num 50"
atk_params_map["codeattack"]="--max_iter $max_iter --top_k $top_k"
atk_params_map["alert"]="--max_iter $max_iter --top_k $top_k --max_iter_mutant 10 --cross_probability 0.7"
atk_params_map["mhm"]="--max_iter $max_iter --top_k $top_k --n_candi 60 --prob_threshold 0.95"



for tag_model in "${tag_models[@]}"
do
    for atk in "${attcks[@]}"
    do
        params=${atk_params_map[$atk]}
        for tgt_dataset in "${tgt_datasets[@]}"
        do
            lang=${datasets_lang_map[$tgt_dataset]}
            python test.py --max_iter $max_iter --top_k $top_k --tgt_model $tgt_model --test_dataset $tgt_dataset --data_lang $lang --attack $atk $params || continue
            echo "python test.py --max_iter $max_iter --top_k $top_k --tgt_model $tgt_model --test_dataset $tgt_dataset --data_lang $lang --attack $atk $params" || continue
        done
    done
done