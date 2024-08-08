#!/bin/bash

tgt_models=("mrm8488/codebert-base-finetuned-detect-insecure-code" "claudios/VulBERTa-MLP-ReVeal")
attcks=("greedy" "ga" "codefooler" "codeattack" "alert" "mhm")
max_iter=200
top_k=60


declare -A atk_params_map
atk_params_map["greedy"]="--max_iter $max_iter --top_k $top_k"
atk_params_map["ga"]="--max_iter $max_iter --top_k $top_k --max_iter_mutant 10 --cross_probability 0.7"
atk_params_map["codefooler"]="--max_iter $max_iter --top_k $top_k --sim_score_threshold 0.8 --synonym_num 50"
atk_params_map["codeattack"]="--max_iter $max_iter --top_k $top_k"
atk_params_map["alert"]="--max_iter $max_iter --top_k $top_k --max_iter_mutant 10 --cross_probability 0.7"
atk_params_map["mhm"]="--max_iter $max_iter --top_k $top_k --n_candi 60 --prob_threshold 0.95"



for tgt_model in "${tgt_models[@]}"
do
    for atk in "${attcks[@]}"
    do
        params=${atk_params_map[$atk]}
        echo "python test.py --max_iter $max_iter --top_k $top_k --tgt_model $tgt_model --test_dataset Zaib/java-vulnerability --data_lang java --attack $atk $params" || continue
        python test.py --max_iter $max_iter --top_k $top_k --tgt_model $tgt_model --test_dataset Zaib/java-vulnerability --data_lang java --attack $atk $params || continue
    done
done

for tgt_model in "${tgt_models[@]}"
do
    for atk in "${attcks[@]}"
    do
        params=${atk_params_map[$atk]}
        echo "python test.py --max_iter $max_iter --top_k $top_k --tgt_model $tgt_model --test_dataset google/code_x_glue_cc_defect_detection --data_lang c --attack $atk $params" || continue
        python test.py --max_iter $max_iter --top_k $top_k --tgt_model $tgt_model --test_dataset google/code_x_glue_cc_defect_detection --data_lang c --attack $atk $params || continue
    done
done