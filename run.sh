python test.py --attack greedy --max_iter 200 --top_k 60 --test_dataset Zaib/java-vulnerability --data_lang java --tgt_model mrm8488/codebert-base-finetuned-detect-insecure-code
python test.py --attack greedy --max_iter 200 --top_k 60 --test_dataset google/code_x_glue_cc_defect_detection --tgt_model claudios/VulBERTa-MLP-ReVeal --data_lang c
python test.py --attack ga --max_iter 200 --max_iter_mutant 10 --top_k 60 --cross_probability 0.7  --data_lang java --tgt_model mrm8488/codebert-base-finetuned-detect-insecure-code
python test.py --attack ga --max_iter 200 --max_iter_mutant 10 --top_k 60 --cross_probability 0.7 --test_dataset google/code_x_glue_cc_defect_detection --tgt_model claudios/VulBERTa-MLP-ReVeal --data_lang c

 
python test.py --attack alert --max_iter 200  --max_iter_mutant 10 --top_k 60 --cross_probability 0.7  --test_dataset Zaib/java-vulnerability --data_lang java --tgt_model mrm8488/codebert-base-finetuned-detect-insecure-code
python test.py --attack mhm --max_iter 200 --n_candi 60 --prob_threshold 0.95 --top_k 60 --test_dataset Zaib/java-vulnerability --data_lang java --tgt_model mrm8488/codebert-base-finetuned-detect-insecure-code
#python test.py --attack codefooler --max_iter 200 --sim_score_threshold 0.8 --synonym_num 50 --top_k 60

python test.py --attack alert --max_iter 200  --max_iter_mutant 10 --top_k 60 --cross_probability 0.7 --test_dataset google/code_x_glue_cc_defect_detection --tgt_model claudios/VulBERTa-MLP-ReVeal --data_lang c
python test.py --attack mhm --max_iter 200 --n_candi 60 --prob_threshold 0.95 --top_k 60 --test_dataset google/code_x_glue_cc_defect_detection --tgt_model claudios/VulBERTa-MLP-ReVeal --data_lang c
#python test.py --attack codefooler --max_iter 200 --sim_score_threshold 0.8 --synonym_num 50 --top_k 60 --test_dataset google/code_x_glue_cc_defect_detection --tgt_model claudios/VulBERTa-MLP-ReVeal --data_lang c



