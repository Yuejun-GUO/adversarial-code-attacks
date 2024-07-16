python test.py --attack greedy --max_iter 500 --top_k 60
python test.py --attack ga --max_iter 500 --max_iter_mutant 10 --top_k 60 --cross_probability 0.7
python test.py --attack alert --max_iter 500  --max_iter_mutant 10 --top_k 60 --cross_probability 0.7
python test.py --attack mhm --max_iter 500 --n_candi 60 --prob_threshold 0.95 --top_k 60
python test.py --attack codefooler --max_iter 500 --sim_score_threshold 0.8 --synonym_num 50 --top_k 60

