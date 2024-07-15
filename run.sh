python test.py --attack greedy --max_iter 500 --top_k 60
python test.py --attack ga --max_iter 500 --max_iter_mutant 10 --top_k 60
python test.py --attack alert --max_iter 500  --max_iter_mutant 10 --top_k 60
python test.py --attack mhm --max_iter 500 --_n_candi 500 --_prob_threshold 0.95 --top_k 60
python test.py --attack codefooler --max_iter 500 --sim_score_threshold 0.8 --synonym_num 100 --top_k 60

