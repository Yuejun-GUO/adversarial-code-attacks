import json
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, Value
from attacks import GreedyAttack, GeneticAlgorithm, ALERT, MHM, CodeAttack, CodeFooler
from datetime import datetime
import argparse
import os
import sys
import torch
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")


def lang_case_insensitive_choices(value):
    value = value.lower()
    choices = ['java', 'c', 'python', 'javascript']
    if value not in choices:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value} (choose from 'java', 'c', 'python', 'javascript').")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default="greedy",
                        choices=['greedy', 'mhm', 'ga', 'alert', 'codefooler', 'codeattack'])
    parser.add_argument("--test_dataset", type=str, default="Zaib/java-vulnerability",
                        help="Dataset card on hugginface.")
    parser.add_argument("--data_lang", type=lang_case_insensitive_choices, default="java",
                        help="Programming language of the test dataset.")
    parser.add_argument("--tgt_tokenizer", type=str, default="mrm8488/codebert-base-finetuned-detect-insecure-code",
                        help="The tokniezer of target model.")
    parser.add_argument("--tgt_model", type=str, default="mrm8488/codebert-base-finetuned-detect-insecure-code",
                        help="Model card on huggingface for vulnerability detection.")
    parser.add_argument("--block_size", default=510, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training.")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Apply to all attacks. Maximum number of attack iterations.")
    parser.add_argument("--top_k", type=int, default=60,
                        help="Apply to all attacks. Top k tokens.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="The dir to store results.")

    # params for Genetic Algorithm attack and the ALERT attack
    parser.add_argument("--max_iter_mutant", type=int, default=5,
                        help="Only used in the Genetic and ALERT attack. Maximum number of mutation iterations.")
    parser.add_argument("--cross_probability", type=float, default=0.7,
                        help="Only used in the Genetic and ALERT attack. Cross probability in the genetic algorithm "
                             "for mutation.")

    # params for MHM attack
    parser.add_argument("--n_candi", type=int, default=30,
                        help="Only used in MHM attack. Maximum number of variable renaming candidates.")
    parser.add_argument("--prob_threshold", type=float, default=1,
                        help="Only used in MHM attack. Threshold for the acceptance rate.")

    # params for CodeFooler
    parser.add_argument("--import_score_threshold", type=float, default=-1,
                        help="Only used in CodeFooler. Required mininum importance score.")
    parser.add_argument("--sim_score_threshold", type=float, default=0.7,
                        help="Only used in CodeFooler. Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num", type=int, default=50,
                        help="Only used in CodeFooler. Number of synonyms to extract.")

    args = parser.parse_args()

    dataset = load_dataset(args.test_dataset, split='test')
    if 'func' in dataset.column_names:
        dataset = dataset.rename_column('func', 'code')

    if 'target' in dataset.column_names:
        dataset = dataset.rename_column('target', 'label')
    
    dataset = dataset.map(lambda example: {'label': min(example['label'], 1)})
    dataset = dataset.cast_column('label',  Value("int64"))

    if not args.tgt_tokenizer:
        args.tgt_tokenizer = args.tgt_model
    tokenizer = AutoTokenizer.from_pretrained(args.tgt_tokenizer, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.tgt_model, trust_remote_code=True).to(device)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    if args.attack.lower() == "mhm":
        atk = MHM(model, tokenizer, args.data_lang,
                  max_iter=args.max_iter,
                  top_k=args.top_k,
                  _n_candi=args.n_candi,
                  _prob_threshold=args.prob_threshold,
                  block_size=args.block_size
                  )
        atk_params = {
            "max_iter": args.max_iter,
            "top_k": args.top_k,
            "n_candi": args.n_candi,
            "prob_threshold": args.prob_threshold,
        }
    elif args.attack.lower() == "greedy":
        atk = GreedyAttack(model, tokenizer, args.data_lang,
                           max_iter=args.max_iter,
                           top_k=args.top_k,
                           block_size=args.block_size
                           )
        atk_params = {
            "max_iter": args.max_iter,
            "top_k": args.top_k,
        }
    elif args.attack.lower() == "ga":
        atk = GeneticAlgorithm(model, tokenizer, args.data_lang,
                               max_iter=args.max_iter,
                               top_k=args.top_k,
                               max_iter_mutant=args.max_iter_mutant,
                               cross_probability=args.cross_probability,
                               block_size=args.block_size
                               )
        atk_params = {
            "max_iter": args.max_iter,
            "top_k": args.top_k,
            "max_iter_mutant": args.max_iter_mutant,
            "cross_probability": args.cross_probability,
        }
    elif args.attack.lower() == "alert":
        atk = ALERT(model, tokenizer, args.data_lang,
                    max_iter=args.max_iter,
                    top_k=args.top_k,
                    max_iter_mutant=args.max_iter_mutant,
                    cross_probability=args.cross_probability,
                    block_size=args.block_size
                    )
        atk_params = {
            "max_iter": args.max_iter,
            "top_k": args.top_k,
            "max_iter_mutant": args.max_iter_mutant,
            "cross_probability": args.cross_probability,
        }
    elif args.attack.lower() == "codeattack":
        atk = CodeAttack(model, tokenizer, args.data_lang,
                         max_iter=args.max_iter,
                         block_size=args.block_size
                         )
        atk_params = {
            "max_iter": args.max_iter,
            "top_k": args.top_k,
        }
    elif args.attack.lower() == "codefooler":
        atk = CodeFooler(model, tokenizer, args.data_lang,
                         max_iter=args.max_iter,
                         import_score_threshold=args.import_score_threshold,
                         sim_score_threshold=args.sim_score_threshold,
                         synonym_num=args.synonym_num,
                         block_size=args.block_size
                         )
        atk_params = {
            "max_iter": args.max_iter,
            "import_score_threshold": args.import_score_threshold,
            "sim_score_threshold": args.sim_score_threshold,
            "synonym_num": args.synonym_num,
        }
    else:
        sys.exit("Attack not supported.")

    parameter_js = {
        "test_set": args.test_dataset,
        "test_size": len(dataset),
        "tokenizer": args.tgt_tokenizer,
        "model": args.tgt_model,
        "block_size": args.block_size,
        "start_time": datetime.now().isoformat(),
        "programming language": args.data_lang,
        "attack method": atk.name,
        "atk_params": atk_params,
    }

    print(parameter_js)
    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'{args.results_dir}/result_{args.attack}_{timestamp}.json'

    results_js = {}
    for index, item in enumerate(dataset):
        print(index)
        start_time = time.time()
        code = item['code']
        label = item['label']
        result = atk(code, label)
        result = {"index": int(index), 'run_time_seconds': time.time() - start_time, **result}
        results_js[int(index)] =  result

    with open(file_name, 'w') as fout:
        json.dump({"params": parameter_js, "results": results_js}, fout, indent=4)
        fout.write("\n")


if __name__ == '__main__':
    main()
