import json
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from attacks import GreedyAttack, GeneticAlgorithm, ALERT, MHM, CodeAttack, CodeFooler
from datetime import datetime
import argparse
import sys
import torch
import pdb


device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
print(device)


def lang_case_insensitive_choices(value):
    value = value.lower()
    choices = ['java', 'c', 'python', 'javascript']
    if value not in choices:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value} (choose from 'java', 'c', 'python', 'javascript').")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default="codefooler",
                        choices=['greedy', 'mhm', 'ga', 'alert', 'codefooler', 'codeattack'])
    parser.add_argument("--test_dataset", type=str, default="Zaib/java-vulnerability",
                        help="Dataset card on hugginface.")
    parser.add_argument("--data_lang", type=lang_case_insensitive_choices, default="java",
                        help="Programming language of the test dataset.")
    parser.add_argument("--tgt_tokenizer", type=str, default="mrm8488/codebert-base-finetuned-detect-insecure-code",
                        help="The tokniezer of target model.")
    parser.add_argument("--tgt_model", type=str, default="mrm8488/codebert-base-finetuned-detect-insecure-code",
                        help="Model card on huggingface for vulnerability detection.")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Maximum number of attack iterations. Apply to all attacks.")
    parser.add_argument("--top_k", type=int, default=60,
                        help="Top k tokens. Apply to 'greedy', 'ga', 'alert', 'codeattack'")

    # params for Generatic Algorithm attack and the ALERT attack
    parser.add_argument("--max_iter_mutant", type=int, default=5,
                        help="Maximum number of mutation iterations.")

    # params for MHM attack
    parser.add_argument("--_n_candi", type=int, default=30,
                        help="Maximum number of variable renaming candidates.")
    parser.add_argument("--_prob_threshold", type=float, default=1,
                        help="Threshold for the acceptance rate.")

    # params for CodeFooler
    parser.add_argument("--import_score_threshold", type=float, default=-1,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold", type=float, default=0.7,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num", type=int, default=50,
                        help="Number of synonyms to extract.")

    # params for CodeAttack
    parser.add_argument("--use_imp", action='store_true',
                        help="A boolean flag to either attack random words or attack only important/vulnerable words.")
    parser.add_argument("--theta", type=float, default=0.4,
                        help="Only used in CodeAttacks. The percentage of tokens to attack.")
    args = parser.parse_args()

    dataset = load_dataset(args.test_dataset, data_files={"test": "test.csv"})
    tokenizer = AutoTokenizer.from_pretrained(args.tgt_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.tgt_model).to(device)

    if args.attack.lower() == "mhm":
        atk = MHM(model, tokenizer, args.data_lang,
                  max_iter=args.max_iter,
                  _n_candi=30,
                  _prob_threshold=1)
    elif args.attack.lower() == "greedy":
        atk = GreedyAttack(model, tokenizer, args.data_lang,
                           max_iter=args.max_iter,
                           top_k=args.top_k)
    elif args.attack.lower() == "ga":
        atk = GeneticAlgorithm(model, tokenizer, args.data_lang,
                               max_iter=args.max_iter,
                               top_k=args.top_k,
                               max_iter_mutant=args.max_iter_mutant)
    elif args.attack.lower() == "alert":
        atk = ALERT(model, tokenizer, args.data_lang,
                    max_iter=args.max_iter,
                    top_k=args.top_k,
                    max_iter_mutant=args.max_iter_mutant)
    elif args.attack.lower() == "codeattack":
        atk = CodeAttack(model, tokenizer, args.data_lang,
                         max_iter=args.max_iter,
                         use_imp=True,
                         theta=0.4)
    elif args.attack.lower() == "codefooler":
        atk = CodeFooler(model, tokenizer, args.data_lang,
                         max_iter=args.max_iter,
                         import_score_threshold=args.import_score_threshold,
                         sim_score_threshold=args.sim_score_threshold,
                         synonym_num=args.synonym_num)
    else:
        sys.exit("Attack not supported.")

    parameter_js = {
        "test_set": args.test_dataset,
        "test_size": len(dataset['test']),
        "tokenizer": args.tgt_tokenizer,
        "model": args.tgt_model,
        "start_time": datetime.now().isoformat(),
        "programming language": args.data_lang,
        "attack method": atk.name
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'result_{args.attack}_{timestamp}.json'
    with open(file_name, 'w') as fout:
        json.dump(parameter_js, fout, indent=4)
        fout.write("\n")

    for index, item in enumerate(dataset['test']):
        start_time = time.time()
        code = item['code']
        label = item['label']
        result = atk(code, label)
        result = {"index": int(index), 'run_time_seconds': time.time() - start_time, **result}
        with open(file_name, 'a') as fout:
            json.dump(result, fout, indent=4)
            fout.write("\n")
            fout.flush()


if __name__ == '__main__':
    main()
