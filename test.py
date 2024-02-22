import json
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from attacks import Greedy, GeneticAlgorithm, ALERT, MHM
from datetime import datetime
import argparse
import sys
import torch
import pdb


device = torch.device("cuda:0" if torch.cuda.is_available() else "mps") #cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default="greedy")
    args = parser.parse_args()

    dataset = load_dataset("Zaib/java-vulnerability", data_files={"test": "test.csv"})
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code')
    model = AutoModelForSequenceClassification.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code').to(device)

    if args.attack == "mhm":
        atk = MHM(model, tokenizer, 'java', max_iter=5, _n_candi=30, _prob_threshold=1)
    elif args.attack == "greedy":
        atk = Greedy(model, tokenizer, 'java')
    elif args.attack == "ga":
        atk = GeneticAlgorithm(model, tokenizer, 'java', max_iter=5)
    elif args.attack == "gi":
        atk = ALERT(model, tokenizer, 'java', max_iter=5)
    else:
        sys.exit()

    parameter_js = {
        "test_set": "Zaib/java-vulnerability",
        "test_size": len(dataset['test']),
        "tokenizer": 'mrm8488/codebert-base-finetuned-detect-insecure-code',
        "model": 'mrm8488/codebert-base-finetuned-detect-insecure-code',
        "start_time": datetime.now().isoformat(),
        "programming language": 'java',
        "attack method": atk.name
    }
    with open(f'result_{args.attack}.json', 'w') as fout:
        json.dump(parameter_js, fout, indent=4)
        fout.write("\n")

    for index, item in enumerate(dataset['test']):
        start_time = time.time()
        code = item['code']
        label = item['label']
        result = atk(code, label)
        result = {"index": int(index), 'run_time_seconds': time.time() - start_time, **result}
        with open(f'result_{args.attack}.json', 'a') as fout:
            json.dump(result, fout, indent=4)
            fout.write("\n")
            fout.flush()


if __name__ == '__main__':
    main()
