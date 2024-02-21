import json
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from attacks import Greedy
from datetime import datetime
import pdb


def main():
    dataset = load_dataset("Zaib/java-vulnerability", data_files={"test": "test.csv"})
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code')
    model = AutoModelForSequenceClassification.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code').to('mps')

    atk = Greedy(model, tokenizer, 'java')

    model.eval()
    parameter_js = {
        "test_set": "Zaib/java-vulnerability",
        "tokenizer": 'mrm8488/codebert-base-finetuned-detect-insecure-code',
        "model": 'mrm8488/codebert-base-finetuned-detect-insecure-code',
        "start_time": datetime.now().isoformat(),
        "programming language": 'java',
        "attack method": 'Greedy attack'
    }
    with open('attack_result.json', 'w') as fout:
        json.dump(parameter_js, fout, indent=4)
        fout.write("\n")

    for index, item in enumerate(dataset['test']):
        if index < 214:
            continue
        start_time = time.time()
        code = item['code']
        label = item['label']
        result = atk(code, label)
        result["index"] = int(index)
        result['run_time_seconds'] = time.time() - start_time
        if index == 216:
            break
        with open('attack_result.json', 'a') as fout:
            # fout.write(json.dumps(result) + "\n")
            json.dump(result, fout, indent=4)
            fout.write("\n")
            fout.flush()


if __name__ == '__main__':
    main()
