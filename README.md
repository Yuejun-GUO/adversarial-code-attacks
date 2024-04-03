# Robustness Benchmarking for Deep Learning-based Code Analysis (HELEN)

Adversarial attacks are one of the biggest challenges facing deep learning (DL)-based AI solutions, e.g., a panda image can be misclassified as a gibbon with an imperceptible perturbation. On the other hand, adversarial attacks have become a useful concept for testing the robustness of DL models when they are used for software code analysis such as vulnerability detection. HELEN offers a comprehensive and unified platform for benchmarking robustness testing of deep learning (DL) models in source code analysis. This tool supports four existing adversarial attacks: ALERT, MHM, Greedy, Generic Algorithm.

### Test example
The [data](https://huggingface.co/datasets/Zaib/java-vulnerability) and [model](https://huggingface.co/mrm8488/codebert-base-finetuned-detect-insecure-code) for testing are publicly available on HuggingFace.
```commandline
python test.py --attack greedy
```

### Acknowledge
This project is supported by the Google Cloud Research Credits.