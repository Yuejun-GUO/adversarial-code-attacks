# Robustness Benchmarking for Deep Learning-based Code Analysis (HELEN)

Conducting adversarial attacks is useful for testing the robustness of deep learning models. HELEN offers a comprehensive and unified platform for benchmarking robustness testing of deep learning models in a critical cybersecurity task -- software vulnerability detection. Currently, it supports six adversarial attacks that focus on the variable renaming methodology.

Table of contents
=================

<!--ts-->
   * [Environment](#requirements)
   * [Attacks, data, models](#attacks-data-models)
      * [Adversarial attacks](#adversarial-attacks)
      * [Test datasets](#test-datasets)
      * [Deep learning models](#deep-learning-models)
   * [Benchmarking results](#benchmarking-results)
      * [Results on dataset 1](#results-on-dataset-1)
      * [Results on dataset 2](#results-on-dataset-2)
   * [Acknowledge](#acknowledge)
<!--te-->

### Requirements
------------------
Required packages
```bash
tree-sitter
transformers
torch
numpy
```

If the built file "parser/my-languages.so" doesn't work for you, please rebuild as the following command:

```bash
cd parser
bash build.sh
```

### Attacks, data, models

#### Adversarial attacks
------------------------

- [Greedy attack](https://dl.acm.org/doi/10.1145/3510003.3510146): when searching for a substitute of a variable name, the greedy search methodology is utilized. The implementation is adapted from [attack-pretrain-models-code](https://github.com/soarsmu/attack-pretrain-models-of-code).

  - parameters: 
    
    - ```max_iter```: the maximum number of attack iterations.
    - ```top_k```: the [masked language modeling (MLM)](https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling) is used to generate natural substitutes of each token. This parameter determines that the top ```k``` probable ones will be used as substitutes.

  - usage:
  ```commandline
  python test.py \
  --max_iter 200 \
  --top_k 60 \
  --tgt_model mrm8488/codebert-base-finetuned-detect-insecure-code \
  --tgt_tokenizer mrm8488/codebert-base-finetuned-detect-insecure-code \
  --test_dataset google/code_x_glue_cc_defect_detection \
  --data_lang c \
  --attack greedy \
  ```

- [GA-Attack](https://dl.acm.org/doi/10.1145/3510003.3510146) is an attack based on genetic algorithms (GA). Compared the greedy attack, GA-Attack usilizes genetic algorithms to search for a substitute of a variable name. The implementation is adapted from [attack-pretrain-models-code](https://github.com/soarsmu/attack-pretrain-models-of-code).
  - parameters: 
    
    - ```max_iter```: the maximum number of attack iterations.
    - ```top_k```: the [masked language modeling (MLM)](https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling) is used to generate natural substitutes of each token. This parameter determines that the top ```k``` probable ones will be used as substitutes.
    - ```max_iter_mutant```: the maximum number of iterations to introduce mutants (variations) in the population.
    - ```cross_probability```: the probability of applying the crossover operation to two parent solutions.

  - usage:
  ```commandline
  python test.py \
  --max_iter 200 \
  --top_k 60 \
  --max_iter_mutant 10 \
  --cross_probability 0.7 \
  --tgt_model mrm8488/codebert-base-finetuned-detect-insecure-code \
  --tgt_tokenizer mrm8488/codebert-base-finetuned-detect-insecure-code \
  --test_dataset google/code_x_glue_cc_defect_detection \
  --data_lang c \
  --attack ga \
  ```

- [ALERT attack](https://dl.acm.org/doi/10.1145/3510003.3510146) is short for N**a**tura**l**n**e**ss Awa**r**e A**t**tack. Given a test code sample, this attack first applies the greedy attack to craft adversarial examples, if not successful, the GA-Attack is applied to further find the appropriate substitutes to generate adversairal examples. The implementation is adapted from [attack-pretrain-models-code](https://github.com/soarsmu/attack-pretrain-models-of-code).
  
  - parameters: 
    
    - ```max_iter```: the maximum number of attack iterations.
    - ```top_k```: the [masked language modeling (MLM)](https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling) is used to generate natural substitutes of each token. This parameter determines that the top ```k``` probable ones will be used as substitutes.
    - ```max_iter_mutant```: the maximum number of iterations to introduce mutants (variations) in the population.
    - ```cross_probability```: the probability of applying the crossover operation to two parent solutions.

  - usage:
  ```commandline
  python test.py \
  --max_iter 200 \
  --top_k 60 \
  --max_iter_mutant 10 \
  --cross_probability 0.7 \
  --tgt_model mrm8488/codebert-base-finetuned-detect-insecure-code \
  --tgt_tokenizer mrm8488/codebert-base-finetuned-detect-insecure-code \
  --test_dataset google/code_x_glue_cc_defect_detection \
  --data_lang c \
  --attack alert \
  ```

- CodeFooler attack is adapted from the [TextFooler attack](https://ojs.aaai.org/index.php/AAAI/article/view/6311) in the natural language processing (NLP) field. The implementation of CodeFooler is adapted from the [original implementation of the TextFooler attack](https://github.com/reddy-lab-code-research/CodeAttack/tree/main).

  - parameters: 
    - ```max_iter```: the maximum number of attack iterations.
    - ```top_k```: the [masked language modeling (MLM)](https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling) is used to generate natural substitutes of each token. This parameter determines that the top ```k``` probable ones will be used as substitutes.
    - ```sim_score_threshold```:  threshold that determines the minimum cosine similarity between a given variable name and its potential substitute.
    - ```synonym_num```: the number of closest subsitutes according to the cosine similarity between a given variable name and others in the vocabulary.

  - usage:
  ```commandline
  python test.py \
  --max_iter 200 \
  --top_k 60 \
  --sim_score_threshold 0.8 \
  --synonym_num 50 \
  --tgt_model mrm8488/codebert-base-finetuned-detect-insecure-code \
  --tgt_tokenizer mrm8488/codebert-base-finetuned-detect-insecure-code \
  --test_dataset google/code_x_glue_cc_defect_detection \
  --data_lang c \
  --attack codefooler \
  ```

- [CodeAttack](https://dl.acm.org/doi/10.1609/aaai.v37i12.26739) is designed to generate adversarial code for generative tasks involving code understanding and code generation. The implementation is adapted from the [original implementation of CodeAttack](https://github.com/ZZR0/CodeAttack) for testing software vulnerability detection (binary classification) models.

  - parameters: 
    - ```max_iter```: the maximum number of attack iterations.
    - ```top_k```: the [masked language modeling (MLM)](https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling) is used to generate natural substitutes of each token. This parameter determines that the top ```k``` probable ones will be used as substitutes.

  - usage:
  ```commandline
  python test.py \
  --max_iter 200 \
  --top_k 60 \
  --tgt_model mrm8488/codebert-base-finetuned-detect-insecure-code \
  --tgt_tokenizer mrm8488/codebert-base-finetuned-detect-insecure-code \
  --test_dataset google/code_x_glue_cc_defect_detection \
  --data_lang c \
  --attack codeattack \
  ```

- [MHM attack](https://ojs.aaai.org/index.php/AAAI/article/view/5469) is short for Metropolis-Hastings Modifier attack. The implementation is adapted from [attack-pretrain-models-code](https://github.com/soarsmu/attack-pretrain-models-of-code).

  - parameters: 
    - ```max_iter```: the maximum number of attack iterations.
    - ```top_k```: the [masked language modeling (MLM)](https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling) is used to generate natural substitutes of each token. This parameter determines that the top ```k``` probable ones will be used as substitutes.
    - ```n_candi```: the number of candidate substitutes.
    - ```prob_threshold```: the threshold of the acceptance rate.

  - usage:
  ```commandline
  python test.py \
  --max_iter 200 \
  --top_k 60 \
  --n_candi 60 \
  --prob_threshold 0.95 \
  --tgt_model mrm8488/codebert-base-finetuned-detect-insecure-code \
  --tgt_tokenizer mrm8488/codebert-base-finetuned-detect-insecure-code \
  --test_dataset google/code_x_glue_cc_defect_detection \
  --data_lang c \
  --attack mhm \
  ```
#### Test datasets
-----------------

- [Zaib/java-vulnerability](https://huggingface.co/datasets/Zaib/java-vulnerability) is a dataset for software vulnerability detection. The test set includes 319 code functions in Java.

- [google/code_x_glue_cc_defect_detection](https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection) is also an open dataset on Hugging Face for software vulnerability detection. Its test set inclues 2,732 code functions in C.

#### Deep learning models
-------------------------

- [mrm8488/codebert-base-finetuned-detect-insecure-code](https://huggingface.co/mrm8488/codebert-base-finetuned-detect-insecure-code) is a finetuned model on CodeBERT using the [CodeXGLUE - Devign dataset](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection) for software vulnerability detection.

- [claudios/VulBERTa-MLP-Draper](https://huggingface.co/claudios/VulBERTa-MLP-Draper) is a finetuned model on the [VulBERTa model](https://github.com/ICL-ml4csec/VulBERTa/tree/main) using the [claudios/Draper dataset](https://huggingface.co/datasets/claudios/Draper) for software vulnerability detection.

- [claudios/VulBERTa-MLP-D2A](https://huggingface.co/claudios/VulBERTa-MLP-D2A) is a finetuned model on the [VulBERTa model](https://github.com/ICL-ml4csec/VulBERTa/tree/main) using the [claudios/D2A dataset](https://huggingface.co/datasets/claudios/D2A)  for software vulnerability detection.

- [claudios/VulBERTa-MLP-ReVeal](https://huggingface.co/claudios/VulBERTa-MLP-ReVeal) is a fitunetuned model on the [VulBERTa model](https://github.com/ICL-ml4csec/VulBERTa/tree/main) using the [claudios/ReVeal](https://huggingface.co/datasets/claudios/ReVeal) for software vulnerability detection.


### Benchmarking results 
------------------------

#### Results on dataset 1
* [Zaib/java-vulnerability](https://huggingface.co/datasets/Zaib/java-vulnerability)


<table>
  <tr>
    <td rowspan="2"><strong>Model card on Hugging Face</strong></td>
    <td rowspan="2" style="text-align: center;"><strong>Accuracy (%)</strong></td>
    <td colspan="6" style="text-align: center;"><strong>Adversarial robustness (%)</strong></td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>Greedy attack</strong></td>
    <td style="text-align: center;"><strong>GA-Attack</strong></td>
    <td style="text-align: center;"><strong>ALERT attack</strong></td>
    <td style="text-align: center;"><strong>CodeFooler attack</strong></td>
    <td style="text-align: center;"><strong>CodeAttack</strong></td>
    <td style="text-align: center;"><strong>MHM attack</strong></td>
  </tr>
  <tr>
    <td>mrm8488/codebert-base-finetuned-detect-insecure-code</td>
    <td style="text-align: center;">32.29</td>
    <td style="text-align: center;">98.91</td>
    <td style="text-align: center;">100</td>
    <td style="text-align: center;">98.91</td>
    <td style="text-align: center;">98.91</td>
    <td style="text-align: center;">98.91</td>
    <td style="text-align: center;">98.91</td>
  </tr>
  <tr>
    <td>claudios/VulBERTa-MLP-Draper</td>
    <td style="text-align: center;">67.71</td>
    <td style="text-align: center;">100</td>
    <td style="text-align: center;">100</td>
    <td style="text-align: center;">100</td>
    <td style="text-align: center;">100</td>
    <td style="text-align: center;">100</td>
    <td style="text-align: center;">100</td>
  </tr>
  <tr>
    <td>claudios/VulBERTa-MLP-D2A</td>
    <td style="text-align: center;">57.05</td>
    <td style="text-align: center;">82.48</td>
    <td style="text-align: center;">95.62</td>
    <td style="text-align: center;">85.40</td>
    <td style="text-align: center;">83.94</td>
    <td style="text-align: center;">79.56</td>
    <td style="text-align: center;">100</td>
  </tr>
  <tr>
    <td>claudios/VulBERTa-MLP-ReVeal</td>
    <td style="text-align: center;">64.89</td>
    <td style="text-align: center;">59.49</td>
    <td style="text-align: center;">87.97</td>
    <td style="text-align: center;">67.72</td>
    <td style="text-align: center;">67.72</td>
    <td style="text-align: center;">56.33</td>
    <td style="text-align: center;">71.52</td>
  </tr>
</table>


#### Results on dataset 2
* [google/code_x_glue_cc_defect_detection](https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection)

<table>
  <tr>
    <td rowspan="2"><strong>Model card on Hugging Face</strong></td>
    <td rowspan="2" style="text-align: center;"><strong>Accuracy (%)</strong></td>
    <td colspan="6" style="text-align: center;"><strong>Adversarial robustness (%)</strong></td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>Greedy attack</strong></td>
    <td style="text-align: center;"><strong>GA-Attack</strong></td>
    <td style="text-align: center;"><strong>ALERT attack</strong></td>
    <td style="text-align: center;"><strong>CodeFooler attack</strong></td>
    <td style="text-align: center;"><strong>CodeAttack</strong></td>
    <td style="text-align: center;"><strong>MHM attack</strong></td>
  </tr>
  <tr>
    <td>mrm8488/codebert-base-finetuned-detect-insecure-code</td>
    <td style="text-align: center;">47.14</td>
    <td style="text-align: center;">97.63</td>
    <td style="text-align: center;">99.29</td>
    <td style="text-align: center;">97.71</td>
    <td style="text-align: center;">97.55</td>
    <td style="text-align: center;">96.60</td>
    <td style="text-align: center;">98.26</td>
  </tr>
  <tr>
    <td>claudios/VulBERTa-MLP-Draper</td>
    <td style="text-align: center;">54.06</td>
    <td style="text-align: center;">100</td>
    <td style="text-align: center;">100</td>
    <td style="text-align: center;">100</td>
    <td style="text-align: center;">100</td>
    <td style="text-align: center;">100</td>
    <td style="text-align: center;">100</td>
  </tr>
  <tr>
    <td>claudios/VulBERTa-MLP-D2A</td>
    <td style="text-align: center;">51.02</td>
    <td style="text-align: center;">68.06</td>
    <td style="text-align: center;">89.72</td>
    <td style="text-align: center;">75.26</td>
    <td style="text-align: center;">71.81</td>
    <td style="text-align: center;">56.83</td>
    <td style="text-align: center;">79.22</td>
  </tr>
  <tr>
    <td>claudios/VulBERTa-MLP-ReVeal</td>
    <td style="text-align: center;">54.14</td>
    <td style="text-align: center;">88.77</td>
    <td style="text-align: center;">96.76</td>
    <td style="text-align: center;">91.87</td>
    <td style="text-align: center;">89.33</td>
    <td style="text-align: center;">77.61</td>
    <td style="text-align: center;">91.67</td>
  </tr>
</table>


---------------------------------------------------------------
### Acknowledge
This project is supported by the Google Cloud Research Credits.