import torch
import copy
from ..attack import Attack
import torch.nn.functional as F
import numpy as np
from .run_parser import (get_identifiers, remove_comments_and_docstrings, get_example)
from .utils import (_tokenize, get_identifier_posistions_from_code, map_chromesome,
                    is_valid_variable_name, get_substitues, is_valid_substitue,
                    select_parents, mutate, crossover)
from transformers import (RobertaForMaskedLM, RobertaTokenizer)
import random


class GeneticAlgorithm(Attack):
    '''
    Generaric algorithm attack.
    '''
    def __init__(self, model, tokenizer, lang, max_iter=10):
        super().__init__("GeneticAlgorithm", model, tokenizer, lang)
        self.item = {}
        self.max_iter = max_iter
        self.codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(self.device)
        self.tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

    def forward(self, code=None, label=None, initial_replace=None, *args, **kwargs):
        tokens = self.tokenizer([code], return_tensors="pt", truncation=True, padding='max_length').to(self.device)
        logits = self.model(**tokens).logits
        logits = F.sigmoid(logits)
        logits = torch.Tensor.cpu(logits).detach().numpy()[0]
        pred_label = np.argmax(logits)
        pred_prob = logits[pred_label]
        self.item = {
            "code": code,
            "label": int(label),
            "predict_label": int(pred_label),
            "predict_probability": float(pred_prob),
            "is_attack": False,
            "is_success": False,
            "query_time": self.query_time
        }
        if pred_label != label:
            self.item["note"] = "This code is wrongly predicted by the model."
        else:
            substitutions = self.generate_substituions(code)
            self.item["search_space"] = substitutions
            self.generate_adv(code, label, pred_label, pred_prob, substitutions, self.max_iter, initial_replace)
        return self.item

    def generate_substituions(self, code):
        identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(code, self.lang), self.lang)
        processed_code = " ".join(code_tokens)
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)

        variable_names = []
        for name in identifiers:
            if ' ' in name[0].strip():
                continue
            variable_names.append(name[0])

        sub_words = [self.tokenizer_mlm.cls_token] + sub_words[:512 - 2] + [self.tokenizer_mlm.sep_token]

        input_ids_ = torch.tensor([self.tokenizer_mlm.convert_tokens_to_ids(sub_words)])

        word_predictions = self.codebert_mlm(input_ids_.to(self.device))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, 60, -1)  # seq-len k
        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

        variable_substitue_dict = {}
        with torch.no_grad():
            orig_embeddings = self.codebert_mlm.roberta(input_ids_.to(self.device))[0]
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for tgt_word in names_positions_dict.keys():
            tgt_positions = names_positions_dict[tgt_word]  # the positions of tgt_word in code
            if not is_valid_variable_name(tgt_word, lang=self.lang):
                # if the extracted name is not valid
                continue
            all_substitues = []
            for one_pos in tgt_positions:
                if keys[one_pos][0] >= word_predictions.size()[0]:
                    continue
                substitutes = word_predictions[keys[one_pos][0]:keys[one_pos][1]]  # L, k
                word_pred_scores = word_pred_scores_all[keys[one_pos][0]:keys[one_pos][1]]

                orig_word_embed = orig_embeddings[0][keys[one_pos][0] + 1:keys[one_pos][1] + 1]

                similar_substitutes = []
                similar_word_pred_scores = []
                sims = []
                subwords_leng, nums_candis = substitutes.size()
                for i in range(nums_candis):
                    new_ids_ = copy.deepcopy(input_ids_)
                    new_ids_[0][keys[one_pos][0] + 1:keys[one_pos][1] + 1] = substitutes[:, i]
                    with torch.no_grad():
                        new_embeddings = self.codebert_mlm.roberta(new_ids_.to(self.device))[0]
                    new_word_embed = new_embeddings[0][keys[one_pos][0] + 1:keys[one_pos][1] + 1]
                    sims.append((i, sum(cos(orig_word_embed, new_word_embed)) / subwords_leng))

                sims = sorted(sims, key=lambda x: x[1], reverse=True)
                for i in range(int(nums_candis / 2)):
                    similar_substitutes.append(substitutes[:, sims[i][0]].reshape(subwords_leng, -1))
                    similar_word_pred_scores.append(word_pred_scores[:, sims[i][0]].reshape(subwords_leng, -1))

                similar_substitutes = torch.cat(similar_substitutes, 1)
                similar_word_pred_scores = torch.cat(similar_word_pred_scores, 1)

                substitutes = get_substitues(similar_substitutes,
                                             self.tokenizer_mlm,
                                             self.codebert_mlm,
                                             1,
                                             similar_word_pred_scores,
                                             0)
                all_substitues += substitutes
            all_substitues = set(all_substitues)
            for tmp_substitue in all_substitues:
                if tmp_substitue.strip() in variable_names:
                    continue
                if not is_valid_substitue(tmp_substitue.strip(), tgt_word, self.lang):
                    continue
                try:
                    variable_substitue_dict[tgt_word].append(tmp_substitue)
                except:
                    variable_substitue_dict[tgt_word] = [tmp_substitue]

        return variable_substitue_dict

    def generate_adv(self, code, label, pred_label, pred_prob, substitutions, max_iter, initial_replace=None):
        orig_prob = pred_prob
        current_prob = pred_prob
        identifiers, code_tokens = get_identifiers(code, self.lang)
        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        variable_names = list(substitutions.keys())

        if len(variable_names) == 0:  # no variable exists
            self.item["is_attack"] = False
            self.item["note"] = "This code does not include any variable."
            return self.item

        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

        variable_substitue_dict = {}

        for tgt_word in names_positions_dict.keys():
            variable_substitue_dict[tgt_word] = substitutions[tgt_word]

        fitness_values = []
        base_chromesome = {word: word for word in variable_substitue_dict.keys()}
        population = [base_chromesome]
        for tgt_word in variable_substitue_dict.keys():
            if initial_replace is None:
                initial_candidate = tgt_word
                _the_best_candidate = -1
                most_gap = 0.0
                for a_substitue in variable_substitue_dict[tgt_word]:
                    temp_code = get_example(code, tgt_word, a_substitue, self.lang)
                    new_feature = self.tokenizer([temp_code], return_tensors="pt", truncation=True,
                                                 padding='max_length').to(self.device)
                    logits = self.model(**new_feature).logits
                    self.item["query_time"] += 1
                    logits = F.sigmoid(logits)
                    logits = torch.Tensor.cpu(logits).detach().numpy()[0]
                    temp_prob = logits[label]
                    gap = current_prob - temp_prob
                    if gap > most_gap:
                        most_gap = gap
                        initial_candidate = a_substitue
                if _the_best_candidate == -1:
                    initial_candidate = tgt_word
            else:
                initial_candidate = initial_replace[tgt_word]

            temp_chromesome = copy.deepcopy(base_chromesome)
            temp_chromesome[tgt_word] = initial_candidate
            population.append(temp_chromesome)
            temp_fitness, temp_label = self.compute_fitness(temp_chromesome, pred_prob, pred_label, code)
            fitness_values.append(temp_fitness)

        cross_probability = 0.7
        for i in range(max_iter):
            _temp_mutants = []
            p = random.random()
            chromesome_1, index_1, chromesome_2, index_2 = select_parents(population)
            if p < cross_probability:  # 进行crossover
                if chromesome_1 == chromesome_2:
                    child_1 = mutate(chromesome_1, variable_substitue_dict)
                    continue
                child_1, child_2 = crossover(chromesome_1, chromesome_2)
                if child_1 == chromesome_1 or child_1 == chromesome_2:
                    child_1 = mutate(chromesome_1, variable_substitue_dict)
            else:  # 进行mutates
                child_1 = mutate(chromesome_1, variable_substitue_dict)
            _temp_mutants.append(child_1)

            mutate_fitness_values = []
            for mutant in _temp_mutants:
                _temp_code = map_chromesome(mutant, code, self.lang)
                new_feature = self.tokenizer([_temp_code], return_tensors="pt", truncation=True,
                                             padding='max_length').to(self.device)
                logits = self.model(**new_feature).logits
                self.item["query_time"] += 1
                logits = F.sigmoid(logits)
                logits = torch.Tensor.cpu(logits).detach().numpy()[0]
                temp_label = np.argmax(logits)
                if temp_label != label:
                    adv_code = mutant
                    self.item["is_attack"] = True
                    self.item["is_success"] = True
                    self.item["adv_label"] = temp_label
                    self.item["adv_code"] = adv_code
                    return self.item
                else:
                    self.item["is_attack"] = True

                _tmp_fitness = orig_prob - logits[pred_label]
                mutate_fitness_values.append(_tmp_fitness)

            for index, fitness_value in enumerate(mutate_fitness_values):
                min_value = min(fitness_values)
                if fitness_value > min_value:
                    min_index = fitness_values.index(min_value)
                    population[min_index] = _temp_mutants[index]
                    fitness_values[min_index] = fitness_value

    def compute_fitness(self, chromesome, orig_prob, orig_label, code):
        temp_code = map_chromesome(chromesome, code, self.lang)
        new_feature = self.tokenizer([temp_code], return_tensors="pt", truncation=True,
                                     padding='max_length').to(self.device)
        logits = self.model(**new_feature).logits
        logits = F.sigmoid(logits)
        logits = torch.Tensor.cpu(logits).detach().numpy()[0]
        prob = logits[orig_label]
        fitness_value = orig_prob -prob
        return fitness_value, prob