import torch
import copy
import torch.nn.functional as F
import numpy as np
from ..attack import Attack
from .run_parser import (get_identifiers, remove_comments_and_docstrings,
                         get_example, )
from .utils import (_tokenize, get_identifier_posistions_from_code,
                    is_valid_variable_name, get_substitues, is_valid_substitue,
                    get_importance_score_code)
from transformers import (RobertaForMaskedLM, RobertaTokenizer, RobertaConfig)


class CodeAttack(Attack):
    '''
    CodeAttack
    Paper: CodeAttack: Code-based Adversarial Attacks for Pre-Trained Programming Language Models.
    Conference: AAAI.
    Year: 2023.
    '''

    def __init__(self, model, tokenizer, lang, max_iter=100, top_k=60, block_size=510):
        super().__init__("CodeAttack", model, tokenizer, lang)
        self.item = {}
        self.model_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(self.device)
        self.tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
        self.config_mlm = RobertaConfig.from_pretrained("microsoft/codebert-base-mlm")
        self.max_iter = max_iter
        self.top_k = top_k
        self.block_size = block_size

    def forward(self, code=None, label=None, *args, **kwargs):
        tokens = self.tokenizer([code], return_tensors="pt", truncation=True, padding='max_length', max_length=self.block_size).to(self.device)
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
            self.generate_adv(code, label, pred_prob, logits, substitutions)
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

        word_predictions = self.model_mlm(input_ids_.to(self.device))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.top_k, -1)  # seq-len k
        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

        variable_substitue_dict = {}
        with torch.no_grad():
            orig_embeddings = self.model_mlm.roberta(input_ids_.to(self.device))[0]
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
                        new_embeddings = self.model_mlm.roberta(new_ids_.to(self.device))[0]
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
                                             self.model_mlm,
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

    def generate_adv(self, code, label, pred_prob, pred_logits, substitutions):
        current_prob = pred_prob
        identifiers, code_tokens = get_identifiers(code, self.lang)
        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)

        variable_names = list(substitutions.keys())

        if len(variable_names) == 0:  # no variable exists
            self.item["is_attack"] = False
            self.item["note"] = "This code does not include any variable."
            return self.item

        importance_score, replace_token_positions, names_positions_dict, query_count = get_importance_score_code(words,
                                                                                                            variable_names,
                                                                                                            self.model,
                                                                                                            self.tokenizer,
                                                                                                            label,
                                                                                                            pred_logits,
                                                                                                            self.device,
                                                                                                            self.block_size)
        self.item["query_time"] += query_count
        if importance_score is None:
            self.item["is_attack"] = False
            self.item["note"] = "Variables do not have substitutes."
            return self.item

        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # recompute the importance score
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                total_score += importance_score[token_pos_to_score_pos[token_pos]]

            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
        # sort according to importance_score

        final_code = copy.deepcopy(code)
        replaced_words = {}
        self.item["is_attack"] = True
        self.item["is_success"] = False

        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]
            all_substitues = substitutions[tgt_word]
            most_gap = 0.0
            candidate = None
            for index, substitute in enumerate(all_substitues):
                temp_code = get_example(final_code, tgt_word, substitute, self.lang)
                new_feature = self.tokenizer([temp_code], return_tensors="pt", truncation=True, padding='max_length', max_length=self.block_size).to(self.device)
                logits = self.model(**new_feature).logits
                self.item["query_time"] += 1
                logits = F.sigmoid(logits)
                logits = torch.Tensor.cpu(logits).detach().numpy()[0]
                temp_prob = logits[label]
                temp_label = np.argmax(logits)
                if temp_label != label:
                    candidate = substitute
                    replaced_words[tgt_word] = candidate
                    adv_code = get_example(final_code, tgt_word, candidate, self.lang)
                    self.item["is_attack"] = True
                    self.item["is_success"] = True
                    self.item["adv_label"] = int(temp_label)
                    self.item["adv_code"] = adv_code
                    self.item["replaced_words"] = replaced_words
                    return self.item
                else:
                    gap = current_prob - temp_prob
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute

            if most_gap > 0:
                current_prob = current_prob - most_gap
                final_code = get_example(final_code, tgt_word, candidate, self.lang)
                replaced_words[tgt_word] = candidate
            else:
                replaced_words[tgt_word] = tgt_word
            self.item["replaced_words"] = replaced_words
        return self.item


