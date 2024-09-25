import torch
import copy
import torch.nn.functional as F
import numpy as np
from ..attack import Attack
from .run_parser import (get_identifiers, remove_comments_and_docstrings,
                         get_example, )
from .utils import (_tokenize, get_identifier_posistions_from_code,
                    is_valid_variable_name, get_substitues, is_valid_substitue,
                    isUID)
from transformers import (RobertaForMaskedLM, RobertaTokenizer)
import random
import pdb


class MHM(Attack):
    '''
    MHM attack.
    Paper: Generating adversarial examples for holding robustness of source code processing models.
    Conference: AAAI.
    Year: 2020.
    '''

    def __init__(self, model, tokenizer, lang, max_iter=100, top_k=60, _n_candi=30, _prob_threshold=1, block_size=510):
        super().__init__("MHM Origin attack", model, tokenizer, lang)
        self.item = {}
        self.max_iter = max_iter
        self.top_k = top_k
        self._n_candi = _n_candi
        self._prob_threshold = _prob_threshold
        self.model_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(self.device)
        self.tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
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
            self.generate_adv(code, label, pred_prob, substitutions)
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

    def generate_adv(self, code, label, pre_prob, substitutions):
        identifiers, code_tokens = get_identifiers(code, self.lang)
        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer)
        raw_tokens = copy.deepcopy(words)
        variable_names = list(substitutions.keys())

        uid = get_identifier_posistions_from_code(words, variable_names)
        if len(uid) <= 0:
            self.item["is_attack"] = False
            self.item["note"] = "This code does not include any variable."
            return self.item

        variable_substitue_dict = {}

        for tgt_word in uid.keys():
            variable_substitue_dict[tgt_word] = substitutions[tgt_word]

        for iteration in range(1, 1 + self.max_iter):
            res = self.__replaceUID(_tokens=code, _label=label, pre_prob=pre_prob, _uid=uid,
                                    substitute_dict=variable_substitue_dict,
                                    _n_candi=self._n_candi,
                                    _prob_threshold=self._prob_threshold)
            if res['status'].lower() == 's':
                self.item["is_attack"] = True
                self.item["is_success"] = True
                self.item["adv_label"] = int(res["new_pred"])
                self.item["adv_code"] = code
                return self.item
        self.item["is_attack"] = True
        return self.item

    def __replaceUID(self, _tokens, _label=None, pre_prob=None, _uid={}, substitute_dict={},
                     _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):

        assert _candi_mode.lower() in ["random", "nearby"]

        selected_uid = random.sample(substitute_dict.keys(), 1)[0]  # 选择需要被替换的变量名
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(substitute_dict[selected_uid],
                                   min(_n_candi, len(substitute_dict[selected_uid]))):  # 选出_n_candi数量的候选.
                if c in _uid.keys():
                    continue
                if isUID(c):  # 判断是否是变量名.
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    candi_tokens[-1] = get_example(candi_tokens[-1], selected_uid, c, self.lang)

            candi_idx = 0
            min_prob = 1.0
            for idx, temp_code in enumerate(candi_tokens):
                new_feature = self.tokenizer([temp_code], return_tensors="pt", truncation=True,
                                             padding='max_length, max_length=self.block_size').to(self.device)
                logits = self.model(**new_feature).logits
                self.item["query_time"] += 1
                logits = F.sigmoid(logits)
                logits = torch.Tensor.cpu(logits).detach().numpy()[0]
                temp_prob = logits[_label]
                temp_label = np.argmax(logits)
                if temp_label != _label:

                    return {"status": "s", "alpha": 1, "tokens": candi_tokens,
                            "old_uid": selected_uid, "new_uid": temp_code,
                            "old_prob": pre_prob, "new_prob": temp_prob,
                            "old_pred": _label, "new_pred": temp_label, "nb_changed_pos": _tokens.count(selected_uid)}
                if temp_prob < min_prob:
                    candi_idx = idx
                    min_prob = temp_prob
                    min_uid = temp_code
                    min_pred = temp_label

            # At last, compute acceptance rate.
            alpha = (1 - min_prob + 1e-10) / (1 - pre_prob + 1e-10)
            # 计算这个id对应的alpha值.
            if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                return {"status": "r", "alpha": alpha, "tokens": candi_tokens,
                        "old_uid": selected_uid, "new_uid": min_uid,
                        "old_prob": pre_prob, "new_prob": min_prob,
                        "old_pred": _label, "new_pred": min_pred, "nb_changed_pos": _tokens.count(selected_uid)}
            else:
                return {"status": "a", "alpha": alpha, "tokens": candi_tokens,
                        "old_uid": selected_uid, "new_uid": min_uid,
                        "old_prob": pre_prob, "new_prob": min_prob,
                        "old_pred": _label, "new_pred": min_pred, "nb_changed_pos": _tokens.count(selected_uid)}
        else:
            pass
