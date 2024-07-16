from .GreedyAttack import GreedyAttack
from .GeneticAlgorithm import GeneticAlgorithm
from ..attack import Attack
import numpy as np


class ALERT(Attack):
    '''
    ALERT attack.
    Paper: Natural Attack for Pre-trained Models of Code.
    Conference: ICSE.
    Year: 2022.
    '''

    def __init__(self, model, tokenizer, lang, max_iter=100, top_k=100, max_iter_mutant=10, cross_probability=0.7):
        super().__init__("ALERT", model, tokenizer, lang)
        self.max_iter = max_iter
        self.top_k = top_k
        self.max_iter_mutant = max_iter_mutant
        self.cross_probability = cross_probability

    def forward(self, code=None, label=None, *args, **kwargs):
        atk_1 = GreedyAttack(self.model, self.tokenizer, self.lang, int(np.ceil(self.max_iter/2)), self.top_k)
        atk_2 = GeneticAlgorithm(self.model, self.tokenizer, self.lang, self.max_iter-int(np.ceil(self.max_iter/2)), self.top_k, self.max_iter_mutant, self.cross_probability)
        result_1 = atk_1(code, label)
        if result_1['is_attack'] and not result_1['is_success']:
            initial_replace = result_1['replaced_words']
            result_2 = atk_2(code, label, initial_replace=initial_replace)
            result_2['query_time'] += result_1['query_time']
            return result_2
        else:
            return result_1
