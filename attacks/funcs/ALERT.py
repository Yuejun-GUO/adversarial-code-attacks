from .greedy import Greedy
from .GeneticAlgorithm import GeneticAlgorithm
from ..attack import Attack


class ALERT(Attack):
    '''
    ALERT attack.
    '''

    def __init__(self, model, tokenizer, lang, max_iter):
        super().__init__("ALERT", model, tokenizer, lang)
        self.max_iter = max_iter

    def forward(self, code=None, label=None, *args, **kwargs):
        atk_1 = Greedy(self.model, self.tokenizer, self.lang)
        atk_2 = GeneticAlgorithm(self.model, self.tokenizer, self.lang, self.max_iter)
        result_1 = atk_1(code, label)
        if result_1['is_attack'] and not result_1['is_success']:
            initial_replace = result_1['replaced_words']
            result_2 = atk_2(code, label, initial_replace=initial_replace)
            result_2['query_time'] += result_1['query_time']
            return result_2
        else:
            return result_1
