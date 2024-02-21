import torch


class Attack(object):
    r"""
        Base class for all attacks
        """
    def __init__(self, name, model: torch.nn.Module, tokenizer):
        r"""
        Initialize the attack
        Args:
            name: name of the adversarial attack
            model: model to attack
            tokenizer: the tokenizer
        """
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def forward(self, code, label=None, *args, **kwargs):
        r"""
               It defines the computation performed at every call.
               Should be overridden by all subclasses.
               """
        raise NotImplementedError

    def __call__(self, *code, **kwargs):
        self.model.eval()
        result = self.forward(*code, **kwargs)

        return result
