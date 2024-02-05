import torch


class Attacker(object):
    r"""
        Base class for all attacks
        """
    def __init__(self, attacker_name: str, model: torch.nn.Module, tokenizer):
        r"""
        Initialize the attack
        Args:
            attack_name: name of the adversarial attack
            model: model to attack
        """
        self.attacker = attacker_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
