from src.evaluation.unsloth_adapter import UnslothLM

class Evaluator:
    def __init__(self, eval_engine=None):
        if eval_engine == "unsloth":
            self.engine = UnslothLM
        else:
            raise f"not implemented {eval_engine}"
        pass