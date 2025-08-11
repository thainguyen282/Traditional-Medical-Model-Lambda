from abc import ABC, abstractmethod
class IlmEvaluator(ABC):
    @abstractmethod
    def load(self, model_path,**kwargs):
        pass
    @abstractmethod
    def save(self,results, filename="eval_results.json"):
        pass
    @abstractmethod
    def run_evaluate(self,tasks:list,num_fewshot=5,limit=None,random_seed=42,save_ouput_path=None,print_output=True):
        pass