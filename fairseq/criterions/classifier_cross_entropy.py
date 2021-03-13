import torch.nn as nn
from fairseq.criterions import FairseqCriterion, register_criterion

@register_criterion("classifier_cross_entropy")
class ClassifierCrossEntropy(FairseqCriterion):
    def __init__(self,
                 task, ):
        pass