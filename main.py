import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import sys
import numpy as np
import json
import nltk
from nltk.corpus import stopwords

from dataloader import SQuAD
from evaluate import scores

from models.naivefilter import basicFilter

if __name__ == "__main__":

    test = SQuAD()
    test.loadSquad("Squad/train-v1.1.json","Squad/dev-v1.1.json")
    model1 = basicFilter(test.contexts)
    model1.loadQuestions(test.questions_train)
    print("Loaded")
    model1.computeModel()