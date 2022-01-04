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

from models.naivefilter import BasicFilter
from models.bertfilter import BertFilter
from models.doc2vec import doc2VecFilter

if __name__ == "__main__":

    test = SQuAD()
    test.loadSquad("Squad/train-v1.1.json","Squad/dev-v1.1.json")

    # Basic filter
    model1 = BasicFilter(test.contexts)
    model1.loadQuestions(test.questions_train[:1000])
    print("Loaded")
    model1.computeModel()
    #model1.getConfidence()

    # BERT encodings comparison
    # model2 = BertFilter()
    # model2 = doc2VecFilter()
    # model2.encodeContexts(test.contexts[:100])
    # print("Model loaded")
    # model2.encodeQuestions(test.questions_train)
    # print("Questions encoded")
    # model2.computeModel()