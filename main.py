import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import argparse
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


# Create the parser
parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
parser.add_argument('--model', metavar='model', type=str, help='the model used', choices=['bert', 'filter'])
parser.add_argument('--file', metavar='file', type=str, help='the model used', choices=['train', 'dev'])

# Execute the parse_args() method
args = parser.parse_args()


# Load the datasets
test = SQuAD()
if args.file == "train":
    test.loadSquad("Squad/train-v1.1.json")
elif args.file == "dev":
    test.loadSquad("Squad/dev-v1.1.json")


if args.model == "filter":
# Basic filter
    model1 = BasicFilter(test.contexts)
    model1.loadQuestions(test.questions[:1000])
    print("Model loaded")
    model1.computeModel()

elif args.model == "bert":
    # doc2vec not working
    # model2 = doc2VecFilter()
    
    # BERT encodings comparison
    model2 = BertFilter()

    model2.encodeContexts(test.contexts[:2000])
    print("Model loaded")
    model2.encodeQuestions(test.questions[:500])
    print("Questions encoded")
    model2.computeModel()