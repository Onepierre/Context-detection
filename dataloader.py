import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import json

from collections import Counter



class SQuAD(data.Dataset):
    def __init__(self,):
        super(SQuAD, self).__init__()
        self.contexts = []
        self.questions_train = [] 
        self.questions_test = []

    def loadSquad(self, data_train_path, data_test_path):
        self.contexts = [] 
        self.questions_train = [] 
        self.questions_test = []

        with open(data_train_path,'r') as f:
            dataset = json.load(f)
        id_counter = 0
        for theme in dataset["data"]:
            for paragraph in theme["paragraphs"]:
                self.contexts.append(paragraph["context"])
                for question in paragraph["qas"]:
                    self.questions_train.append((question["question"],id_counter))
                id_counter += 1
        
        # Putting all the contexts in the same tab but using separate questions lists for now
        with open(data_test_path,'r') as f:
            dataset = json.load(f)
        for theme in dataset["data"]:
            for paragraph in theme["paragraphs"]:
                self.contexts.append(paragraph["context"])
                for question in paragraph["qas"]:
                    self.questions_test.append((question["question"],id_counter))
                id_counter += 1
        
