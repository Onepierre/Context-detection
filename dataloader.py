import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import json

from collections import Counter



class SQuAD():
    def __init__(self):
        self.contexts = []
        self.questions = [] 
        


    def loadSquad(self, data_path):
        self.contexts = [] 
        self.questions = [] 

        with open(data_path,'r') as f:
            dataset = json.load(f)

        id_counter = 0
        for theme in dataset["data"]:
            for paragraph in theme["paragraphs"]:
                self.contexts.append(paragraph["context"])
                for question in paragraph["qas"]:
                    self.questions.append((question["question"],id_counter))
                id_counter += 1

        
