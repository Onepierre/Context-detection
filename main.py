import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import json

from collections import Counter
from dataloader import SQuAD

if __name__ == "__main__":

    test = SQuAD()
    test.loadSquad("Squad/train-v1.1.json","Squad/dev-v1.1.json")

    print(test.questions_train[50])
    print(test.contexts[10])
