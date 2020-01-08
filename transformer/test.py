import torchtext
import os
from torchtext import data


dir_path = os.path.dirname(__file__)
print(dir_path)

data_path = os.path.join(dir_path, "../data/couplet")



TEXT = data.Field(sequential=True, tokenize=lambda  x: x.split(" "), lower=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False)
