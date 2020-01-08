import torchtext
import os
from torchtext import data
from torchtext.data.utils import get_tokenizer


dir_path = os.path.dirname(__file__)
print(dir_path)

# def tokenizer(text):
#     return [tok.text for tok in spacy_en.tokenizer(text)]

# TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)

TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"), init_token='<sos>', eos_token='<eos>', lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
print(len(test_txt.examples[0].text))
TEXT.build_vocab(test_txt)
print(len(TEXT.vocab))
print(TEXT.vocab.itos[0])

print(test_txt[0].text[:10])
# 将test_txt 中的text转为onehot tensor
data1 = TEXT.numericalize([test_txt.examples[0].text])
print(data1[:10])
print(data1)
print([TEXT.vocab.itos[i] for i in data1[:10]])
print(data1.size(0))
n_batch = data1.size(0) // 128
print(n_batch)
