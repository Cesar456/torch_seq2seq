import torch
import re
from torch.utils.data import Dataset
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    def __init__(self, name, vocab_path):
        with open(vocab_path, encoding='utf8') as f:
            self.vocab = [line.strip() for line in tqdm(f.readlines(), desc="read vocab")]
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        self.init_word()

    def init_word(self):
        for word in self.vocab:
            self.word2index[word] = self.n_words
            self.word2count[word] = 0
            self.index2word[self.n_words] = word
            self.n_words += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        self.word2count[word] += 1

    def convert_index2sentence(self, indexes):
        return "".join([self.index2word[index] for index in indexes])


class SentenceDataSet(Dataset):
    def __init__(self, dir_path):
        self.input_lang, self.output_lang, self.sentence_pairs = get_pair_data(dir_path, True)
        self.tensors = [tensors_from_pair(self.input_lang, self.output_lang, pair) for pair in self.sentence_pairs]
        self.source = torch.nn.utils.rnn.pad_sequence([x[0] for x in self.tensors], True)
        self.target = torch.nn.utils.rnn.pad_sequence([x[1] for x in self.tensors], True)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return self.source[item], self.target[item]

    def tokenize(self, sentence):
        tensor = tensor_from_sentence(self.output_lang, sentence)
        return tensor

    def convert_index2sentence(self, indexes):
        return "".join([self.output_lang.convert_index2sentence(indexes)])

    def convert_one_sentence2tensor(self, sentence):
        return tensor_from_sentence(self.output_lang, sentence).unsqueeze(0).to(device)


def tensor_from_sentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(1)
    return torch.tensor(indexes, dtype=torch.long)


def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, target_tensor


def normalize_string(s):
    s = re.sub(r"([.!?])", r" \1", s)
    return s


def get_pair_data(dir_path, is_train=True):
    in_file_name = "in.txt"
    out_file_name = "out.txt"
    vocab_path = os.path.join(dir_path, 'vocabs')
    in_lines = open(f"{dir_path}/{'train' if is_train else 'test'}/{in_file_name}", encoding='utf-8').readlines()
    out_lines = open(f"{dir_path}/{'train' if is_train else 'test'}/{out_file_name}", encoding='utf-8').readlines()
    in_lines = [normalize_string(line.strip()) for line in tqdm(in_lines, desc='read in data')]
    out_lines = [normalize_string(line.strip()) for line in tqdm(out_lines, desc='read out data')]

    pairs = list(zip(in_lines, out_lines))

    input_lang = Lang("in", vocab_path)
    output_lang = Lang("out", vocab_path)
    for pair in tqdm(pairs, desc="load data 2 pair"):
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    return input_lang, output_lang, pairs
