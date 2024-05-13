import re
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class SentenceDataSet(Dataset):
    def __init__(self, dir_path):
        self.input_lang, self.output_lang, self.sentence_pairs = get_pair_data(dir_path, True)
        self.tensors = [tensors_from_pair(self.input_lang, self.output_lang, pair) for pair in self.sentence_pairs]

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return self.tensors[item]


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    zip_list = list(zip(*data))
    data0 = list(zip_list[0])
    data1 = list(zip_list[1])
    data_length = [x.size()[0] for x in data0]
    data0 = torch.nn.utils.rnn.pad_sequence(data0, batch_first=True)
    data1 = torch.nn.utils.rnn.pad_sequence(data1, batch_first=True)
    # return data0, data1, data_length
    return data0.unsqueeze(-1), data1.unsqueeze(-1), data_length


def tensor_from_sentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
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
    in_lines = open(f"{dir_path}/{'train' if is_train else 'test'}/{in_file_name}", encoding='utf-8').readlines()
    out_lines = open(f"{dir_path}/{'train' if is_train else 'test'}/{out_file_name}", encoding='utf-8').readlines()
    in_lines = [normalize_string(line.strip()) for line in in_lines]
    out_lines = [normalize_string(line.strip()) for line in out_lines]

    pairs = list(zip(in_lines, out_lines))

    input_lang = Lang("in")
    output_lang = Lang("out")
    for pair in tqdm(pairs):
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    return input_lang, output_lang, pairs


if __name__ == '__main__':
    print(len(get_pair_data("../data/couplet", is_train=False)))
