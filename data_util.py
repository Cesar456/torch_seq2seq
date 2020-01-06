import re
import os

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
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
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    return input_lang, output_lang, pairs


if __name__ == '__main__':
    print(len(get_pair_data("./data/couplet", is_train=False)))
