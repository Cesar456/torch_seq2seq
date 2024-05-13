import os

import sys
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from transformer import model, data_util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # 初始化数值
    batch_size = 128
    epoch_num = 30
    em_size = 200  # embedding dimension
    n_hid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    n_layers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    n_head = 2  # the number of heads in the multi_head_attention models
    dropout = 0.2  # the dropout value

    dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/couplet")

    logger.info("start init data")
    dataset = data_util.SentenceDataSet(dir_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_tokens = dataset.input_lang.n_words  # the size of vocabulary
    logger.info("end init data")

    trans_model = model.TransformerModel(n_tokens, em_size, n_head, n_hid, n_layers, dropout)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trans_model.parameters())

    test_sentence = " ".join("这是一个测试")

    for epoch in tqdm(range(epoch_num), desc="epoch"):
        model.do_train(trans_model, data_loader, optimizer, criterion, n_tokens, dataset, test_sentence=test_sentence)
