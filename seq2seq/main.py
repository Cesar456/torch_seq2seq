import sys
import os

# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)

import logging
from ..seq2seq import model, data_util

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# def show_plot(points):
#     plt.switch_backend('agg')
#     plt.figure()
#     fig, ax = plt.subplots()
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)


if __name__ == '__main__':
    # 初始化数值
    hidden_size = 256
    batch_size = 128
    max_length = 128
    epoch_num = 3
    dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/couplet")

    logger.info("start init data")
    dataset = data_util.SentenceDataSet(dir_path)
    logger.info("end init data")

    logger.info("init model")
    encoder1 = model.EncoderRNN(dataset.input_lang.n_words, hidden_size).to(model.device)
    attn_decoder1 = model.AttnDecoderRNN(hidden_size, dataset.output_lang.n_words, dropout_p=0.1).to(model.device)

    logger.info("start train")
    losses = model.train(dataset, encoder1, attn_decoder1, epoch_num=epoch_num)
    # show_plot(losses)
