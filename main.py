import logging
import model
import data_util
import torch
import random
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from utils import show_plot

torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dir_path = "./data/couplet"
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=128):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=model.device)
    loss = torch.tensor(0.0).to(model.device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=model.device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            top_v, top_i = decoder_output.topk(1)
            decoder_input = top_i.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(random.choice(sentence_pairs)) for _ in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter_ in tqdm(range(1, n_iters + 1)):
        training_pair = training_pairs[iter_ - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter_ % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            logger.info("iter:{}, print_loss_avg:{}".format(iter_, print_loss_avg))

        if iter_ % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)


if __name__ == '__main__':
    # 初始化数值
    hidden_size = 256
    batch_size = 128
    max_length = 128

    logger.info("start init data")
    dataset = data_util.SentenceDataSet(dir_path)
    for i, x in enumerate(dataset):
        print(x)
        if i > 10:
            break
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info("end init data")
    for xt in data_loader:
        print(xt)
        break
    encoder1 = model.EncoderRNN(dataset.input_lang.n_words, hidden_size).to(model.device)
    attn_decoder1 = model.AttnDecoderRNN(hidden_size, dataset.output_lang.n_words, dropout_p=0.1).to(model.device)
    logger.info("start train")

    # train_iters(encoder1, attn_decoder1, 75000, print_every=50)
    # for i, pair in enumerate(sentence_pairs):
    #     input_tensor = tensor_from_sentence(input_lang, pair[0])
    #     target_tensor = tensor_from_sentence(output_lang, pair[1])
