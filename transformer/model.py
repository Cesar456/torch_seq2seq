import math
import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch import nn
import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):
    def __init__(self, n_token, n_inp, n_head, n_hid, n_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(n_inp, dropout)
        encoder_layers = TransformerEncoderLayer(n_inp, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(n_token, n_inp)
        self.n_inp = n_inp
        self.decoder = nn.Linear(n_inp, n_token)

        self.init_weights()

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.n_inp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def do_train(model, data_loader, optimizer, criterion, n_tokens, data_set, test_sentence="如 此 清 风 如 此 夜"):
    model = model.to(device)
    model.train()
    for i, (xt, yt) in enumerate(data_loader):
        optimizer.zero_grad()
        xt = xt.to(device)
        yt = yt.to(device)
        output = model(xt)
        loss = criterion(output.view(-1, n_tokens), yt.view(-1))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if i % 100 == 0:
            logger.info(f"process:{i}/{len(data_loader)}, loss:{loss.item()}")
            do_test(model, test_sentence, data_set, n_tokens)


def do_test(model, sentence, dataset, n_tokens):
    model.eval()
    test_tensor = dataset.convert_one_sentence2tensor(sentence)
    result = model(test_tensor).view(-1, n_tokens).argmax(dim=1).cpu().detach().numpy()[:-1]
    logger.info(f"原文:{sentence}, 生成:{dataset.convert_index2sentence(result)}")
