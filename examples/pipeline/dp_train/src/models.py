import torch.nn as nn
from torch.autograd import Variable
import torch
from opacus.layers import DPLSTM
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from absl import logging


class FeedforwardModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pad_idx,
        H_1,
        H_2,
        D_out,
        word_embeddings,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(word_embeddings)
        self.embedding.weight.requires_grad = False
        self.l1 = nn.Linear(embedding_dim, H_1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(H_1, H_2)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(H_2, D_out)
        logging.info("Building FeedForwardModel.")

    def forward(self, text):
        out = self.embedding(text)
        out = out.mean(1)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


class NBOW(nn.Module):
    #  https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/experimental/1_nbow.ipynb
    def __init__(self, input_dim, emb_dim, output_dim, pad_idx, word_embeddings):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.embedding.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.fc = nn.Linear(emb_dim, output_dim)
        logging.info("Building NBOW.")

    def forward(self, text):
        embedded = self.embedding(text)
        pooled = embedded.mean(1)
        prediction = self.fc(pooled)
        return prediction


class LSTMClassifier(nn.Module):
    # https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/load_data.py
    # + Opacus example
    def __init__(
        self,
        batch_size,
        output_size,
        hidden_size,
        vocab_size,
        embedding_length,
        weights,
        dropout,
        dp,
    ):
        super(LSTMClassifier, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.embedding = nn.Embedding(
            vocab_size, embedding_length
        )  # Initializing the look-up table.
        self.embedding.weight = nn.Parameter(
            weights, requires_grad=False
        )  # Assigning the look-up table to the pre-trained GloVe word embedding.
        if dp:
            logging.info("Building DPLSTM.")
            self.lstm = DPLSTM(embedding_length, hidden_size, batch_first=True)
        else:
            logging.info("Building non-DP LSTM.")
            self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input, hidden):
        input_emb = self.embedding(input)
        lstm_out, _ = self.lstm(input_emb, hidden)
        if not self.dropout is None:
            lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output[:, -1]

    def init_hidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.hidden_size),
            torch.zeros(1, batch_size, self.hidden_size),
        )


class FineTunedBert(BertForSequenceClassification):
    @staticmethod
    def build_new(output_dim, model_name="google/bert_uncased_L-4_H-256_A-4"):
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            config=BertConfig.from_pretrained(
                model_name,
                num_labels=output_dim,
                output_attentions=False,
                output_hidden_states=False,
            ),
        )

        # Freeze the embeddings and the all the transformer encoder layers except the last
        trainable_layers = [
            model.bert.encoder.layer[-1],
            model.bert.pooler,
            model.classifier,
        ]
        total_params = 0
        trainable_params = 0

        for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()

        for layer in trainable_layers:
            for p in layer.parameters():
                p.requires_grad = True
                trainable_params += p.numel()

        print(f"Total parameters count: {total_params}")
        print(f"Trainable parameters count: {trainable_params}")

        return model


class BERTGRUClassification(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()["hidden_size"]

        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0 if n_layers < 2 else dropout,
        )

        self.out = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            )
        else:
            hidden = self.dropout(hidden[-1, :, :])
        output = self.out(hidden)
        return output
