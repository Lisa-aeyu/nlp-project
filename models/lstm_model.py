import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union
import json
import sys

sys.path.append('..')
from models.lamberts_funcs import data_preprocessing, l_predict_class
from models.lamberts_funcs import my_padding, train3, logs_dict_multi_metrics
SEQ_LEN = 111
with open("models/vocab_to_int.json", "r", encoding="utf-8") as json_file:
    vocab_to_int = json.load(json_file)

@dataclass
class ConfigRNN:
    vocab_size: int
    device: str
    n_layers: int
    embedding_dim: int
    hidden_size: int
    seq_len: int
    bidirectional: Union[bool, int]


net_config = ConfigRNN(
    vocab_size=len(vocab_to_int) + 1,
    device='cuda',
    n_layers=4,
    embedding_dim=32,
    hidden_size=32,
    seq_len=SEQ_LEN,
    bidirectional=False,
)

class LSTMClassifier(nn.Module):
    def __init__(self, rnn_conf=net_config) -> None:
        super().__init__()

        self.embedding_dim = rnn_conf.embedding_dim
        self.hidden_size = rnn_conf.hidden_size
        self.bidirectional = rnn_conf.bidirectional
        self.n_layers = rnn_conf.n_layers

        self.embedding = nn.Embedding(rnn_conf.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.n_layers,
        )
        self.bidirect_factor = 2 if self.bidirectional else 1
        self.attention = nn.Linear(self.hidden_size * self.bidirect_factor, 1)
        self.clf = nn.Sequential(
            nn.Linear(self.hidden_size * self.bidirect_factor, 32),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(32, 3)  # Три класса
        )

    def model_description(self):
        direction = "bidirect" if self.bidirectional else "onedirect"
        return f"lstm_{direction}_{self.n_layers}"

    def forward(self, x: torch.Tensor):
        embeddings = self.embedding(x)
        out, _ = self.lstm(embeddings)
        attn_weights = torch.softmax(self.attention(out), dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(out * attn_weights, dim=1)  # (batch_size, hidden_size * num_directions)
        out = self.clf(context)
        return out