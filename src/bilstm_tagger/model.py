import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

RNNS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class BiLSTMTagger(nn.Module):
    """encoder: 'rnn' | 'lstm' | 'gru' | 'cnn'. bidirectional ignore hota hai cnn mein."""

    def __init__(self, n_words, n_chars, n_tags,
                 word_dim=100, char_dim=30, char_filters=50,
                 hidden=128, layers=1, dropout=0.33,
                 use_char=True, pad_idx=0,
                 encoder="lstm", bidirectional=True):
        super().__init__()
        self.use_char, self.encoder = use_char, encoder
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=pad_idx)
        in_dim = word_dim

        if use_char:
            self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=pad_idx)
            self.char_cnn = nn.Conv1d(char_dim, char_filters, kernel_size=3, padding=1)
            in_dim += char_filters

        self.drop = nn.Dropout(dropout)

        if encoder == "cnn":
            d = 2 * hidden
            self.convs = nn.ModuleList([
                nn.Conv1d(in_dim, d, kernel_size=3, padding=1),
                nn.Conv1d(d, d, kernel_size=3, padding=1),
            ])
            out_dim = d
        else:
            self.enc = RNNS[encoder](in_dim, hidden, num_layers=layers,
                                     bidirectional=bidirectional, batch_first=True,
                                     dropout=dropout if layers > 1 else 0.0)
            out_dim = hidden * (2 if bidirectional else 1)

        self.out = nn.Linear(out_dim, n_tags)

    def forward(self, words, chars, lengths):
        x = self.word_emb(words)                                   # (B, T, W)

        if self.use_char:
            B, T, L = chars.shape
            c = self.char_emb(chars.view(B * T, L))                # (B*T, L, C)
            c = torch.relu(self.char_cnn(c.transpose(1, 2)))       # (B*T, F, L)
            c = c.max(dim=2).values                                # (B*T, F)
            x = torch.cat([x, c.view(B, T, -1)], dim=-1)

        x = self.drop(x)

        if self.encoder == "cnn":
            mask = (torch.arange(x.size(1), device=x.device)[None, :]
                    < lengths.to(x.device)[:, None]).unsqueeze(-1)  # (B, T, 1)
            h = x * mask
            for conv in self.convs:
                h = torch.relu(conv(h.transpose(1, 2))).transpose(1, 2) * mask
            out = h
        else:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            out, _ = self.enc(packed)
            out, _ = pad_packed_sequence(out, batch_first=True,
                                         total_length=words.size(1))

        return self.out(self.drop(out))                            # (B, T, n_tags)