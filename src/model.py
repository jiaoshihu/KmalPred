import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.5):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1)  # logit
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, Lw, D]
        mask:[B, Lw] bool (True=valid)
        return: [B] logits
        """
        lengths = mask.sum(dim=1).to(torch.int64).cpu()
        lengths = torch.clamp(lengths, min=1)

        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: [num_layers*num_dir, B, H]

        if self.bidirectional:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 2H]
        else:
            feat = h_n[-1]  # [B, H]

        logit = self.head(feat).squeeze(1)  # [B]
        return logit
