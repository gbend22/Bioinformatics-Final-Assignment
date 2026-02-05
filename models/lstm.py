import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, num_genes, seq_len, input_size,
                 hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.view(-1, self.seq_len, self.input_size)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.fc(x)
