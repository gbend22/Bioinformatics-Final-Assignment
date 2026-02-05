import torch
import torch.nn as nn
import numpy as np

class TransformerClassifier(nn.Module):
    def __init__(
        self, 
        num_genes=19800, 
        num_classes=32,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        chunk_size=512
    ):
        super(TransformerClassifier, self).__init__()
        
        self.num_genes = num_genes
        self.d_model = d_model
        self.chunk_size = chunk_size
        
        self.input_projection = nn.Sequential(
            nn.Linear(num_genes, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.seq_len = 1024 // d_model
      
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * self.seq_len, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_projection(x)
        x = x.view(batch_size, self.seq_len, self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.reshape(batch_size, -1)
        x = self.classifier(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
