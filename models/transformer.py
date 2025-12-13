import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Net_Transformer(nn.Module):
    def __init__(
        self,
        input_size,
        target_length,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        device,
        max_len=500,
    ):
        super().__init__()
        self.device = device
        self.target_length = target_length
        self.d_model = d_model

        self.input_linear = nn.Linear(input_size, d_model)
        self.output_linear = nn.Linear(d_model, input_size)

        self.pos_encoder = PositionalEncoding(d_model, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(
            torch.ones(sz, sz, device=self.device), diagonal=1
        ).bool()

    def forward(self, src):
        B, T_in, _ = src.size()

        # ---- Encoder ----
        src_emb = self.input_linear(src) * (self.d_model ** 0.5)
        src_emb = self.pos_encoder(src_emb)
        memory = self.transformer.encoder(src_emb)

        # ---- Decoder ----
        outputs = []

        # start token = last observed value
        tgt_raw = src_emb[:, -1:, :]   # raw decoder embeddings

        for t in range(self.target_length):
            tgt_with_pos = self.pos_encoder(tgt_raw)
            tgt_mask = self.generate_square_subsequent_mask(tgt_with_pos.size(1))

            dec = self.transformer.decoder(
                tgt_with_pos, memory, tgt_mask=tgt_mask
            )

            pred = self.output_linear(dec[:, -1:, :])
            outputs.append(pred)

            # embed prediction for next step
            pred_emb = self.input_linear(pred)
            tgt_raw = torch.cat([tgt_raw, pred_emb], dim=1)

        return torch.cat(outputs, dim=1)
