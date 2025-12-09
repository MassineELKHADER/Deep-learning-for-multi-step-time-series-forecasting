import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size):
        super(EncoderRNN, self).__init__()  
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)

    def forward(self, input, hidden): # input [batch_size, length T, dimensionality d]      
        output, hidden = self.gru(input, hidden)      
        return output, hidden
    
    def init_hidden(self,device):
        #[num_layers*num_directions,batch,hidden_size]   
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)
    
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers,fc_units, output_size):
        super(DecoderRNN, self).__init__()      
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, fc_units)
        self.out = nn.Linear(fc_units, output_size)         
        
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden) 
        output = F.relu( self.fc(output) )
        output = self.out(output)      
        return output, hidden
    
class Net_GRU(nn.Module):
    def __init__(self, encoder, decoder, target_length, device):
        super(Net_GRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.device = device
        
    def forward(self, x):
        input_length  = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:,ei:ei+1,:]  , encoder_hidden)
            
        decoder_input = x[:,-1,:].unsqueeze(1) # first decoder input= last element of input sequence
        decoder_hidden = encoder_hidden
        
        outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]  ).to(self.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output
            outputs[:,di:di+1,:] = decoder_output
        return outputs      

class Net_Transformer(nn.Module):
    def __init__(self, input_size, target_length, d_model, nhead,
                 num_encoder_layers, num_decoder_layers, dim_feedforward, device):
        super().__init__()
        self.device = device
        self.target_length = target_length
        self.d_model = d_model

        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.output_linear = nn.Linear(d_model, input_size)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(self.device)

    def forward(self, src):
        batch, src_len, _ = src.size()

        # ---- Encode input sequence ----
        src = self.input_linear(src) * (self.d_model ** 0.5)
        src = src + self.pos_encoder[:, :src_len, :]
        memory = self.transformer.encoder(src)

        # ---- Autoregressive decode ----
        outputs = []
        tgt = src[:, -1:, :]  # start token = last observed value

        for _ in range(self.target_length):
            tgt = tgt + self.pos_encoder[:, :tgt.size(1), :]
            mask = self.generate_square_subsequent_mask(tgt.size(1))

            dec = self.transformer.decoder(tgt, memory, tgt_mask=mask)
            pred = self.output_linear(dec[:, -1:, :])
            outputs.append(pred)

            tgt = torch.cat([tgt, self.input_linear(pred)], dim=1)

        return torch.cat(outputs, dim=1)  # [batch, target_length, input_size]
