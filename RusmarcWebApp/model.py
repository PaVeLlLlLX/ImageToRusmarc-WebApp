import torch
import torch.nn as nn
import gc
import math
import torch.backends.cudnn as cudnn
from torchvision.models import resnet34
from torch.nn import TransformerDecoder, TransformerDecoderLayer, LayerNorm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

cudnn.benchmark = True
cudnn.deterministic = False

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

def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)


class ImageToRusmarcModel(nn.Module):
    def __init__(self, num_tokens, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, sos_token_id=1, eos_token_id=2, pad_token_id=0):
        super().__init__()
        self.d_model = d_model
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.num_tokens = num_tokens

        # Энкодер
        cnn_backbone = resnet34(pretrained=True)
        self.cnn_layers = nn.Sequential(*list(cnn_backbone.children())[:-2]) # [B, 512, H', W']
        
        self.cnn_channel_proj = nn.Conv2d(cnn_backbone.fc.in_features, d_model, kernel_size=1)
        self.encoder_pos_encoder = PositionalEncoding(d_model, dropout)

        # Декодер
        self.token_embedding = nn.Embedding(num_tokens, d_model)
        self.decoder_pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.output_layer = nn.Linear(d_model, num_tokens)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                if 'cnn_layers' not in name:
                    print(f"Initializing weights for: {name}")
                    if module.weight.dim() > 1:
                        nn.init.xavier_uniform_(module.weight)
                    if hasattr(module, 'bias') and module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if 'cnn_layers' not in name:
                    print(f"Initializing LayerNorm weights for: {name}")
                    nn.init.constant_(module.bias, 0)
                    nn.init.constant_(module.weight, 1.0)

    def encode(self, src_img):
        # src_img: [B, C, H, W]
        features = self.cnn_layers(src_img) # [B, C_cnn, H', W']
        features = self.cnn_channel_proj(features) # [B, d_model, H', W']

        b, d, h_prime, w_prime = features.shape
        memory = features.flatten(2) # [B, d_model, H'*W']
        memory = memory.permute(2, 0, 1) # [H'*W', B, d_model]

        memory = self.encoder_pos_encoder(memory)
        return memory # [SeqLen_enc, B, d_model]

    def decode(self, tgt_seq, memory, tgt_mask, tgt_padding_mask):
        # tgt_seq: [SeqLen_dec, B] (batch_first=False)
        # memory: [SeqLen_enc, B, d_model]
        # tgt_mask: [SeqLen_dec, SeqLen_dec]
        # tgt_padding_mask: [B, SeqLen_dec]

        tgt_emb = self.token_embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = self.decoder_pos_encoder(tgt_emb) # [SeqLen_dec, B, d_model]

        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=None
        ) # [SeqLen_dec, B, d_model]

        logits = self.output_layer(output) # [SeqLen_dec, B, num_tokens]
        return logits

    def forward(self, src_img, tgt_seq):
        # src_img: [B, C, H, W]
        # tgt_seq: [B, T]

        tgt_in = tgt_seq[:, :-1]
        tgt_in = tgt_in.permute(1, 0) # [T-1, B]

        device = src_img.device
        tgt_seq_len = tgt_in.size(0)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device) # [T-1, T-1]
        tgt_padding_mask = (tgt_in.T == self.pad_token_id) # [B, T-1]

        memory = self.encode(src_img) # [SeqLen_enc, B, d_model]

        logits = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask) # [T-1, B, num_tokens]

        return logits #[SeqLen, Batch, Classes]

    def generate(self, image, max_len=1000):
        batch_size = image.size(0)
        device = image.device

        memory = self.encode(image)  # [SeqLen_enc, B, d_model]
        
        tgt_tokens = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)
        
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            if not active_mask.any():
                break 
            
            active_tokens = tgt_tokens[active_mask]  # [active_B, current_len]
            
            tgt_in_step = active_tokens.permute(1, 0)  # [current_len, active_B]
            tgt_mask_step = generate_square_subsequent_mask(tgt_in_step.size(0), device)
            tgt_padding_mask_step = (active_tokens == self.pad_token_id)  # [active_B, current_len]
            
            output_step = self.decode(
                tgt_in_step, 
                memory[:, active_mask, :],
                tgt_mask_step, 
                tgt_padding_mask_step
            )  # [current_len, active_B, num_tokens]
            
            logits_last_token = output_step[-1, :, :]  # [active_B, num_tokens]
            next_token = logits_last_token.argmax(dim=-1, keepdim=True)  # [active_B, 1]
            
            tgt_tokens = torch.cat([
                tgt_tokens, 
                torch.full((batch_size, 1), self.pad_token_id, dtype=torch.long, device=device)
            ], dim=1)
            
            tgt_tokens[active_mask, -1:] = next_token
            
            active_mask[active_mask.clone()] = (next_token.squeeze(1) != self.eos_token_id)
        
        return tgt_tokens  # [B, final_len]
