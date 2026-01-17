
import torch 
import torch.nn as nn 
from src.data.processed import get_vocab_sizes

class Transformers(nn.Module):
  def __init__(self,embbed_dim,hidden_dim,num_head,num_layer,vocab_size,y1_size,y2_size,max_len):
    super().__init__()
    self.embbeds = nn.Embedding(vocab_size,embbed_dim,padding_idx=0)
    self.posembd = nn.Embedding(max_len,embbed_dim)
    encoder_layer = nn.TransformerEncoderLayer(d_model=embbed_dim,nhead=num_head,
                                      dim_feedforward=hidden_dim,dropout=0.2,batch_first=True)
    self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layer)
    self.fc1 = nn.Linear(embbed_dim,y1_size)
    self.fc2 = nn.Linear(embbed_dim,y2_size)

  def forward(self,x):

    batch_size,seq_len = x.size()
    positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
    padding_mask = (x == 0)

    embbeds = self.embbeds(x)
    pos_embed = self.posembd(positions)
    embbeded = embbeds+pos_embed
    encode = self.encoder(embbeded,src_key_padding_mask=padding_mask)

    y1 = self.fc1(encode)
    y2 = self.fc2(encode)

    return y1,y2

vocab_size,y1_size,y2_size = get_vocab_sizes()

def get_transformer():
  transformer = Transformers(160,180,4,2,vocab_size,y1_size,y2_size,1632)
  return transformer

