import torch 
import torch.nn as nn 
from transformers import AutoTokenizer, AutoModel
from src.data.processed import get_vocab_sizes
model_name =  "Rostlab/prot_bert"

encoder = AutoModel.from_pretrained(model_name)

amino_size,y1_size,y2_size = get_vocab_sizes()

class classifierhead(nn.Module):
  def __init__(self,encoder,hidden_dim,y1_size,y2_size):
    super().__init__()
    self.encoder = encoder
    self.drop = nn.Dropout(0.1)
    self.fc1 = nn.Linear(hidden_dim,y1_size)
    self.fc2 = nn.Linear(hidden_dim,y2_size)

  def forward(self,x,attention_mask):
    out = self.encoder(x,attention_mask=attention_mask)
    out = out.last_hidden_state
    out = self.drop(out)
    y1_logits = self.fc1(out)
    y2_logits = self.fc2(out)
    return y1_logits ,y2_logits



model = classifierhead(encoder,1024,y1_size,y2_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_pretrained_train():
  for param in model.encoder.parameters():
    param.requires_grad = False
  return model 

def get_pretrained_inference():
  return model 
  