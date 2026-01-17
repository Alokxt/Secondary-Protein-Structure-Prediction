import torch 
import torch.nn as nn 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.data.processed import get_vocab_sizes

class BiRNN(nn.Module):
  def __init__(self,vocab_size, embed_dim, hidden_dim,y1_size, y2_size, padding_idx=0):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=padding_idx)
    self.conv = nn.Conv1d(in_channels=embed_dim,out_channels=128,kernel_size=3,padding=1)
    self.act1 = nn.ReLU()
    self.birnn = nn.RNN(input_size=128,hidden_size=hidden_dim,num_layers=2,bidirectional=True
                        ,batch_first=True)
    self.proj = nn.Linear(2*hidden_dim,hidden_dim)
    self.act2 = nn.ReLU()
    self.drop = nn.Dropout(p=0.3)
    self.fc1 = nn.Linear(in_features=hidden_dim,out_features=y1_size)
    self.fc2 = nn.Linear(in_features=hidden_dim,out_features=y2_size)
  def forward(self,x,lengths):
    embedds = self.embedding(x)
    embedds = embedds.transpose(1,2)
    convd  = self.act1(self.conv(embedds))
    convd = convd.transpose(1,2)
    packed = pack_padded_sequence(
            convd,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
    bidir,_ = self.birnn(packed)
    out,_ = pad_packed_sequence(
            bidir, batch_first=True
        )
    proj = self.act2(self.proj(out))
    dropped = self.drop(proj)
    y1 = self.fc1(dropped)
    y2 = self.fc2(dropped)
    return y1,y2


class BiLSTM(nn.Module):
    def __init__(self,embbed,hidden_dim,vocab_size,y1_size,y2_size):
        super().__init__()
        self.embbeding = nn.Embedding(vocab_size,embbed, padding_idx=0)
        self.lstm = nn.LSTM(embbed,hidden_dim,batch_first = True,num_layers=2,
                            bidirectional=True, dropout=0.2)
        self.layernorm = nn.LayerNorm(hidden_dim*2)
        self.drop =  nn.Dropout(0.3)
        self.fc_hidden = nn.Linear(hidden_dim*2, hidden_dim)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, y1_size)
        self.fc2 = nn.Linear(hidden_dim, y2_size)
    def forward(self,x,lengths):
        embeds = self.embbeding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embeds,lengths.cpu(),batch_first=True,
                                                   enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.layernorm(out)
        out = self.drop(out)
        hidd_fc = self.fc_hidden(out)
        logits = self.act(hidd_fc)
        y1 = self.fc1(logits)
        y2 = self.fc2(logits)
        return y1,y2


class BiGRU(nn.Module):
    def __init__(self, embbed, hidden_dim, vocab_size, y1_size, y2_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embbed, padding_idx=0
        )

        self.gru = nn.GRU(
            embbed,
            hidden_dim,
            batch_first=True,
            num_layers=2,
            bidirectional=True,
            dropout=0.2
        )

        self.layernorm = nn.LayerNorm(hidden_dim * 2)
        self.drop = nn.Dropout(0.3)

        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.ReLU()

        self.fc1 = nn.Linear(hidden_dim, y1_size)
        self.fc2 = nn.Linear(hidden_dim, y2_size)

    def forward(self, x, lengths):


        embeds = self.embedding(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeds,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.gru(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True
        )

        out = self.layernorm(out)
        out = self.drop(out)

        hidden_fc = self.fc_hidden(out)
        logits = self.act(hidden_fc)

        y1 = self.fc1(logits)
        y2 = self.fc2(logits)

        return y1, y2

vocab_size,y1_size,y2_size = get_vocab_sizes()

def get_birnn():
   bi_rnn =BiRNN(vocab_size,100,128,y1_size, y2_size, padding_idx=0)
   return bi_rnn

def get_bilstm():
   bi_lstm = BiLSTM(84,128,vocab_size,y1_size,y2_size)
   return bi_lstm

def get_birgu():
   bi_gru = BiGRU(64, 256, vocab_size, y1_size, y2_size)
   return bi_gru

