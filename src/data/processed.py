from torch.utils.data import Dataset
import torch.nn as nn 
import numpy as np 
import pandas as pd 
import torch 
from src.data.splitted import get_splitted_data,get_vocabs
from transformers import AutoTokenizer

#model_name =  "Rostlab/prot_bert"
#tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)



def convert_to_integer(x,d,mx):
    l  = []
    for i in range(len(x)):
        if x[i] in d:
            l.append(d[x[i]])
        else:
            l.append(1)
    padded = [0]*(mx-len(l))+l


    return np.array(padded)



class CustomData1(Dataset):
  def __init__(self,amino_acid,sst_8,sst_3,Seq,sst8=None,sst3=None):
    super().__init__()
    self.X = Seq
    self.y1 = sst8
    self.y2 = sst3
    self.amino_acid = amino_acid
    self.sst_8 = sst_8
    self.sst_3 = sst_3
  def __len__(self):
    return len(self.X)
  def __getitem__(self, index):
    x = torch.tensor(convert_to_integer(self.X[index],self.amino_acid,len(self.X[index])),dtype=torch.long)
    if self.y1 is None:
      return x
    sst8 = torch.tensor(convert_to_integer(self.y1[index],self.sst_8,len(self.y1[index])),dtype=torch.long)
    sst3 = torch.tensor(convert_to_integer(self.y2[index],self.sst_3,len(self.y2[index])),dtype=torch.long)
    return x,sst8,sst3

class CustomData2(Dataset):
    def __init__(self, seq, sst8, sst3, tokenizer, sst8_vocab, sst3_vocab):
        self.seq = seq
        self.sst8 = sst8
        self.sst3 = sst3
        self.tokenizer = tokenizer
        self.sst8_vocab = sst8_vocab
        self.sst3_vocab = sst3_vocab

    def __getitem__(self, idx):
        x = self.seq[idx]

        inputs = self.tokenizer(
            x,
            truncation=True,
            max_length=512,
            padding=False,
            return_special_tokens_mask=True
        )



        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        special_mask = inputs["special_tokens_mask"]

        l1 = convert_to_integer(self.sst8[idx], self.sst8_vocab,len(self.sst8[idx]))
        l2 = convert_to_integer(self.sst3[idx], self.sst3_vocab,len(self.sst3[idx]))

        labels1 = []
        labels2 = []

        token_idx = 0
        for is_special in special_mask:
            if is_special:
                labels1.append(-100)
                labels2.append(-100)
            else:
                labels1.append(l1[token_idx])
                labels2.append(l2[token_idx])
                token_idx += 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels1": labels1,
            "labels2": labels2,
        }

    def __len__(self):
        return len(self.seq)


train,test = get_splitted_data()
amino_vocab,sst8_vocab,sst3_vocab = get_vocabs()

def get_vocab_sizes():
   return len(amino_vocab),len(sst8_vocab),len(sst3_vocab)

def get_processed_train():
  
  
  train_set =  CustomData1(amino_vocab,sst8_vocab,sst3_vocab,train['seq'].to_numpy(),train['sst8'].to_numpy(),train['sst3'].to_numpy())
  

  return train_set

def get_processed_val():
   val_set = CustomData1(amino_vocab,sst8_vocab,sst3_vocab,test['seq'].to_numpy(),test['sst8'].to_numpy(),test['sst3'].to_numpy())
   return val_set


def get_processed_train2():
   train_seq = train['seq'].apply(lambda x:" ".join(x)).to_list()
   train_set = CustomData2(train_seq,train['sst8'].to_list(),train['sst3'].to_list(),tokenizer,sst8_vocab,sst3_vocab)
   return train_set
def get_processed_val2():
  test_seq = test['seq'].apply(lambda x:" ".join(x)).to_list()
  test_set = CustomData2(test_seq,test['sst8'].to_list(),test['sst3'].to_list(),tokenizer,sst8_vocab,sst3_vocab)
  return test_set

#def get_tokenizer():
#   return tokenizer

sst8_id2char = {v: k for k, v in sst8_vocab.items()}
sst3_id2char = {v: k for k, v in sst3_vocab.items()}
   
  

def convert_idx_sst8(preds):
    return "".join(
        sst8_id2char[idx.item()]
        for idx in preds
        if idx.item() != 0 and idx.item() != 1   
    )
   
def convert_idx_sst3(preds):
    return "".join(
        sst3_id2char[idx.item()]
        for idx in preds
        if idx.item() != 0   and idx.item() != 1
    )
   
   
def preprocess_input(x):
   x = torch.tensor(convert_to_integer(x,amino_vocab,len(x)),dtype=torch.long)
   return x 