from sklearn.model_selection import train_test_split
import pandas as pd 

data = pd.read_csv('src//data//train (2).csv')

def get_vocabs():
  amino_acids = {'PAD':0,'Unk':1}
  sst_8 = {'PAD':0,'Unk':1}
  sst_3 = {'PAD':0,'Unk':1}
  for i in range(data.shape[0]):
    d = data.iloc[i]
    for j in d['seq']:
      if j not in amino_acids:
        amino_acids[j] = len(amino_acids)
    for j in d['sst8']:
      if j not in sst_8:
        sst_8[j] = len(sst_8)
    for j in d['sst3']:
      if j not in sst_3:
        sst_3[j] = len(sst_3)

  return amino_acids,sst_8,sst_3



def get_splitted_data():
    train,test = data[:5000],data[5000:]

    return train,test 

