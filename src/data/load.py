from torch.utils.data import DataLoader
from src.data.processed import get_processed_train,get_processed_val,get_processed_train2,get_processed_val2
import torch 
from torch.nn.utils.rnn import pad_sequence



def collate_fn(batch):
    if len(batch[0]) == 3:
        xs, y1s, y2s = zip(*batch)

        lengths = torch.tensor([len(x) for x in xs])

        xs  = pad_sequence(xs, batch_first=True, padding_value=0)
        if y1s[0] is not None and y2s[0] is not None:
            y1s = pad_sequence(y1s, batch_first=True, padding_value=0)
            y2s = pad_sequence(y2s, batch_first=True, padding_value=0)

            return xs, y1s, y2s, lengths
        return xs,lengths
    else:
        xs = batch
        lengths = torch.tensor([len(x) for x in xs])

        xs = pad_sequence(xs,batch_first=True,padding_value=0)

        return xs,lengths


def collate_fn2(batch):
    input_ids = [torch.tensor(b["input_ids"]) for b in batch]
    attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
    labels1 = [torch.tensor(b["labels1"]) for b in batch]
    labels2 = [torch.tensor(b["labels2"]) for b in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels1 = pad_sequence(labels1, batch_first=True, padding_value=-100)
    labels2 = pad_sequence(labels2, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels1": labels1,
        "labels2": labels2,
    }



def get_train_loader(batch_size=32,pin_memory=True):
    train_set = get_processed_train()
    train_load = DataLoader(train_set,batch_size=batch_size,collate_fn=collate_fn,shuffle=True,pin_memory=pin_memory)
    return train_load


def get_validation_loader(batch_size=32,pin_memory=True):
    val_set = get_processed_val()
    val_load = DataLoader(val_set,batch_size=batch_size,shuffle=False,collate_fn=collate_fn,pin_memory=pin_memory)
    return val_load



def get_train_loader2(batch_size=1,pin_memory=True):
    train_set = get_processed_train2()
    train_load = DataLoader(train_set,batch_size=batch_size,collate_fn=collate_fn2,shuffle=True,pin_memory=pin_memory)
    return train_load


def get_validation_loader2(batch_size=1,pin_memory=True):
    val_set = get_processed_val2()
    val_load = DataLoader(val_set,batch_size=batch_size,shuffle=False,collate_fn=collate_fn2,pin_memory=pin_memory)
    return val_load
