import torch 
import torch.nn as nn 
from sklearn.metrics import f1_score
from src.models.training import models
from src.data.load import get_validation_loader
from src.data.processed import get_vocab_sizes
from src.plots.plot_results import plot_eval
from src.models.transformer_archi import get_transformer

def evaluate(model, dataloader, device,y1_size,y2_size):
    model.eval()
   
    total_tokens_y1 = 0
    total_tokens_y2 = 0
    all_y1_pred = []
    all_y1_true = []
    all_y2_pred = []
    all_y2_true = []
    

    correct_y1 = 0
    correct_y2 = 0

    with torch.no_grad():
        for x, y1, y2, lengths in dataloader:
            x = x.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            lengths = lengths.to(device)

            out1, out2 = model(x, lengths)
          
       
            out1 = out1.reshape(-1, y1_size)
            out2 = out2.reshape(-1, y2_size)

            y1 = y1.reshape(-1)
            y2 = y2.reshape(-1)


            mask1 = y1 != 0
            mask2 = y2 != 0 
           

            pred1 = out1.argmax(dim=1)
            pred2 = out2.argmax(dim=1)
            all_y1_pred.extend(pred1[mask1].cpu().numpy())
            all_y2_pred.extend(pred2[mask2].cpu().numpy())
            all_y1_true.extend(y1[mask1].cpu().numpy())
            all_y2_true.extend(y2[mask2].cpu().numpy())

            correct_y1 += ((pred1 == y1) & mask1).sum().item()
            correct_y2 += ((pred2 == y2) & mask2).sum().item()
            
            total_tokens_y1 += mask1.sum().item()
            total_tokens_y2 += mask2.sum().item()
    y1_f1_val = f1_score(all_y1_true,all_y1_pred,average="macro")
    y2_f1_val = f1_score(all_y2_true,all_y2_pred,average="macro")
   
    acc_y1 = correct_y1 / max(total_tokens_y1,1)
    acc_y2 = correct_y2 / max(total_tokens_y2 ,1)
    print(f"sst8 : accuracy: {acc_y1} , f1_score: {y1_f1_val}")
    print(f"sst3 : accuracy: {acc_y2} , f1_score: {y2_f1_val}")


    return acc_y1, acc_y2,y1_f1_val,y2_f1_val

vocab_size,y1_size,y2_size = get_vocab_sizes()
val_load = get_validation_loader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_metrics = {}



for m in list(models.keys()):
  models[m].load_state(torch.load("saved_models/saved_{m}_model.pth", map_location=device))
  print(f"for model {m} ")
  acc_y1, acc_y2,y1_f1_val,y2_f1_val = evaluate(models[m],val_load,device,y1_size,y2_size)
  d = {
      "sst8_accuracy":acc_y1,
      "sst8_f1_scores":y1_f1_val,
      "sst3_accuracy":acc_y2,
      "sst3_f1_scores":y2_f1_val
  }
  test_metrics[m] =d 


plot_eval(test_metrics,"sst8","plots/sst8_validation.png")
plot_eval(test_metrics,"sst3","plots/sst3_validation.png")


