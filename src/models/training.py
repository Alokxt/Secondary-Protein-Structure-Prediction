from src.data.load import get_processed_train
from src.data.processed import get_vocab_sizes
from src.models.Scratch import get_bilstm,get_birgu,get_birnn
import torch 
from sklearn.metrics import f1_score
import torch.nn as nn 
from src.data.load import get_train_loader
from src.plots.plot_results import plot_train
from src.models.transformer_archi import get
from src.models.transformer_archi import get_transformer
def training(model,train_load,loss_fn1,loss_fn2,opt,device,y1_size,y2_size,epochs=10):
    model = model.to(device)

    loss_history_y1 = []
    loss_history_y2 = []
    f1_scores_y1 = []
    f1_scores_y2 = []
    accuracy_y1 = []
    accuracy_y2 = []


    for epoch in range(epochs):
        model.train()
        y1_pred = []
        y2_pred = []
        y1_true = []
        y2_true = []
        correct_y1 =0
        correct_y2 = 0
        y1_loss =0
        y2_loss = 0
        total_y1 =0
        total_y2 = 0



        for xs, y1s, y2s, lengths in train_load:
            xs = xs.to(device)
            y1s = y1s.to(device)
            y2s = y2s.to(device)
            lengths = lengths.to(device)

            opt.zero_grad()

            y1_logits, y2_logits = model(xs, lengths)

            y1 = y1_logits.reshape(-1, y1_size)
            y2 = y2_logits.reshape(-1, y2_size)

            trues1 = y1s.view(-1)
            trues2 = y2s.view(-1)

            loss1 = loss_fn1(
                y1,
                trues1
            )

            loss2 = loss_fn2(
                y2,
                trues2
            )
            mask1 = trues1 != 0
            mask2 = trues2 != 0

            total_y1 += mask1.sum().item()
            total_y2 += mask2.sum().item()

            _,pred1 = torch.max(y1,1)
            _,pred2 = torch.max(y2,1)
            y1_pred.extend(pred1[mask1].detach().cpu().numpy())
            y2_pred.extend(pred2[mask2].detach().cpu().numpy())

            y1_true.extend(trues1[mask1].detach().cpu().numpy())
            y2_true.extend(trues2[mask2].detach().cpu().numpy())

            correct_y1 += ((pred1[mask1] == trues1[mask1])).sum().item()
            correct_y2 += ((pred2[mask2] == trues2[mask2])).sum().item()


            y1_loss += loss1.item()
            y2_loss += loss2.item()
            loss = loss1 + loss2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()

        y1_f1 = f1_score(y1_true,y1_pred,average="macro")
        y2_f1 = f1_score(y2_true,y2_pred,average="macro")
        acc_y1 = correct_y1/total_y1
        acc_y2 = correct_y2/total_y2

        loss_history_y1.append(y1_loss/len(train_load))
        loss_history_y2.append(y2_loss/len(train_load))
        f1_scores_y1.append(y1_f1)
        f1_scores_y2.append(y2_f1)
        accuracy_y1.append(acc_y1)
        accuracy_y2.append(acc_y2)

        log = {
            "sst_8_accuracy":acc_y1,
            "sst_8_f1_score":y1_f1,
            "sst_8_loss":y1_loss/len(train_load),
            "sst_3_accuracy":acc_y2,
            "sst_3_f1_score":y2_f1,
            "sst_3_loss":y2_loss/len(train_load)
        }


        print(f"Epoch {epoch}: Average batch loss  {log}")
    return loss_history_y1,loss_history_y2,f1_scores_y1,f1_scores_y2,accuracy_y1,accuracy_y2

models  = {
        "Bi-rnn":get_birnn(),
        "Bi-lstm":get_bilstm(),
        "Bi-Gru":get_birgu()
    }

train_load = get_train_loader(batch_size=32)
metrics = {}
vocab_size,y1_size,y2_size  = get_vocab_sizes()

def visualize_results(models,metrics):
    plot_train(models, metrics, "sst8_loss", "sst8 cross-entropy loss of different models over epoch", "Loss","plots/sst8_loss.png")
    plot_train(models, metrics, "sst8_f1_scores","sst8 f1_scores of different models over epoch", "f1","plots/sst8_f1.png")
    plot_train(models, metrics, "sst8_accuracy", "sst8 accuracy of different models over epoch", "accuracy","plots/sst8_accuracy.png")
    plot_train(models, metrics, "sst3_loss", "sst3 cross-entropy loss of different models over epoch", "Loss","plots/sst3_loss.png")
    plot_train(models, metrics, "sst3_f1_scores", "sst3 f1_scores  of different models over epoch", "f1","plots/sst3_f1.png")
    plot_train(models, metrics, "sst3_accuracy","sst3 accuracy of different models over epoch", "accuracy","plots/sst3_accuracy.png")


def train_models(models,epoch=10):
    
    for m in list(models.keys()):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss1 = nn.CrossEntropyLoss(ignore_index=0)
        loss2 = nn.CrossEntropyLoss(ignore_index=0)
        opt = torch.optim.Adam(models[m].parameters(),lr=3e-3)
        print(f" for model {m}")
        loss_y1,loss_y2,f1_y1,f1_y2,acc_y1,acc_y2 = training(models[m],train_load,loss1,loss2,opt,device,y1_size,y2_size,epochs=epoch)
        d = {
            "sst8_loss":loss_y1,
            "sst8_f1_scores":f1_y1,
            "sst8_accuracy":acc_y1,
            "sst3_loss":loss_y2,
            "sst3_f1_scores":f1_y2,
            "sst3_accuracy":acc_y2,
        }
        metrics[m] = d 
        torch.save(models[m].state_dict(), f"saved_models/saved_{models[m]}_model.pth")
        torch.cuda.empty_cache()
        visualize_results(models,metrics)


train_models(models)

models2 = {
    "transformer":get_transformer()
}

train_models(models2)



