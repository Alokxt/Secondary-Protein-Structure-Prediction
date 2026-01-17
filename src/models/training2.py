import torch 
import torch.nn as nn 
from src.data.processed import get_vocab_sizes
from src.models.pretrained import get_pretrained_train
from src.data.load import get_train_loader2,get_validation_loader2
from src.plots.plot_results import plot_multiple
amino_vocab,y1_size,y2_size = get_vocab_sizes()

loss_fn1 = nn.CrossEntropyLoss(ignore_index=-100)
loss_fn2 = nn.CrossEntropyLoss(ignore_index=-100)

train_loader = get_train_loader2()
test_loader = get_validation_loader2()

model = get_pretrained_train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3)


def train_pretrained(model, loader, optimizer, device,loss_fn1,loss_fn2):
    model.train()
    total_loss = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels1 = batch["labels1"].to(device)
        labels2 = batch["labels2"].to(device)

        optimizer.zero_grad()

        y1_logits, y2_logits = model(input_ids, attention_mask)


        loss1 = loss_fn1(
            y1_logits.view(-1, y1_logits.size(-1)),
            labels1.view(-1)
        )
        loss2 = loss_fn2(
            y2_logits.view(-1, y2_logits.size(-1)),
            labels2.view(-1)
        )

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

from sklearn.metrics import f1_score
import torch

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_y1_preds = []
    all_y1_true = []

    all_y2_preds = []
    all_y2_true = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels1 = batch["labels1"].to(device)
        labels2 = batch["labels2"].to(device)

        y1_logits, y2_logits = model(input_ids, attention_mask)

        y1_preds = torch.argmax(y1_logits, dim=-1)
        y2_preds = torch.argmax(y2_logits, dim=-1)

        # Flatten + mask (-100)
        mask1 = labels1 != -100
        mask2 = labels2 != -100

        all_y1_preds.extend(y1_preds[mask1].cpu().tolist())
        all_y1_true.extend(labels1[mask1].cpu().tolist())

        all_y2_preds.extend(y2_preds[mask2].cpu().tolist())
        all_y2_true.extend(labels2[mask2].cpu().tolist())

    f1_y1 = f1_score(all_y1_true, all_y1_preds, average="micro")
    f1_y2 = f1_score(all_y2_true, all_y2_preds, average="micro")

    return f1_y1, f1_y2


losses = []
y1_f1 = []
y2_f1 = []
epochs = 5
for epoch in range(epochs):
    train_loss = train_pretrained(model, train_loader, optimizer, device,loss_fn1,loss_fn2)
    torch.save(model.state_dict(), f"model_weights_{epoch}.pt")
    f1_y1,f1_y2 = evaluate(model, test_loader, device)
    losses.append(train_loss)
    y1_f1.append(f1_y1)
    y2_f1.append(f1_y2)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val sst8 f1 score:   {f1_y1:.4f}")
    print(f"Val sst3 f1 score:   {f1_y2:.4f}")



d = {
    "loss":losses,
    "sst8_f1_scores":y1_f1,
    "sst3_f1_scores":y2_f1
}

plot_multiple(d)