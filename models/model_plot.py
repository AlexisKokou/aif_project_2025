import torch.nn as nn
import tqdm
import torch
class PlotClassification(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        for name, param in encoder.named_parameters():
            if not "classifier" in name:
                param.requires_grad = False
        self.encoder = encoder        

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        att = out.attentions
        hidden_states = out.hidden_states
        return logits, att, hidden_states
    


def train_model(model, optimizer, dataloader, criterion,epochs=5, device = "cuda"):
  model.train()
  for epoch in range(epochs):
    running_loss = 0.0

    t = tqdm.tqdm(dataloader)
    for i, batch in enumerate(t):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds,_,_ = model(input_ids,attention_mask=attention_mask)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        t.set_description(f"epoch:{epoch+1} loss: {(running_loss / (i+1)):.4f}")


def test_model(model, dataloader, device="cuda"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        t = tqdm.tqdm(dataloader)
        for batch in t:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds,_,_ = model(input_ids, attention_mask=attention_mask)

            predicted_classes = torch.argmax(preds, dim=1)
            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

            t.set_description(
                f"acc: {100 * correct / total:.2f}%"
            )

    accuracy = correct / total
    return accuracy