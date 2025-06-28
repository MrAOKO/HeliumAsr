import torch
from model import DFCNN
import torch.nn as nn
import torch.optim as optim
from Dataset import HeliumASRDataset, extract_fbank, build_vocab, collate_fn
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
data_dir = 'D:/Code/Python/helium_asr/data/train'
vocab = build_vocab(data_dir)
train_set = HeliumASRDataset(data_dir, vocab, lambda x, sr: extract_fbank(x, sr, n_mels=80))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)

num_epochs = 10000
model = DFCNN(num_classes=len(vocab))
model.to(device)
#model.load_state_dict(torch.load("D:/model/dfcnn_epoch.pth"))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
best_loss = float('inf')
patience = 10
counter = 0
stop_training = False
for epoch in range(num_epochs):
    if stop_training:
        print("Training stopped early due to early stopping.")
        break
    model.train()
    test_loss =0
    for features, targets, input_lengths, target_lengths in train_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)  # (time, batch, num_classes)
        input_lengths = torch.full(size=(features.size(0),), fill_value=logits.shape[1], dtype=torch.long)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        test_loss += loss.item()
        avg_loss = test_loss / len(train_loader)
        print(f"Loss: {loss.item()}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            print(f"Epoch {epoch+1}, New best loss: {best_loss:.4f}")
            # Save the model after training
            torch.save(model.state_dict(), f"D:/model/DFCNN_epoch.pth")
            # Save the vocabulary
            with open('D:/model/vocab.pkl', 'wb') as f:
                pickle.dump(vocab, f)
            print(f"模型已保存: D:/model/DFCNN_epoch.pth")
            print(len(vocab), "个字符已保存到 D:/model/vocab.pkl")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}, no improvement for {patience} epochs.")
                #stop_training = True
                break
        
