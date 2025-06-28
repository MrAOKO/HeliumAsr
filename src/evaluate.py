import torch
from model import DFCNN
import torch.nn as nn
from Dataset import HeliumASRDataset, extract_fbank, build_vocab, collate_fn
from jiwer import cer, wer  
import pickle
# Load vocabulary
with open('D:/model/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
print(f"加载词表成功，共 {len(vocab)} 个字符")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'D:/Code/Python/helium_asr/data/train'
eval_set = HeliumASRDataset(data_dir, vocab, lambda x, sr: extract_fbank(x, sr, n_mels=80))
eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = DFCNN(num_classes=len(vocab))
model.load_state_dict(torch.load("D:/model/DFCNN_ResTrans_epoch.pth"))
model.to(device)
model.eval()

idx2char = {v: k for k, v in vocab.items()}

def ctc_greedy_decoder(log_probs, idx2char):
    pred = log_probs.argmax(dim=-1).transpose(0, 1)  
    results = []
    for seq in pred:
        last = None
        s = []
        for i in seq:
            i = i.item()
            if i != 0 and i != last:
                s.append(idx2char.get(i, ''))
            last = i
        results.append(''.join(s))
    return results

total_cer = 0
total_wer = 0
count = 0
with torch.no_grad():
    for features, targets, input_lengths, target_lengths in eval_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)
        preds = ctc_greedy_decoder(log_probs, idx2char)
        targets_str = []
        for t, l in zip(targets, target_lengths):
            targets_str.append(''.join([idx2char[i.item()] for i in t[:l]]))
        for pred, ref in zip(preds, targets_str):
            print(f"预测: {pred} | 真实: {ref}") 
            total_cer += cer(ref, pred)
            total_wer += wer(ref, pred)
            count += 1
print(f"平均CER: {total_cer/count:.4f}")
print(f"平均WER: {total_wer/count:.4f}")