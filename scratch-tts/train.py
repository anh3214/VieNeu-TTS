"""Training CLI for randomly initialized autoregressive text-to-mel TTS."""
import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from tts import CharVocab, ScratchTTS, normalize

SR, N_FFT, HOP, N_MELS = 24000, 1024, 256, 80

class SpeechDataset(Dataset):
    def __init__(self, rows, vocab):
        self.rows, self.vocab = rows, vocab
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=N_FFT,
                         hop_length=HOP, n_mels=N_MELS, power=1.0)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, text = self.rows[index]
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.shape[-1] < N_FFT:
            raise ValueError(f"Too short: {path}")
        mel = torch.log1p(self.mel(wav)).squeeze(0).transpose(0, 1)
        return torch.tensor(self.vocab.encode(text), dtype=torch.long), mel

def collate(samples):
    texts, mels = zip(*samples)
    tx = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(m) for m in mels], dtype=torch.long)
    target = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True)
    inp = torch.cat([target.new_zeros(target.size(0), 1, target.size(2)), target[:, :-1]], dim=1)
    return tx, inp, target, lengths

def loss_fn(pred, stop_logits, target, lengths):
    steps = torch.arange(target.shape[1], device=target.device)[None, :]
    mask = steps < lengths[:, None]
    mel_loss = (torch.abs(pred - target) * mask.unsqueeze(-1)).sum() / (mask.sum() * target.size(-1)).clamp(min=1)
    stop_targets = (steps == (lengths - 1)[:, None]).float()
    # Balance stop frames and non-stop frames, ignore padded frames.
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_targets, reduction="none")
    weights = torch.where(stop_targets.bool(), 10.0, 1.0)
    stop_loss = (stop_loss * weights * mask).sum() / (weights * mask).sum().clamp(min=1)
    return mel_loss + 0.2 * stop_loss

def read_rows(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"audio_path", "text"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires audio_path,text headers")
        rows = [(path.parent / row["audio_path"], normalize(row["text"])) for row in reader]
    rows = [(p, t) for p, t in rows if t and p.is_file()]
    if len(rows) < 2:
        raise ValueError("Need at least two valid audio/transcript pairs")
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--output", default="runs/small")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-frames", type=int, default=2400)
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_rows(args.metadata)
    random.shuffle(rows)
    nval = max(1, int(len(rows) * .05))
    train_rows, val_rows = rows[nval:], rows[:nval]
    vocab = CharVocab("".join(t for _, t in rows))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScratchTTS(len(vocab.symbols), d_model=args.d_model,
                       layers=args.layers, max_frames=args.max_frames).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    def loader(r, shuffle):
        return DataLoader(SpeechDataset(r, vocab), batch_size=args.batch_size,
                          shuffle=shuffle, collate_fn=collate, num_workers=0)
    train_dl, val_dl = loader(train_rows, True), loader(val_rows, False)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.
        for tx, inp, tgt, lengths in train_dl:
            if tgt.size(1) > args.max_frames or tx.size(1) > model.params["max_text"]:
                raise ValueError("Clip exceeds model max_frames/max_text; trim clips or increase the limit")
            tx, inp, tgt, lengths = [x.to(device) for x in (tx, inp, tgt, lengths)]
            pred, stop = model(tx, inp)
            loss = loss_fn(pred, stop, tgt, lengths)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            train_sum += loss.item()
        model.eval()
        val_sum = 0.
        with torch.no_grad():
            for tx, inp, tgt, lengths in val_dl:
                tx, inp, tgt, lengths = [x.to(device) for x in (tx, inp, tgt, lengths)]
                pred, stop = model(tx, inp)
                val_sum += loss_fn(pred, stop, tgt, lengths).item()
        val = val_sum / len(val_dl)
        print(f"epoch={epoch} train={train_sum/len(train_dl):.4f} val={val:.4f}", flush=True)
        state = {"model": model.state_dict(), "config": model.params,
                 "vocab": vocab.to_dict(), "epoch": epoch, "val_loss": val}
        torch.save(state, out / "last.pt")
        if val < best:
            best = val
            torch.save(state, out / "best.pt")

if __name__ == "__main__":
    main()
