"""Scratch Vietnamese text-to-mel Transformer. No pretrained weights."""
import re
import unicodedata
import torch
from torch import nn

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

class CharVocab:
    def __init__(self, chars):
        self.symbols = [PAD, BOS, EOS, UNK] + sorted(set(chars) - {PAD, BOS, EOS, UNK})
        self.lookup = {x: i for i, x in enumerate(self.symbols)}

    def encode(self, text):
        return [self.lookup.get(c, 3) for c in normalize(text)] + [2]

    def to_dict(self):
        return {"symbols": self.symbols}

    @classmethod
    def from_dict(cls, data):
        obj = cls([])
        obj.symbols = list(data["symbols"])
        obj.lookup = {c: i for i, c in enumerate(obj.symbols)}
        return obj

class ScratchTTS(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_mels=80, nhead=4, layers=4, max_text=1024, max_frames=2400):
        super().__init__()
        self.params = dict(vocab_size=vocab_size, d_model=d_model, n_mels=n_mels,
                           nhead=nhead, layers=layers, max_text=max_text, max_frames=max_frames)
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.txt_pos = nn.Embedding(max_text, d_model)
        self.mel_pos = nn.Embedding(max_frames + 1, d_model)
        self.mel_in = nn.Linear(n_mels, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout=.1,
                                       batch_first=True, norm_first=True), layers,
            enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, d_model * 4, dropout=.1,
                                       batch_first=True, norm_first=True), layers)
        self.mel_out = nn.Linear(d_model, n_mels)
        self.stop_out = nn.Linear(d_model, 1)
        self.n_mels = n_mels

    def encode_text(self, text):
        b, t = text.shape
        if t > self.params["max_text"]:
            raise ValueError("Text exceeds max_text")
        pos = torch.arange(t, device=text.device)[None, :]
        x = self.token(text) + self.txt_pos(pos)
        return self.encoder(x, src_key_padding_mask=text.eq(0))

    def decode(self, memory, text, mel_input):
        b, t, _ = mel_input.shape
        if t > self.params["max_frames"]:
            raise ValueError("Mel exceeds max_frames")
        pos = torch.arange(t, device=mel_input.device)[None, :]
        y = self.mel_in(mel_input) + self.mel_pos(pos)
        causal = nn.Transformer.generate_square_subsequent_mask(t, device=mel_input.device)
        y = self.decoder(y, memory, tgt_mask=causal, memory_key_padding_mask=text.eq(0))
        return self.mel_out(y), self.stop_out(y).squeeze(-1)

    def forward(self, text, mel_input):
        return self.decode(self.encode_text(text), text, mel_input)

    @torch.no_grad()
    def generate(self, text, max_frames=600, min_frames=25, stop_threshold=.6):
        self.eval()
        memory = self.encode_text(text)
        mel = torch.zeros(text.size(0), 1, self.n_mels, device=text.device)
        outputs = []
        for step in range(min(max_frames, self.params["max_frames"])):
            pred, stop = self.decode(memory, text, mel)
            frame = pred[:, -1:, :]
            outputs.append(frame)
            if step + 1 >= min_frames and torch.sigmoid(stop[:, -1]).min().item() > stop_threshold:
                break
            mel = torch.cat([mel, frame], dim=1)
        return torch.cat(outputs, dim=1)
