"""Synthesize a WAV from a trusted scratch-TTS checkpoint."""
import argparse
from pathlib import Path

import torch
import torchaudio

from train import SR, N_FFT, HOP, N_MELS
from tts import CharVocab, ScratchTTS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--output", default="output.wav")
    p.add_argument("--max-frames", type=int, default=600)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocab = CharVocab.from_dict(checkpoint["vocab"])
    model = ScratchTTS(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    tx = torch.tensor([vocab.encode(args.text)], dtype=torch.long, device=device)
    mel_log = model.generate(tx, max_frames=args.max_frames)[0].T.cpu()
    mel_amp = torch.expm1(mel_log.clamp(min=0)).clamp(min=1e-5)
    inv = torchaudio.transforms.InverseMelScale(n_stft=N_FFT // 2 + 1,
                            n_mels=N_MELS, sample_rate=SR)
    griffin = torchaudio.transforms.GriffinLim(n_fft=N_FFT, hop_length=HOP, power=1.0, n_iter=60)
    wav = griffin(inv(mel_amp)).unsqueeze(0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.output, wav, SR)
    print(f"Saved {args.output}; {wav.shape[-1]/SR:.2f}s")

if __name__ == "__main__":
    main()
