# VietSpeech Scratch TTS — research baseline

An **independent, randomly initialized Vietnamese text-to-mel speech synthesis baseline**. No VieNeu weights or pretrained TTS checkpoint is loaded. This is a first working research baseline, **not** a claim of superior quality to VieNeu. The baseline uses an autoregressive encoder-decoder Transformer and Griffin-Lim waveform reconstruction; a production-quality neural codec/vocoder, forced alignment, prosody and streaming remain future work.

## Features
- Unicode Vietnamese character tokenization preserving tonal marks, plus normalization of whitespace and common punctuation.
- Trainable text encoder, causal mel decoder and stop-token predictor, all initialized from scratch.
- Explicit speaker-independent single-speaker data format; train/test split and gradient clipping.
- End-to-end train, checkpoint, evaluation, and synthesis CLIs.
- Reproducible seed, trainable small model, configuration options.
- Unit tests covering normalization, text padding and model tensor shapes.

## Setup

Python 3.10–3.12 recommended. Install a matching PyTorch/torchaudio build for your CUDA version first, then:

```bash
pip install -r requirements.txt
```

Prepare **consented/licensed** recordings of one speaker (24 kHz mono WAV recommended, 2–10 seconds per utterance), with clean transcripts. Example CSV:

```csv
audio_path,text
audio/0001.wav,Xin chào mọi người.
audio/0002.wav,Hôm nay thời tiết rất đẹp.
```

Paths are relative to the CSV directory. Do not put copyrighted/non-consensual voice recordings into a training set.

## Train from scratch

```bash
python train.py --metadata data/metadata.csv --output runs/small --epochs 100 --batch-size 8 --d-model 256
```

## Synthesize

```bash
python synthesize.py --checkpoint runs/small/best.pt --text "Xin chào Việt Nam." --output output.wav
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Limits and next experiments

This baseline intentionally **does not** use pre-trained weights. It does not have a neural audio codec, speaker cloning, multilingual voice support, or a streaming server. Teacher-forced mel regression can suffer from exposure bias, and Griffin-Lim sounds less natural than a trained neural vocoder. Quality depends strongly on clean recordings, good alignments and training time. Compare models using Vietnamese tone error rate, MOS with human raters, intelligibility via ASR, real-time factor on the same GPU, and zero-shot speaker similarity **only after those features have been implemented**. Add a neural codec/flow-matching model, robust alignment and multi-speaker conditioning in later phases.

Security: checkpoint files use PyTorch serialization and must be loaded only from trusted sources.
