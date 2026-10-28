import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tts import CharVocab, ScratchTTS, normalize

class CoreTest(unittest.TestCase):
    def test_unicode_tone(self):
        self.assertEqual(normalize("  Tiếng    Việt  "), "Tiếng Việt")
        vocab = CharVocab("Tiếng Việt")
        self.assertNotIn(3, vocab.encode("Tiếng Việt"))

    def test_shapes_and_backprop(self):
        model = ScratchTTS(20, d_model=32, nhead=4, layers=1)
        text = torch.tensor([[4, 5, 2], [6, 2, 0]])
        mel = torch.randn(2, 5, 80)
        pred, stop = model(text, mel)
        self.assertEqual(tuple(pred.shape), (2, 5, 80))
        self.assertEqual(tuple(stop.shape), (2, 5))
        (pred.square().mean() + stop.square().mean()).backward()
        self.assertIsNotNone(model.mel_out.weight.grad)

if __name__ == "__main__":
    unittest.main()
