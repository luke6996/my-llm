import tempfile
import unittest
from pathlib import Path

import torch

from myllm_tiny.config import ModelConfig
from myllm_tiny.model import MyLLM
from myllm_tiny.train import build_scheduler, load_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        config = ModelConfig(
            vocab_size=16,
            d_model=16,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            head_dim=4,
            ffn_dim=32,
            max_seq_len=8,
        )
        model = MyLLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, warmup_steps=1, total_steps=4)
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "latest.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                scaler,
                step=7,
                tokens_seen=896,
            )

            restored_model = MyLLM(config)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            restored_scheduler = build_scheduler(
                restored_optimizer, warmup_steps=1, total_steps=4
            )
            restored_scaler = torch.cuda.amp.GradScaler(enabled=False)
            step, tokens_seen = load_checkpoint(
                path,
                restored_model,
                restored_optimizer,
                restored_scheduler,
                restored_scaler,
            )

        self.assertEqual((step, tokens_seen), (7, 896))
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()

