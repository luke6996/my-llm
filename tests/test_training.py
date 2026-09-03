import unittest

import torch

from myllm_tiny.config import ModelConfig
from myllm_tiny.model import MyLLM
from myllm_tiny.train import causal_lm_loss


class TrainingTests(unittest.TestCase):
    def test_single_batch_can_overfit(self):
        torch.manual_seed(1)
        config = ModelConfig(
            vocab_size=16,
            d_model=24,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            head_dim=6,
            ffn_dim=48,
            max_seq_len=8,
        )
        model = MyLLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        initial_loss = float(causal_lm_loss(model, input_ids).detach())
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = causal_lm_loss(model, input_ids)
            loss.backward()
            optimizer.step()
        final_loss = float(causal_lm_loss(model, input_ids).detach())
        self.assertLess(final_loss, initial_loss * 0.35)


if __name__ == "__main__":
    unittest.main()
