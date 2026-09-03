import unittest

import torch

from myllm_tiny.config import ModelConfig
from myllm_tiny.model import MyLLM


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = ModelConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            ffn_dim=64,
            max_seq_len=16,
        )

    def test_forward_and_loss_shapes(self):
        model = MyLLM(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (3, 8))
        logits, loss = model(input_ids, targets=input_ids)
        self.assertEqual(logits.shape, (3, 8, self.config.vocab_size))
        self.assertTrue(loss.ndim == 0)
        self.assertEqual(model.lm_head.weight.data_ptr(), model.token_embedding.weight.data_ptr())

    def test_future_tokens_do_not_change_past_logits(self):
        model = MyLLM(self.config).eval()
        prefix = torch.tensor([[1, 2, 3, 4]])
        changed = torch.tensor([[1, 2, 3, 7]])
        with torch.no_grad():
            first = model(prefix)
            second = model(changed)
        self.assertTrue(torch.allclose(first[:, :3], second[:, :3], atol=1e-6, rtol=1e-5))

    def test_rejects_long_context(self):
        model = MyLLM(self.config)
        input_ids = torch.zeros((1, self.config.max_seq_len + 1), dtype=torch.long)
        with self.assertRaises(ValueError):
            model(input_ids)


if __name__ == "__main__":
    unittest.main()

