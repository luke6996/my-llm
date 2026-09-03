import unittest

import torch

from myllm_tiny.data import batch_sequences, packed_sequences, token_stream


class FakeTokenizer:
    eos_id = 99
    bos_id = 98

    def encode(self, text, *, add_bos=False, add_eos=False):
        ids = [int(char) for char in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids


class DataPipelineTests(unittest.TestCase):
    def test_token_stream_adds_eos_between_documents(self):
        result = list(token_stream(["12", "3"], FakeTokenizer()))
        self.assertEqual(result, [1, 2, 99, 3, 99])

    def test_packing_is_non_overlapping(self):
        sequences = list(packed_sequences(["123", "45"], FakeTokenizer(), 4))
        self.assertEqual([sequence.tolist() for sequence in sequences], [[1, 2, 3, 99]])

    def test_batching(self):
        sequences = (torch.tensor([i, i + 1]) for i in range(0, 6, 2))
        batches = list(batch_sequences(sequences, 2))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()

