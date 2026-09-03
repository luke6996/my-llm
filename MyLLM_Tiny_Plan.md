# MyLLM-Tiny：從零訓練小型 LLM 計畫書

## 1. 專案目標

使用 **Google Colab 免費版 GPU**，從零實作並訓練一個小型 Decoder-only Transformer，完整走過 LLM 的基本生命週期：

```text
Dataset
  ↓
Tokenizer
  ↓
Tokenization / Packing
  ↓
Transformer
  ↓
Pretraining
  ↓
Checkpoint
  ↓
Text Generation
```

本專案的目標不是訓練出可與商用 LLM 比較的模型，而是理解並親手實作：

- BPE Tokenizer
- Token Embedding
- RMSNorm
- RoPE
- Grouped Query Attention（GQA）
- SwiGLU
- Decoder-only Transformer
- Causal Language Modeling Loss
- AdamW + Learning Rate Scheduler
- Checkpoint / Resume Training
- Autoregressive Text Generation

---

## 2. 執行環境

主要環境：

- Google Colab 免費版
- NVIDIA T4 GPU（若當次 runtime 有分配到）
- Python
- PyTorch
- Hugging Face `datasets`
- Hugging Face `tokenizers`

每次啟動 Colab 時先確認 GPU：

```bash
!nvidia-smi
```

由於免費 Colab runtime 可能中斷，因此所有重要 checkpoint 必須同步保存至 Google Drive。

---

## 3. 第一版模型規格

### MyLLM-Tiny

| 參數 | 設定 |
|---|---:|
| Architecture | Decoder-only Transformer |
| Parameters | 約 8M–10M |
| Vocabulary Size | 8,192 |
| Layers | 6 |
| Hidden Size | 256 |
| Q Heads | 4 |
| KV Heads | 2 |
| Head Dimension | 64 |
| FFN Hidden Size | 768 |
| Context Length | 256 |
| Position Encoding | RoPE |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Attention | GQA |
| Precision | FP16 |
| Weight Tying | Yes |

預計設定：

```python
config = {
    "vocab_size": 8192,
    "d_model": 256,
    "n_layers": 6,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 64,
    "ffn_dim": 768,
    "max_seq_len": 256,
    "dropout": 0.0,
}
```

---

## 4. Dataset

第一版使用公開 LLM pretraining dataset，例如：

- FineWeb-Edu

使用 streaming 模式讀取，避免一次下載完整資料集。

```python
from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    "sample-10BT",
    split="train",
    streaming=True,
)
```

### 第一版資料量

採分階段方式：

| Phase | Training Tokens | 目的 |
|---|---:|---|
| Phase 1 | 1M | 確認 pipeline 與 loss 正常 |
| Phase 2 | 5M | 確認模型開始學習語言統計 |
| Phase 3 | 10M | 第一版完整實驗 |
| Optional | 20M–50M | GPU 額度允許時延伸 |

---

## 5. Tokenizer

自行訓練 BPE Tokenizer，不使用 GPT-2、Llama 等既有 tokenizer。

目標：

```text
Raw Text
   ↓
BPE Training
   ↓
Vocabulary = 8192
   ↓
Token IDs
```

Special Tokens：

```text
<pad>
<unk>
<bos>
<eos>
```

Tokenizer 訓練完成後保存：

```text
artifacts/
└── tokenizer.json
```

---

## 6. Data Pipeline

資料處理流程：

```text
FineWeb-Edu
     ↓
Raw documents
     ↓
Tokenizer
     ↓
EOS separator
     ↓
Token stream
     ↓
Sequence Packing
     ↓
256-token blocks
     ↓
Training Batch
```

第一版不做複雜的資料品質篩選，主要目標是確保完整 pretraining pipeline 能運作。

---

## 7. 模型實作

模型模組預計拆成：

```text
Token Embedding
      ↓
Transformer Block × 6
      │
      ├── RMSNorm
      ├── GQA + RoPE
      ├── Residual
      ├── RMSNorm
      ├── SwiGLU
      └── Residual
      ↓
Final RMSNorm
      ↓
LM Head
      ↓
Vocabulary Logits
```

### 需要自行實作

- `RMSNorm`
- `RoPE`
- `GroupedQueryAttention`
- `SwiGLU`
- `TransformerBlock`
- `MyLLM`

Attention 的底層計算第一版可以使用：

```python
torch.nn.functional.scaled_dot_product_attention
```

第一版不實作 Triton 或 CUDA kernel。

---

## 8. Training Objective

使用標準 autoregressive next-token prediction。

例如：

```text
Input:
[A, B, C, D]

Target:
[B, C, D, E]
```

Loss：

```python
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    input_ids[:, 1:].reshape(-1)
)
```

---

## 9. Training 設定

初始設定：

| 設定 | 值 |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 0.1 |
| Scheduler | Cosine Decay |
| Warmup | 約總 steps 的 1% |
| Gradient Clipping | 1.0 |
| Precision | FP16 |
| Context Length | 256 |

Batch size 依實際 GPU VRAM 調整。

例如：

```text
Micro Batch = 8
Gradient Accumulation = 8
Sequence Length = 256
```

若 OOM：

```text
Micro Batch
8 → 4 → 2
```

並增加 gradient accumulation 維持相近的 global batch。

---

## 10. Checkpoint / Resume

免費 Colab runtime 可能隨時中斷，因此 training 必須支援 resume。

Checkpoint 至少保存：

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "tokens_seen": tokens_seen,
}
```

建議保存到：

```text
Google Drive
└── MyLLM-Tiny/
    └── checkpoints/
        ├── checkpoint_1000.pt
        ├── checkpoint_2000.pt
        └── latest.pt
```

訓練流程：

```text
Colab Session #1
0 → 1M tokens
      ↓
checkpoint

Colab Session #2
resume
1M → 5M tokens
      ↓
checkpoint

Colab Session #3
resume
5M → 10M tokens
```

---

## 11. Evaluation

第一版主要觀察：

### Training

- Training Loss
- Validation Loss
- Tokens / Second
- GPU VRAM Usage
- Tokens Seen

### Generation

固定使用幾組 prompt：

```text
The meaning of life is

Artificial intelligence is

The capital of France is
```

觀察模型從：

```text
random / meaningless output
```

逐漸變成：

```text
具有基本英文語法與局部語意的 output
```

---

## 12. 專案結構

預計專案：

```text
myllm-tiny/
├── README.md
├── config.py
├── tokenizer.py
├── data.py
├── layers.py
├── attention.py
├── model.py
├── train.py
├── generate.py
├── eval.py
│
├── notebooks/
│   └── colab.ipynb
│
├── artifacts/
│   └── tokenizer.json
│
└── checkpoints/
```

Colab Notebook 主要負責：

1. 安裝 dependencies
2. Mount Google Drive
3. Clone / Load project
4. 啟動 training
5. Resume checkpoint
6. 執行 generation / evaluation

核心模型邏輯保留在 `.py` 檔案中。

---

## 13. 開發階段

### Milestone 1 — Environment

- 建立 Colab
- 確認 GPU
- 安裝套件
- Mount Google Drive

### Milestone 2 — Tokenizer

- Streaming 讀取 FineWeb-Edu
- 收集 tokenizer training corpus
- Train 8K BPE tokenizer
- 驗證 encode / decode

### Milestone 3 — Data Pipeline

- Tokenization
- EOS insertion
- Sequence packing
- 產生 256-token batches

### Milestone 4 — Model

依序完成：

1. RMSNorm
2. RoPE
3. GQA
4. SwiGLU
5. TransformerBlock
6. MyLLM

每個 component 都先做 shape / numerical sanity check。

### Milestone 5 — Training

- Forward pass
- Causal LM Loss
- Backpropagation
- AdamW
- LR Scheduler
- FP16
- Gradient clipping

先 overfit 小 batch，確認模型真的能學習。

### Milestone 6 — Pretraining

依序：

```text
1M tokens
↓
5M tokens
↓
10M tokens
```

持續觀察 loss 與 generation。

### Milestone 7 — Generation

自行實作 autoregressive generation：

- Greedy decoding
- Temperature
- Top-k sampling

---

## 14. 第一版不做的事情

為控制免費 Colab 的計算量與專案複雜度，第一版暫時不做：

- Hugging Face `AutoModel`
- Hugging Face `Trainer`
- Pretrained weights
- Pretrained tokenizer
- Triton
- CUDA kernel
- FlashAttention custom kernel
- Distributed Training
- FSDP / Tensor Parallel
- MoE
- MLA
- MTP
- SFT
- DPO / RLHF
- Long Context

---

## 15. 後續實驗方向

第一版完成後，可以逐步加入 architecture experiments：

```text
MyLLM v0
GQA Transformer
     ↓
MyLLM v1
MHA vs GQA
     ↓
MyLLM v2
RMSNorm vs LayerNorm
     ↓
MyLLM v3
SwiGLU vs GELU
     ↓
MyLLM v4
Triton RMSNorm / RoPE
     ↓
MyLLM v5
MLA
     ↓
MyLLM v6
MoE
     ↓
MyLLM v7
MTP
```

每次只改一個主要變因，固定比較：

- Validation Loss
- Training Throughput
- VRAM Usage
- Generation Quality
- Tokens / Second

---

## 16. 專案成功條件

第一版完成的判定標準：

- [ ] 自行訓練 tokenizer
- [ ] Tokenizer 可以 encode / decode
- [ ] Dataset 可以 streaming
- [ ] Sequence packing 正常
- [ ] 自行實作 Transformer
- [ ] Causal masking 正確
- [ ] Forward pass 正常
- [ ] Loss 可以持續下降
- [ ] 可以保存 checkpoint
- [ ] Colab 中斷後可以 resume
- [ ] 可以自行 autoregressive generate
- [ ] Output 開始出現基本自然語言結構

達成以上條件，即視為成功完成第一個 **from-scratch LLM pretraining project**。
