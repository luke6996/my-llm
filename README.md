# MyLLM-Tiny

MyLLM-Tiny 是依照 `MyLLM_Tiny_Plan.md` 從零實作的小型 decoder-only Transformer。第一版的目標是完整走過 tokenizer、資料 packing、模型 forward、next-token pretraining、checkpoint/resume 與 autoregressive generation，而不是追求商用模型的能力。

## 專案結構

```text
my-llm/
├── myllm_tiny/
│   ├── config.py       # 模型與訓練設定
│   ├── tokenizer.py    # BPE tokenizer 訓練與 encode/decode
│   ├── data.py         # streaming 文件、EOS 與 sequence packing
│   ├── layers.py       # RMSNorm、RoPE、SwiGLU
│   ├── attention.py    # GQA attention
│   ├── model.py        # decoder-only Transformer
│   ├── train.py        # pretraining、AMP、checkpoint/resume
│   ├── generate.py     # greedy / temperature / top-k generation
│   └── eval.py         # validation loss 與簡單生成評估
├── tests/
├── requirements.txt
└── MyLLM_Tiny_Plan.md
```

## 安裝

建議使用 Python 3.10+、PyTorch 2.0+：

```bash
pip install -r requirements.txt
```

在 Colab 中也可以直接執行：

```python
!pip install -r requirements.txt
```

## 先做本機 sanity check

不需要下載資料集即可驗證模型與資料 packing：

```bash
python -m unittest discover -s tests -v
python -m myllm_tiny.train --synthetic --total-tokens 32768 --checkpoint-dir checkpoints/smoke
```

`--synthetic` 只用重複的短英文文件，適合確認 loss 會下降；正式 pretraining 再使用 FineWeb-Edu。

## Colab 使用流程

### 1. 確認 GPU 與安裝套件

```python
!nvidia-smi
!pip install -r requirements.txt
```

### 2. 訓練 8K tokenizer

以下命令會 streaming 讀取 FineWeb-Edu，只取一小批文件建立 8K BPE tokenizer：

```bash
python -m myllm_tiny.tokenizer \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --max-documents 50000 \
  --vocab-size 8192 \
  --output artifacts/tokenizer.json
```

若要先做離線測試，也可以準備每行一份文件：

```bash
python -m myllm_tiny.tokenizer \
  --text-file data/train.txt \
  --vocab-size 8192 \
  --output artifacts/tokenizer.json
```

若只是想快速驗證完整 training pipeline，也可以跳過自訓步驟，改用 Hugging Face 上的 GPT-2 tokenizer。這只下載 tokenizer，不會使用 GPT-2 模型權重：

```bash
python -m myllm_tiny.train \
  --pretrained-tokenizer gpt2 \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --total-tokens 1000000
```

GPT-2 tokenizer 的 vocabulary 約 50K，因此模型會比使用 8K 自訓 tokenizer 大；正式的 8K 版本仍可在之後切換回 `--tokenizer artifacts/tokenizer.json`。

### 3. 開始第一階段 pretraining

```bash
python -m myllm_tiny.train \
  --tokenizer artifacts/tokenizer.json \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --total-tokens 1000000 \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir /content/drive/MyDrive/MyLLM-Tiny/checkpoints \
  --checkpoint-interval 25
```

Colab runtime 中斷後，使用相同參數並加上：

```bash
--resume /content/drive/MyDrive/MyLLM-Tiny/checkpoints/latest.pt
```

在 T4 OOM 時先把 `--micro-batch-size` 從 8 降到 4 或 2，再提高 gradient accumulation 維持近似 global batch。

### 4. 生成文字

```bash
python -m myllm_tiny.generate \
  --checkpoint checkpoints/latest.pt \
  --tokenizer artifacts/tokenizer.json \
  --prompt "Artificial intelligence is" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 50
```

## 設計上的第一版取捨

- 使用 `torch.nn.functional.scaled_dot_product_attention`，不自行寫 CUDA/Triton kernel。
- 生成先採完整 context 重算，沒有 KV cache；這讓第一版程式較容易驗證，之後再優化速度。
- tokenizer 與 FineWeb-Edu 依賴採 lazy import，因此模型單元測試只需要 PyTorch。
- checkpoint 會保存模型、optimizer、scheduler、AMP scaler、step、tokens seen 與 RNG state，方便 Colab resume。
