# nano-gpt

A character-level GPT trained from scratch on Tiny Shakespeare, built step by step in a Colab notebook (following Karpathy's "Let's build GPT").

## Contents

| File | What it is |
|------|------------|
| `NanoGPT_updated.ipynb` | Main notebook: tokenizer → batching → bigram baseline → self-attention → multi-head attention + transformer block → GPT → training and sampling |
| `NanoGPT.ipynb` | Earlier version of the same notebook |
| `nanoGPT.pt` | Best checkpoint from the final training run (stored with Git LFS) |

## Model

- Character-level tokenizer, vocab size 65
- 6 layers, 6 heads, `n_embd=384`, context length 256, dropout 0.2
- ~10.8M parameters
- AdamW, lr 3e-4, batch size 64, 5000 iterations, mixed precision (fp16 autocast) on GPU
- 90/10 train/val split; the checkpoint is saved whenever val loss improves

## Checkpoint

`nanoGPT.pt` is a dict with `model` (state_dict), `optimizer`, `step` and `best_val`.

```bash
git lfs install
git clone https://github.com/Snehanjan2001/nano-gpt.git
```

Load it in the notebook after defining the `GPT` class and hyperparameters:

```python
ckpt = torch.load("nanoGPT.pt", map_location=device)
model = GPT(vocab_size, n_layer=6).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
```

## Running

Open `NanoGPT_updated.ipynb` in Google Colab with a GPU runtime and run all cells. It downloads `input.txt` (Tiny Shakespeare) itself.
