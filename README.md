# Gideon: a GPT built from scratch

Sebastian Raschka's book *[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)*. I have transformed this into something epic.

## Files

| File | Chapter | What it does |
| --- | --- | --- |
| `data_preparation.py` | 2 | Tokenizes text with BPE (`tiktoken`) and builds a sliding-window dataset |
| `attention.py` | 3 | Self-attention, causal attention and multi-head attention |
| `gpt_model.py` | 4 | The GPT model: layer norm, GELU, transformer blocks and text generation |
| `pretraining.py` | 5 | Training loop, loss evaluation, temperature and top-k sampling |
| `gpt2_weights.py` | 5.5 | Downloads GPT-2 weights from Hugging Face (safetensors) and loads them into the model |
| `classification_finetune.py` | 6 | Fine-tunes GPT-2 to classify SMS messages as spam or not |
| `instruction_finetune.py` | 7 | Fine-tunes GPT-2 to follow instructions and scores its answers with Llama 3 |
| `the-verdict.txt` | 2, 5 | Short story used as training text |

Later chapters import from earlier ones, and each file runs its chapter's demos when run directly.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python gpt_model.py                # generate text with an untrained model
python pretraining.py              # pretrain on the-verdict.txt
python classification_finetune.py  # spam classifier
python instruction_finetune.py     # instruction fine-tuning
```

The scripts download what they need (GPT-2 weights, datasets) on first run. These files are gitignored.

Chapter 7 scores the model's answers with a local Llama 3 model. To run that step, install [Ollama](https://ollama.com) and run `ollama pull llama3`. Without it, the rest of the chapter still runs and the scoring step is skipped.

The scripts run on CPU, Apple Silicon (MPS) or CUDA. Chapter 7 defaults to GPT-2 small (124M) so it fits in 8 GB of memory. The book uses GPT-2 medium (355M), which you can switch to with `CHOOSE_MODEL` in `instruction_finetune.py`.

## Credits and license

The code is adapted from Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) and is shared under the same Apache 2.0 license. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
