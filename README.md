# GPT From Scratch Chatbot (MLX)

A small decoder-only Transformer (causal multi-head self-attention, about 11M parameters) trained from scratch on
Apple Silicon with MLX. English only.

## How to run

Open `gpt_from_scratch_chatbot.ipynb` in VS Code, select the kernel **GPT MLX Python 3.13**, restart the kernel, and
Run All. Top to bottom:

1. **First training** (section 7): `make_tokens(TRAIN_FILES)` + `train_model(...)` on one big starting file,
   FineWeb-Edu (clean educational English, ~170 shards / ~12 h). Once trained, this step takes seconds.
2. **Chat** with the model, then **check accuracy** on fixed test data: perplexity and top-1/top-5 next-token
   accuracy on held-out wiki, chat and TinyStories text, plus a chat quiz with grammar checks.
3. **Retrain** (section 10): with `ONE_FILE_PER_RUN = True`, trains only the **next unfinished file** of
   `RETRAIN_FILES` and then checks accuracy again. Run All again for the next file.

`RETRAIN_FILES` is in training order: knowledge files (Simple Wikipedia, Wikipedia parts) alternate with chat files
(lmsys, SmolTalk), and it ends with chat, so the model finishes as a chatbot. Files the checkpoint has already
trained are skipped automatically. Interrupting saves the exact position; Run All continues from there.

`make_tokens` detects each file's kind: txt/csv chat files, Wikipedia XML, chat parquet (`conversation` or
`messages` column) and plain-text parquet (`text` column). It cleans the text (English ASCII, repairs broken
encodings, keeps complete sentences from wiki) and saves ~5M-token shards in `data/shards/`, so each file is
tokenized only once.

## Files

- `gpt_from_scratch_chatbot.ipynb`: everything (cleaning, tokens, model, training, chat, accuracy, retrain).
- `tokenizer.json`: English-only byte-level BPE (12,474 tokens). It lowercases and folds accents.
- `checkpoints/chatbot_mlx.*`: weights, optimizer state, and a JSON log of trained shards.
  `chatbot_mlx.previous.*` is the checkpoint from before the last shard.
- `data/shards/`: tokens per source file (safe to delete; rebuilt when needed).
- `data/eval_history.jsonl`: every accuracy check, shown as a progress table by `show_eval_history()`.
- `chat_dataset/`: raw data; `chat_dataset/grammar/` has TinyStories, FineWeb-Edu, Simple Wikipedia and SmolTalk.

## Growing the model (model_002)

`CHECKPOINT = "checkpoints/model_002"` is created the first time by section 6b: a bigger copy of
`model_001_best` (12 layers × 512 wide, 16 heads, 512-token context, ~51M parameters) that starts out making the
same predictions and keeps its record of trained data. It trains at ~3,900 tokens/s (1B tokens ≈ 3 days).

Its `RETRAIN_FILES` entries (data in `chat_dataset/understanding/` and FineWeb-Edu 001/002):
1. FineWeb-Edu 001 + SQuAD v2 (answer from a text, or reply "i don't have proper information") + Dolly-15k
2. FineWeb-Edu 002 + wiki 6 + wiki 7 (optional, more knowledge)
3. Final polish: OpenAssistant (English only) + everyday conversations + your txt files

The accuracy check adds a **reading test** (SQuAD v2 validation: answers found / "no information" when the answer
is missing) and questions about made-up things, where "I don't know" is the right answer.

## Training again later

| You want to... | Do this |
|---|---|
| Continue after stopping | restart the kernel, Run All |
| Train the next file | Run All again (it picks the next unfinished file of `RETRAIN_FILES`) |
| Train the whole list in one go | `ONE_FILE_PER_RUN = False` (`MIX_FILES = True` interleaves the files) |
| Add new data | add files to `RETRAIN_FILES` at the position you want them trained |
| Train over everything again | `TRAINING_ROUND = 2` (then 3, 4 ...) |
| Undo the last shard | copy `checkpoints/chatbot_mlx.previous.*` over `checkpoints/chatbot_mlx.*` |
| Start a fresh model | `CHECKPOINT = "checkpoints/chatbot_fresh"` (any new name; optionally edit `NEW_MODEL`), then Run All. The old model stays untouched and the tokens are reused. |

Only one process can train at a time: a second one stops with an "another training process" error. Stop a
background run with `kill <pid>` (it saves first).
