# GPT From Scratch Chatbot

This folder contains a small educational chatbot trained from scratch in PyTorch.

It is not production quality. It is meant to teach the complete pipeline:

1. Train or load a local BPE tokenizer.
2. Convert chat text into token IDs.
3. Train a decoder-only Transformer.
4. Save and load a checkpoint.
5. Generate assistant-style replies.

## Files

- `chat_data.txt`: tiny starter chat dataset.
- `chat_dataset/conversations.json`: structured conversation dataset used first when present.
- `chat_dataset/human_text.txt`: user-side conversation lines.
- `chat_dataset/robot_text.txt`: assistant-side conversation lines.
- `chat_dataset/labeled_conversations.txt`: paired conversations with `<user>:` and `<assistant>:` labels.
- `chat_dataset/daily_english_conversations.txt`: extra clean English practice conversations.
- `chat_dataset/ai_learning_conversations.txt`: extra AI and machine learning conversations.
- `chat_dataset/support_conversations.txt`: extra supportive conversation examples.
- `train_chatbot.py`: tokenizer, GPT model, training loop, saving, and chat generation.
- `chat.py`: load checkpoint and chat interactively.
- `tokenizer.json`: local BPE tokenizer created from the training text.
- `tokenizer_config.json`: metadata for future tokenizer reuse.
- `special_tokens_map.json`: special-token definitions.
- `model_config.json`: tiny model architecture metadata.

`conversations.json` is used during training when present. It stores conversations like:

```json
{
  "id": "daily_english_conversations-0001",
  "source": "daily_english_conversations.txt",
  "messages": [
    {"role": "user", "content": "hello, how are you today?"},
    {"role": "assistant", "content": "hello! i am doing well and ready to chat with you."}
  ]
}
```

The text files are kept as editable source/fallback data. Every `*_conversations.txt` file can be regenerated into JSON. `labeled_conversations.txt` was created by pairing `human_text.txt` and `robot_text.txt` line by line:

```text
<user>: line from human_text.txt
<assistant>: matching line from robot_text.txt
```

## Run

From the project root:

```bash
python3 GenAI/gpt_from_scratch_chatbot/train_chatbot.py
```

The script saves:

```text
GenAI/gpt_from_scratch_chatbot/checkpoints/tiny_chatbot_combined.pt
```

## Important

This model is intentionally tiny. It will not chat like ChatGPT. To improve it:

- Add much more chat data to `chat_data.txt`.
- Increase `TOKENIZER_VOCAB_SIZE` and retrain the tokenizer when you have more data.
- Increase `N_LAYER`, `N_HEAD`, and `N_EMBD`.
- Train longer by increasing `MAX_ITERS`.
