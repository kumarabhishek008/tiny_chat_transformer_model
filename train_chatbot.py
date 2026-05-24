"""
Train a tiny GPT-style chatbot from scratch in PyTorch.

This is an educational decoder-only Transformer. It does not use pretrained
weights, Hugging Face models, or BERT. The tokenizer is character-level so the
entire pipeline is visible in one file.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "chat_data.txt"
CHAT_DATASET_DIR = BASE_DIR / "chat_dataset"
CONVERSATIONS_JSON_PATH = CHAT_DATASET_DIR / "conversations.json"
LABELED_CONVERSATIONS_PATTERN = "*_conversations.txt"
LABELED_CONVERSATIONS_PATH = CHAT_DATASET_DIR / "labeled_conversations.txt"
HUMAN_TEXT_PATH = CHAT_DATASET_DIR / "human_text.txt"
ROBOT_TEXT_PATH = CHAT_DATASET_DIR / "robot_text.txt"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "tiny_chatbot_combined.pt"
TOKENIZER_PATH = BASE_DIR / "tokenizer.json"
TOKENIZER_CONFIG_PATH = BASE_DIR / "tokenizer_config.json"
SPECIAL_TOKENS_MAP_PATH = BASE_DIR / "special_tokens_map.json"
MODEL_CONFIG_PATH = BASE_DIR / "model_config.json"
TOKENIZER_VOCAB_SIZE = 2000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<user>", "<assistant>"]

BATCH_SIZE = 16
BLOCK_SIZE = 128
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1

torch.manual_seed(1337)


def load_text():
    parts = []

    if DATA_PATH.exists():
        parts.append(DATA_PATH.read_text(encoding="utf-8-sig").strip())

    if CONVERSATIONS_JSON_PATH.exists():
        dataset = json.loads(CONVERSATIONS_JSON_PATH.read_text(encoding="utf-8"))
        json_examples = []

        for conversation in dataset.get("conversations", []):
            lines = []
            for message in conversation.get("messages", []):
                role = message.get("role")
                content = message.get("content", "").strip()
                if role in {"user", "assistant"} and content:
                    lines.append(f"<{role}>: {content}")

            if lines:
                json_examples.append("\n".join(lines))

        parts.append("\n\n".join(json_examples))
    else:
        labeled_paths = sorted(CHAT_DATASET_DIR.glob(LABELED_CONVERSATIONS_PATTERN))
        for labeled_path in labeled_paths:
            parts.append(labeled_path.read_text(encoding="utf-8-sig").strip())
        if not labeled_paths and HUMAN_TEXT_PATH.exists() and ROBOT_TEXT_PATH.exists():
            human_lines = HUMAN_TEXT_PATH.read_text(encoding="utf-8-sig").splitlines()
            robot_lines = ROBOT_TEXT_PATH.read_text(encoding="utf-8-sig").splitlines()

            paired_examples = []
            for human_line, robot_line in zip(human_lines, robot_lines):
                human_line = human_line.strip()
                robot_line = robot_line.strip()

                if not human_line or not robot_line:
                    continue

                paired_examples.append(
                    f"<user>: {human_line}\n<assistant>: {robot_line}"
                )

            parts.append("\n\n".join(paired_examples))

    if not parts:
        raise FileNotFoundError(
            f"No training data found at {DATA_PATH} or {CHAT_DATASET_DIR}"
        )

    return "\n\n".join(parts).lower()


def save_project_configs(vocab_size):
    TOKENIZER_CONFIG_PATH.write_text(
        json.dumps(
            {
                "tokenizer_class": "ByteLevelBPETokenizer",
                "tokenizer_file": TOKENIZER_PATH.name,
                "model_max_length": BLOCK_SIZE,
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "additional_special_tokens": ["<user>", "<assistant>"],
                "chat_template": "<user>: {user_message}\n<assistant>:",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    SPECIAL_TOKENS_MAP_PATH.write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "additional_special_tokens": ["<user>", "<assistant>"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    MODEL_CONFIG_PATH.write_text(
        json.dumps(
            {
                "model_type": "tiny_gpt_from_scratch",
                "vocab_size": vocab_size,
                "block_size": BLOCK_SIZE,
                "n_embd": N_EMBD,
                "n_head": N_HEAD,
                "n_layer": N_LAYER,
                "dropout": DROPOUT,
                "tokenizer_file": TOKENIZER_PATH.name,
                "checkpoint_file": str(CHECKPOINT_PATH.relative_to(BASE_DIR)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class ChatTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text.lower()).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def train_or_load_tokenizer(text, retrain=False):
    if TOKENIZER_PATH.exists() and not retrain:
        print(f"Loading tokenizer from: {TOKENIZER_PATH}")
        return ChatTokenizer(Tokenizer.from_file(str(TOKENIZER_PATH)))

    print(f"Training tokenizer and saving to: {TOKENIZER_PATH}")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=TOKENIZER_VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    tokenizer.save(str(TOKENIZER_PATH))
    chat_tokenizer = ChatTokenizer(tokenizer)
    save_project_configs(chat_tokenizer.vocab_size)
    return chat_tokenizer


def make_batch(data):
    starts = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[start : start + BLOCK_SIZE] for start in starts])
    y = torch.stack([data[start + 1 : start + BLOCK_SIZE + 1] for start in starts])
    return x.to(DEVICE), y.to(DEVICE)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.query = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.value = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)),
        )

    def forward(self, x):
        batch, time, channels = x.shape
        head_size = channels // N_HEAD

        k = self.key(x).view(batch, time, N_HEAD, head_size).transpose(1, 2)
        q = self.query(x).view(batch, time, N_HEAD, head_size).transpose(1, 2)
        v = self.value(x).view(batch, time, N_HEAD, head_size).transpose(1, 2)

        weights = q @ k.transpose(-2, -1) * head_size**-0.5
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = CausalSelfAttention()
        self.feed_forward = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(N_EMBD)
        self.layer_norm_2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYER)])
        self.layer_norm = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, input_ids, targets=None):
        batch, time = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(time, device=DEVICE))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            logits_flat = logits.view(batch * time, channels)
            targets_flat = targets.view(batch * time)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            context = input_ids[:, -BLOCK_SIZE:]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(20):
            x, y = make_batch(data)
            _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def train():
    text = load_text()
    tokenizer = train_or_load_tokenizer(text)
    save_project_configs(tokenizer.vocab_size)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split_index = int(0.9 * len(encoded))
    train_data = encoded[:split_index]
    val_data = encoded[split_index:]

    model = TinyGPT(tokenizer.vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"device: {DEVICE}")
    print(f"characters in dataset: {len(text)}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    for step in range(MAX_ITERS + 1):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(
                f"step {step:04d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

        x, y = make_batch(train_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_path": str(TOKENIZER_PATH),
            "config": {
                "batch_size": BATCH_SIZE,
                "block_size": BLOCK_SIZE,
                "n_embd": N_EMBD,
                "n_head": N_HEAD,
                "n_layer": N_LAYER,
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"saved checkpoint: {CHECKPOINT_PATH}")

    return model, tokenizer


def chat_once(model, tokenizer, user_message, max_new_tokens=300):
    prompt = f"<user>: {user_message.lower()}\n<assistant>:"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(DEVICE)
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)[0].tolist()
    generated = tokenizer.decode(output_ids)
    answer = generated.split("<assistant>:")[-1]
    answer = answer.split("<user>:")[0]
    return answer.strip()


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}. Run train_chatbot.py first."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    tokenizer = ChatTokenizer(Tokenizer.from_file(checkpoint["tokenizer_path"]))
    model = TinyGPT(tokenizer.vocab_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def main():
    model, tokenizer = train()

    print("\nSample chats after training:")
    for question in [
        "hello",
        "what is a transformer?",
        "how do language models generate text?",
    ]:
        print(f"\nuser: {question}")
        print(f"assistant: {chat_once(model, tokenizer, question)}")


if __name__ == "__main__":
    main()
