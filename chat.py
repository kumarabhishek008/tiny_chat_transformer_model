"""
Chat with the tiny GPT model after training.

Run training first:
    python3 GenAI/gpt_from_scratch_chatbot/train_chatbot.py

Then chat:
    python3 GenAI/gpt_from_scratch_chatbot/chat.py
"""

from train_chatbot import chat_once, load_checkpoint


def main():
    model, tokenizer = load_checkpoint()

    print("Tiny chatbot loaded. Type 'exit' to stop.")
    while True:
        user_message = input("\nuser: ").strip()
        if user_message.lower() in {"exit", "quit"}:
            break

        answer = chat_once(model, tokenizer, user_message)
        print(f"assistant: {answer}")


if __name__ == "__main__":
    main()
