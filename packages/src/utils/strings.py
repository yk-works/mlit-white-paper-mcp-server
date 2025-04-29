def ja_tokens(tokenizer, text: str) -> str:
    return " ".join(t.text for t in tokenizer.tokenize(text))
