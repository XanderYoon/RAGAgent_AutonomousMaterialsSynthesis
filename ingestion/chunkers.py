def chunk_text(text, max_tokens=None, tokenizer=None):
    if max_tokens is None:
        raise ValueError("max_tokens must be provided")
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        test_chunk = " ".join(current)
        if len(tokenizer.encode(test_chunk)) > max_tokens:
            current.pop()
            chunks.append(" ".join(current))
            current = [word]

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_text2(text, max_tokens=None, tokenizer=None, overlap=None):
    if max_tokens is None:
        raise ValueError("max_tokens must be provided")
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")
    if overlap is None:
        raise ValueError("overlap must be provided")

    words = text.split()
    chunks = []
    current = []
    i = 0

    while i < len(words):
        word = words[i]
        current.append(word)
        test_chunk = " ".join(current)

        if len(tokenizer.encode(test_chunk)) > max_tokens:
            current.pop()
            chunks.append(" ".join(current))
            overlap_start = max(len(current) - overlap, 0)
            current = current[overlap_start:] + [word]

        i += 1

    if current:
        chunks.append(" ".join(current))

    return chunks
