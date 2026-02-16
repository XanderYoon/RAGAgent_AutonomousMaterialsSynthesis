def chunk_text(text, max_tokens=None, tokenizer=None):
    """Split text into token-limited chunks without overlap.

    Args:
        text: Raw text to split into chunks.
        max_tokens: Maximum tokens allowed per chunk.
        tokenizer: Tokenizer providing an ``encode`` method.

    Returns:
        Ordered list of text chunks within the token budget.

    Raises:
        ValueError: If ``max_tokens`` or ``tokenizer`` is not provided.
    """
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
    """Split text into token-limited chunks with word overlap.

    Args:
        text: Raw text to split into chunks.
        max_tokens: Maximum tokens allowed per chunk.
        tokenizer: Tokenizer providing an ``encode`` method.
        overlap: Number of trailing words to carry into the next chunk.

    Returns:
        Ordered list of text chunks with overlap between adjacent chunks.

    Raises:
        ValueError: If ``max_tokens``, ``tokenizer``, or ``overlap`` is missing.
    """
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
