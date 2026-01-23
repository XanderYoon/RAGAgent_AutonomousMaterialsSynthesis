def estimate_embedding_cost(token_count, model="text-embedding-3-small"):
    price_per_million = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10,
    }.get(model, 0.02)

    return (token_count / 1_000_000) * price_per_million
