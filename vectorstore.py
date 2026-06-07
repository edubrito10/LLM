COLLECTION_NAME = "lupa_docs"


def build_vectorstore(chunks: list[str], embeddings):
    """
    Cria uma coleção ChromaDB em memória e indexa todos os chunks.

    Args:
        chunks:     lista de strings (chunks do documento)
        embeddings: objeto de embeddings (OllamaEmbeddings)

    Returns:
        Coleção ChromaDB pronta para consulta.
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.Client(Settings(anonymized_telemetry=False))

    # Remove coleção anterior se existir (novo documento)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    # Indexar em batches para evitar sobrecarga
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        vectors = embeddings.embed_documents(batch)
        collection.add(embeddings=vectors, documents=batch, ids=ids)

    return collection


def retrieve_context(query: str, collection, embeddings, k: int = 4) -> list[str]:
    """
    Recupera os k chunks mais relevantes para uma dada query.

    Args:
        query:      pergunta do utilizador
        collection: coleção ChromaDB
        embeddings: objeto de embeddings
        k:          número de chunks a recuperar

    Returns:
        Lista de strings com os chunks mais relevantes.
    """
    query_vector = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=k)
    return results["documents"][0] if results["documents"] else []
