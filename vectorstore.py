COLLECTION_NAME = "lupa_docs"


def build_vectorstore(chunks: list[str], embeddings):
    """
    Cria uma base de dados na memória RAM. Tem um mecanismo de segurança 
    que apaga coleções antigas para não misturar PDFs diferentes. 
    O segredo aqui é o 'batch_size': os chunks são processados de 50 em 50
    para não esgotar a memória da placa gráfica (VRAM) de uma só vez.
    """
    import chromadb
    from chromadb.config import Settings

    # Arranca o ChromaDB em memória (sem gravar lixo no disco rígido)
    client = chromadb.Client(Settings(anonymized_telemetry=False))

    # Remove coleção anterior se existir (preparação para um novo documento)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    # Processamento em lotes (Batches) para eficiência e gestão de memória
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"chunk_{i + j}" for j in range(len(batch))] # Cria IDs únicos: chunk_0, chunk_1...
        
        # Converte o lote de texto em vetores matemáticos
        vectors = embeddings.embed_documents(batch)
        
        # Guarda os vetores, o texto original e os IDs na base de dados
        collection.add(embeddings=vectors, documents=batch, ids=ids)

    return collection


def retrieve_context(query: str, collection, embeddings, k: int = 4) -> list[str]:
    """
    Transforma a pergunta do utilizador num vetor e procura no ChromaDB
    os 'k' chunks matematicamente mais próximos (Similaridade de Cosseno).
    Não pesquisa palavras exatas, pesquisa o 'significado' da pergunta.
    """
    # 1. Converte a pergunta do utilizador para a mesma "língua matemática" (vetor)
    query_vector = embeddings.embed_query(query)
    
    # 2. Pede ao ChromaDB os k resultados mais parecidos com o vetor da pergunta
    results = collection.query(query_embeddings=[query_vector], n_results=k)
    
    # 3. Devolve apenas a lista de textos encontrados (se não houver, devolve vazio)
    return results["documents"][0] if results["documents"] else []