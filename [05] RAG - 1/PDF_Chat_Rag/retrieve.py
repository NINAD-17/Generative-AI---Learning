def retrieve_relevant_chunks(user_query, embedding, collection_name):
    from langchain_qdrant import QdrantVectorStore

    vector_store = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name=collection_name,
        embedding=embedding
    )

    results = vector_store.similarity_search(user_query, k=3)

    final_result = []
    for result in results:
        final_result.append({
            "content": result.page_content if hasattr(result, 'page_content') else "",
            "page_num": result.metadata.get("page", ""),
            "total_pages": result.metadata.get("total_pages", "")
        })
    
    return final_result