from langchain_qdrant import QdrantVectorStore

# Retrieve relevant chunks from Qdrant for each query
def retrieve_relevant_chunks(user_query, max_chunks, embedding, collection_name, score_threshold=0.7):
    if not user_query or not user_query.strip():
        raise ValueError("❌ Cannot embed an empty query for retrieval.")
    
    vector_store = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name=collection_name,
        embedding=embedding
    )

    # Perform similarity search with the specified number of chunks
    results = vector_store.similarity_search_with_score(user_query, k=max_chunks)

    final_result = []

    # Prepare the final result with content and metadata
    for result, score in results:
        if score >= score_threshold:
            final_result.append({
                "content": result.page_content if hasattr(result, 'page_content') else "",
                "page_num": result.metadata.get("page", ""),
                "total_pages": result.metadata.get("total_pages", ""),
                "score": score
            })

    # print(f"\nQUERY: {user_query} -------- CHUNKS: {final_result}\n\n")
    
    # Return the final result containing relevant chunks
    return final_result