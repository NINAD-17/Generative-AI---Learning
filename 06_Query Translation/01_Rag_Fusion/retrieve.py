import asyncio
from langchain_qdrant import QdrantVectorStore

# Retrieve relevant chunks from Qdrant for each query
def retrieve_relevant_chunks(user_query, max_chunks, embedding, collection_name):
    vector_store = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name=collection_name,
        embedding=embedding
    )

    # Perform similarity search with the specified number of chunks
    results = vector_store.similarity_search(user_query, k=max_chunks)

    final_result = []

    # Prepare the final result with content and metadata
    for result in results:
        final_result.append({
            "content": result.page_content if hasattr(result, 'page_content') else "",
            "page_num": result.metadata.get("page", ""),
            "total_pages": result.metadata.get("total_pages", "")
        })

    # print(f"\nQUERY: {user_query} -------- CHUNKS: {final_result}\n\n")
    
    # Return the final result containing relevant chunks
    return final_result

# Async wrapper for the synchronous retrieval function (retrieve_relevant_chunks)
async def async_retrieve_chunks(query, max_chunks, embedding, collection_name):
    # run synchronous 'retrieve_relevant_chunks' function in a thread
    result = await asyncio.to_thread(retrieve_relevant_chunks, query, max_chunks, embedding, collection_name)
    print(f"For Query -> {query} - {len(result)} chunks found")
    return result

# Parallel Processing (retrieve chunks from all the queries concurrently)
async def process_queries_parallely(queries, max_chunks, embedding, collection_name):
    # list of tasks to call 'async_retrieve_chunks'
    tasks = [
        async_retrieve_chunks(query, max_chunks, embedding, collection_name) 
        for query in queries 
    ]

    # run tasks concurrently
    results = await asyncio.gather(*tasks) # output list of list (array or array) [[], [], ...]
    return results

async def retrieve(generated_queries, max_chunks, embedding, collection_name):
    # Retrieve relevant chunks - output array of array
    retrieved_lists_of_chunks = await process_queries_parallely(generated_queries, max_chunks, embedding, collection_name)

    # Flatten results - add all the chunks in one list
    combined_chunks = [] # instead of for loops, you can do - [chunk for result in all_results for chunk in result]
    for chunk_list in retrieved_lists_of_chunks:
        for chunk in chunk_list:
            combined_chunks.append(chunk)
                
    # Remove duplicates
    unique_chunks = {}
    for chunk in combined_chunks:
        content = chunk.get("content", "") # get value from "page_content"
        if content and content not in unique_chunks:
            unique_chunks[content] = chunk # key is page_content and value is entire chunk

    final_chunks = []
    for chunk in unique_chunks.values():
        final_chunks.append(chunk)

    print(f"️🟩 Successfully retrieved {len(final_chunks)} chunks\n")
    return final_chunks