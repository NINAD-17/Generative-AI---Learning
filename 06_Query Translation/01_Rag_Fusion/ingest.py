def ingest_pdf_to_qdrant(pdf_path, collection_name, embedding):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore

    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    vector_store = QdrantVectorStore.from_documents(
        documents=[],
        url="http://localhost:6333",
        collection_name=collection_name,
        embedding=embedding
    )
    vector_store.add_documents(documents=chunks)
    print("Ingestion complete.")
