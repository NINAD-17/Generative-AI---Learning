import re
import random
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

def ingest_pdf_to_qdrant(pdf_path, collection_name, embedding):
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


def should_ingest(pdf_path, registry_file="ingested_pdfs.txt"):
    registry_file = Path(__file__).parent / registry_file
    registry_file.touch(exist_ok=True)

    pdf_name = pdf_path.name
    existing_entries = registry_file.read_text().splitlines() # Reads the entire contents of the registry file and splits it into a list of lines.

    for line in existing_entries:
        if pdf_name in line:
            collection = line.split(":")[-1].strip()
            print(f"🟡 Already ingested '{pdf_name}' under collection: {collection}")
            return False, collection  # Return existing collection name

    # If not found, create a new unique collection name
    base_name = pdf_path.stem
    clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', base_name)
    words = clean_name.split('_')[:3]
    short_name = '_'.join(words)
    random_suffix = str(random.randint(1000, 9999))
    collection_name = f"{short_name}_{random_suffix}"

    # Write to registry
    with open(registry_file, "a") as f:
        f.write(f"{pdf_name}:{collection_name}\n")

    print(f"🟢 New file detected. Ingesting '{pdf_name}' as collection: {collection_name}")
    return True, collection_name
