# RAG Pipeline with LangChain and Qdrant
from pathlib import Path
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv() # load environment variables from .env file
api_key = os.getenv("GOOGLE_API_KEY")

# STEP 1: LOADER
pdf_path = Path(__file__).parent / "data/Reach - SMA.pdf"

loader = PyPDFLoader(file_path=pdf_path) # loader is an instance of pyPDFLoader tied to selected PDF
docs = loader.load() # creates a list of document (page of PDF) objects - pages of the PDF

# print(docs)
# print(docs[1])

# STEP 2: SPLITTER
# initialize text splitter with recursive character text splitter - it don't loose the meaning of the text while chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, # size of each chunk in characters - apporx 1000 characters
    chunk_overlap = 200 
)

# split each pages of pdf into chunk of 1000 characters
split_docs = text_splitter.split_documents(documents=docs)

# print(split_docs)
print("Lenght of documents (pages):", len(docs))
print("Lenght of documents after chunking:", len(split_docs))

# STEP 3: EMBEDDING
# initialize embedding model - it will convert the text into vector representation
embedding = GoogleGenerativeAIEmbeddings( # this is our embedder or embedding model
    model="models/text-embedding-004",
    google_api_key=api_key,
)

# ### TEST -> EMBEDDING MODEL
# # Test input for embedding
# test_text = ["hello world"]

# # Generate embeddings
# try:
#     embedding_vector = embedding.embed_documents(test_text)
#     print("Embedding generated successfully!")
#     print("Embedding shape:", len(embedding_vector[0])) # dimensions
#     print("First 5 values:", embedding_vector[0][:5])
# except Exception as e:
#     print("Error generating embedding:", e)
# ### TEST END

# STEP 4: VECTOR STORE
vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedding
)

vector_store.add_documents(documents=split_docs)
print("Ingestion Done!")

# STEP 5: RETRIEVAL
retriever = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name = "learning_langchain",
    embedding = embedding
)

relevant_chunks = retriever.similarity_search(
    query = "Reach Examples"
)

print("Relevant Chunks: ", relevant_chunks)
