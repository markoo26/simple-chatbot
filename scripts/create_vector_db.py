import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "krk-parking-chatbot"
DIMENSION = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
SIMILARITY_THRESHOLD = 0.75  # Threshold for semantic similarity (0-1, higher = more similar)
MAX_CHUNK_SENTENCES = 10  # Maximum sentences per chunk

# File paths
MARKDOWN_FILES = [
    "documents/general_information.md",
    "documents/parking_details.md",
    "documents/location.md",
    "documents/booking_process.md"
]


class SemanticChunker:
    """
    Chunks text based on semantic similarity between sentences.
    Groups sentences together until similarity drops below threshold.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.75):
        """
        Initialize the semantic chunker.

        Args:
            model_name: Name of the sentence transformer model
            similarity_threshold: Cosine similarity threshold for chunking (0-1)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def chunk_text(self, text: str, source_file: str) -> List[Dict]:
        """
        Chunk text based on semantic similarity between sentences.

        Args:
            text: Input text to chunk
            source_file: Name of the source file

        Returns:
            List of dictionaries containing chunks and metadata
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        print(f"\n📄 Processing {source_file}: {len(sentences)} sentences found")

        if not sentences:
            return []

        # Generate embeddings for all sentences
        print("   Generating sentence embeddings...")
        sentence_embeddings = self.model.encode(sentences)

        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embeddings = [sentence_embeddings[0]]

        for i in range(1, len(sentences)):
            # Calculate average embedding of current chunk
            avg_chunk_embedding = np.mean(current_chunk_embeddings, axis=0)

            # Calculate similarity with next sentence
            similarity = self.cosine_similarity(avg_chunk_embedding, sentence_embeddings[i])

            # Check if we should continue the current chunk or start a new one
            should_continue_chunk = (
                    similarity >= self.similarity_threshold and
                    len(current_chunk_sentences) < MAX_CHUNK_SENTENCES
            )

            if should_continue_chunk:
                # Add to current chunk
                current_chunk_sentences.append(sentences[i])
                current_chunk_embeddings.append(sentence_embeddings[i])
            else:
                # Save current chunk and start new one
                chunk_text = " ".join(current_chunk_sentences)
                chunk_embedding = np.mean(current_chunk_embeddings, axis=0)

                chunks.append({
                    "text": chunk_text,
                    "embedding": chunk_embedding.tolist(),
                    "source": source_file,
                    "num_sentences": len(current_chunk_sentences),
                    "chunk_id": len(chunks)
                })

                # Start new chunk
                current_chunk_sentences = [sentences[i]]
                current_chunk_embeddings = [sentence_embeddings[i]]

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_embedding = np.mean(current_chunk_embeddings, axis=0)

            chunks.append({
                "text": chunk_text,
                "embedding": chunk_embedding.tolist(),
                "source": source_file,
                "num_sentences": len(current_chunk_sentences),
                "chunk_id": len(chunks)
            })

        print(f"   ✓ Created {len(chunks)} semantic chunks")

        return chunks


def initialize_pinecone():
    """Initialize Pinecone client and create/connect to index."""

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables. Please check your .env file.")

    print("\n🔌 Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    existing_indexes = pc.list_indexes().names()

    if INDEX_NAME in existing_indexes:
        print(f"   Index '{INDEX_NAME}' already exists. Deleting old index...")
        pc.delete_index(INDEX_NAME)
        print("   ✓ Old index deleted")

    # Create new index
    print(f"   Creating new index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print("   ✓ Index created successfully")

    # Connect to index
    index = pc.Index(INDEX_NAME)

    return index


def read_markdown_file(filepath: str) -> str:
    """Read and return content of markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"   ⚠️  Warning: File {filepath} not found. Skipping...")
        return ""


def upload_to_pinecone(index, chunks: List[Dict]):
    """Upload chunks to Pinecone index."""

    if not chunks:
        print("   No chunks to upload.")
        return

    print(f"\n📤 Uploading {len(chunks)} chunks to Pinecone...")

    # Prepare vectors for upsert
    vectors = []
    for i, chunk in enumerate(chunks):
        vector_id = f"{chunk['source']}_chunk_{chunk['chunk_id']}"

        vectors.append({
            "id": vector_id,
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "num_sentences": chunk["num_sentences"]
            }
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"   Uploaded batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")

    print("   ✓ All chunks uploaded successfully")

"""Main function to process documents and create vector database."""

print("=" * 70)
print("KRK PARKING CHATBOT - VECTOR DATABASE CREATION")
print("=" * 70)

# Initialize semantic chunker
chunker = SemanticChunker(
    model_name='all-MiniLM-L6-v2',
    similarity_threshold=SIMILARITY_THRESHOLD
)

# Process all markdown files
all_chunks = []

print(f"\n📚 Processing {len(MARKDOWN_FILES)} markdown files...")

for filepath in MARKDOWN_FILES:
    content = read_markdown_file(filepath)

    if content:
        chunks = chunker.chunk_text(content, filepath)
        all_chunks.extend(chunks)
    else:
        print(f"   ⚠️  Skipping {filepath} (empty or not found)")

# Summary statistics
print("\n" + "=" * 70)
print("CHUNKING SUMMARY:")
print("=" * 70)

total_chunks = len(all_chunks)
total_sentences = sum(chunk['num_sentences'] for chunk in all_chunks)

print(f"Total chunks created: {total_chunks}")
print(f"Total sentences: {total_sentences}")
print(f"Average sentences per chunk: {total_sentences / total_chunks:.2f}")
print(f"Similarity threshold used: {SIMILARITY_THRESHOLD}")

# Show breakdown by file
print("\nBreakdown by file:")
for filepath in MARKDOWN_FILES:
    file_chunks = [c for c in all_chunks if c['source'] == filepath]
    if file_chunks:
        file_sentences = sum(c['num_sentences'] for c in file_chunks)
        print(f"   {filepath}: {len(file_chunks)} chunks, {file_sentences} sentences")

# Initialize Pinecone and upload
index = initialize_pinecone()
upload_to_pinecone(index, all_chunks)

# Get index stats
print("\n" + "=" * 70)
print("PINECONE INDEX STATISTICS:")
print("=" * 70)

stats = index.describe_index_stats()
print(f"Index name: {INDEX_NAME}")
print(f"Total vectors: {stats['total_vector_count']}")
print(f"Dimension: {stats['dimension']}")

print("\n" + "=" * 70)
print("✅ VECTOR DATABASE CREATED SUCCESSFULLY!")
print("=" * 70)

# Example query
print("\n💡 Example: Testing similarity search...")
query_text = "How much does parking cost per day?"
query_embedding = chunker.model.encode([query_text])[0].tolist()

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

print(f"\nQuery: '{query_text}'")
print("\nTop 3 most similar chunks:")
for i, match in enumerate(results['matches'], 1):
    print(f"\n{i}. Score: {match['score']:.4f}")
    print(f"   Source: {match['metadata']['source']}")
    print(f"   Text preview: {match['metadata']['text'][:150]}...")

print("\n" + "=" * 70)
print("You can now use this index in your RAG chatbot!")
print("=" * 70)
