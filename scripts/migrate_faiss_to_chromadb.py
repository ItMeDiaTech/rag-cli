"""Migration script to convert FAISS indexes to ChromaDB.

This script migrates existing FAISS vector stores to ChromaDB format,
preserving all metadata and document content.

Usage:
    python scripts/migrate_faiss_to_chromadb.py [--faiss-index PATH] [--faiss-metadata PATH] [--chroma-dir PATH]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import numpy as np
    import faiss
    import chromadb
    # ChromaDB 1.0+ uses different Settings API
    try:
        from chromadb.config import Settings
    except ImportError:
        # For newer versions, Settings might be in different location or not needed
        Settings = None
except ImportError as e:
    print(f"Error: Required dependencies not installed: {e}")
    print("Install with: pip install numpy faiss-cpu chromadb>=1.3.0")
    sys.exit(1)


def load_faiss_index(index_path: str, metadata_path: str):
    """Load FAISS index and metadata.

    Args:
        index_path: Path to FAISS index file
        metadata_path: Path to metadata JSON file

    Returns:
        Tuple of (faiss_index, metadata_list, vectors_array)
    """
    print(f"Loading FAISS index from {index_path}...")

    # Load FAISS index
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors")

    # Load metadata
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} metadata entries")

    # Extract vectors from FAISS
    print("Extracting vectors from FAISS index...")
    vectors = []
    for i in range(index.ntotal):
        vector = index.reconstruct(i)
        vectors.append(vector)

    vectors_array = np.array(vectors, dtype=np.float32)
    print(f"Extracted {len(vectors)} vectors with dimension {vectors_array.shape[1]}")

    return index, metadata, vectors_array


def migrate_to_chromadb(
    vectors: np.ndarray,
    metadata: List[Dict[str, Any]],
    chroma_dir: str,
    collection_name: str = "rag_documents"
):
    """Migrate vectors and metadata to ChromaDB.

    Args:
        vectors: NumPy array of vectors
        metadata: List of metadata dictionaries
        chroma_dir: Directory for ChromaDB persistence
        collection_name: Name of the ChromaDB collection
    """
    print(f"\nMigrating to ChromaDB at {chroma_dir}...")

    # Create directory if needed
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)

    # Initialize ChromaDB client
    # ChromaDB 1.3.0+ API: Settings are passed differently or as kwargs
    try:
        if Settings:
            # Older API with Settings object
            client = chromadb.PersistentClient(
                path=chroma_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        else:
            # Newer API: settings passed as kwargs or use defaults
            client = chromadb.PersistentClient(
                path=chroma_dir,
                anonymized_telemetry=False
            )
    except Exception as e:
        # Fallback: try with minimal configuration
        client = chromadb.PersistentClient(path=chroma_dir)

    # Create or get collection
    try:
        # Try to delete existing collection if it exists
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata={"dimension": vectors.shape[1], "hnsw:space": "l2"}
        )
        print(f"Created new collection '{collection_name}'")

    except Exception as e:
        print(f"Error creating collection: {e}")
        raise

    # Prepare data for ChromaDB
    ids = []
    documents = []
    embeddings = []
    chroma_metadata = []

    for i, meta in enumerate(metadata):
        # Generate ID (use existing ID if available)
        vector_id = meta.get('id', f"vec_{i:08d}")
        ids.append(vector_id)

        # Get text content
        text = meta.get('text', '')
        documents.append(text)

        # Get embedding
        embeddings.append(vectors[i].tolist())

        # Prepare metadata for ChromaDB (flatten nested structures)
        chroma_meta = {
            "text": text,
            "source": meta.get('source', 'unknown'),
            "timestamp": meta.get('timestamp', datetime.now().isoformat())
        }

        # Add custom metadata
        if 'metadata' in meta and isinstance(meta['metadata'], dict):
            for key, value in meta['metadata'].items():
                if isinstance(value, (dict, list)):
                    chroma_meta[f"custom_{key}"] = json.dumps(value)
                else:
                    chroma_meta[f"custom_{key}"] = str(value)

        chroma_metadata.append(chroma_meta)

    # Batch add to ChromaDB (in chunks to avoid memory issues)
    batch_size = 1000
    total_batches = (len(ids) + batch_size - 1) // batch_size

    print(f"\nAdding {len(ids)} vectors in {total_batches} batches...")

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(ids))

        collection.add(
            ids=ids[start_idx:end_idx],
            embeddings=embeddings[start_idx:end_idx],
            documents=documents[start_idx:end_idx],
            metadatas=chroma_metadata[start_idx:end_idx]
        )

        progress = (batch_num + 1) / total_batches * 100
        print(f"Progress: {progress:.1f}% ({end_idx}/{len(ids)} vectors)")

    final_count = collection.count()
    print(f"\nMigration complete! ChromaDB collection contains {final_count} vectors")

    return collection


def verify_migration(collection, original_metadata: List[Dict], vectors: np.ndarray):
    """Verify that migration was successful.

    Args:
        collection: ChromaDB collection
        original_metadata: Original metadata list
        vectors: Original vectors array
    """
    print("\nVerifying migration...")

    # Check count
    chroma_count = collection.count()
    original_count = len(original_metadata)

    if chroma_count != original_count:
        print(f"WARNING: Count mismatch! ChromaDB: {chroma_count}, Original: {original_count}")
        return False

    print(f"Count verified: {chroma_count} vectors")

    # Sample check: retrieve a few vectors and verify they match
    sample_size = min(5, chroma_count)
    print(f"Sampling {sample_size} vectors for verification...")

    for i in range(0, chroma_count, max(1, chroma_count // sample_size)):
        # Get original
        original_vec = vectors[i]
        original_meta = original_metadata[i]

        # Query ChromaDB with the same vector
        results = collection.query(
            query_embeddings=[original_vec.tolist()],
            n_results=1
        )

        if results and results['ids'] and len(results['ids'][0]) > 0:
            # Check if the closest match is very close (distance should be near 0)
            distance = results['distances'][0][0]
            if distance > 0.01:  # Allow small floating point errors
                print(f"WARNING: Vector {i} distance too large: {distance}")
                return False

            print(f"Vector {i}: Verified (distance: {distance:.6f})")
        else:
            print(f"WARNING: Could not find vector {i} in ChromaDB")
            return False

    print("\nVerification successful!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate FAISS index to ChromaDB")
    parser.add_argument(
        "--faiss-index",
        default="data/vectors/vectors.index",
        help="Path to FAISS index file"
    )
    parser.add_argument(
        "--faiss-metadata",
        default="data/vectors/metadata.json",
        help="Path to FAISS metadata file"
    )
    parser.add_argument(
        "--chroma-dir",
        default="data/vectors/chroma_db",
        help="Directory for ChromaDB persistence"
    )
    parser.add_argument(
        "--collection",
        default="rag_documents",
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration after completion"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FAISS to ChromaDB Migration Script")
    print("=" * 60)
    print(f"FAISS index:      {args.faiss_index}")
    print(f"FAISS metadata:   {args.faiss_metadata}")
    print(f"ChromaDB dir:     {args.chroma_dir}")
    print(f"Collection name:  {args.collection}")
    print("=" * 60)

    try:
        # Load FAISS data
        faiss_index, metadata, vectors = load_faiss_index(
            args.faiss_index,
            args.faiss_metadata
        )

        # Migrate to ChromaDB
        collection = migrate_to_chromadb(
            vectors,
            metadata,
            args.chroma_dir,
            args.collection
        )

        # Verify if requested
        if args.verify:
            success = verify_migration(collection, metadata, vectors)
            if not success:
                print("\nWARNING: Verification found issues!")
                sys.exit(1)

        print("\n" + "=" * 60)
        print("Migration completed successfully!")
        print("=" * 60)
        print(f"\nChromaDB collection at: {args.chroma_dir}")
        print(f"Collection name: {args.collection}")
        print(f"Total vectors: {collection.count()}")
        print("\nYou can now use the new ChromaDB vector store in RAG-CLI.")

    except Exception as e:
        print(f"\nError during migration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
