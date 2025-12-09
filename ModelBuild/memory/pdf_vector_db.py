# memory/pdf_vector_db.py
import os
import json
import chromadb
from chromadb.config import Settings
from typing import Optional, List, Dict, Any, Union
import PyPDF2
import fitz  # PyMuPDF
from pathlib import Path
import hashlib
import logging
from datetime import datetime
import uuid
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")


class ContentType(Enum):
    PDF_PAPER = "pdf_paper"
    AGENT_MEMORY = "agent_memory"
    EXTRACTED_PARAMETER = "extracted_parameter"
    CODE_IDEA = "code_idea"
    VIVARIUM_CODE = "vivarium_code"
    CRITIQUE_SUMMARY = "critique_summary"


class MultiAgentSharedDatabase:
    """
    A unified database system for storing PDF papers, agent memories, and extracted parameters.
    Supports multi-agent collaboration with shared memory and parameter extraction.
    """

    def __init__(self,
                 db_dir: str,
                 collection_name: str = "multi_agent_memory",
                 embedding_model: str = "text-embedding-3-large",
                 chunk_size: int = 2000,
                 chunk_overlap: int = 200):
        """
        Initialize the Multi-Agent Shared Database.

        Args:
            db_dir: Directory to store the ChromaDB database
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Size of text chunks for PDF processing
            chunk_overlap: Overlap between chunks
        """
        if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            self.client = None
            logger.warning("OpenAI client not initialized - check OPENAI_API_KEY")

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB with error handling
        try:
            self.chroma_client = chromadb.PersistentClient(path=db_dir)
            self.collection = self.chroma_client.get_or_create_collection(collection_name)
            logger.info(f"ChromaDB initialized at {db_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None

        # Track processed files
        self.processed_files = self._load_processed_files()

    def _load_processed_files(self) -> Dict[str, str]:
        """Load list of already processed files to avoid reprocessing."""
        if not self.collection:
            return {}

        try:
            results = self.collection.get(
                where={"content_type": ContentType.PDF_PAPER.value},
                include=["metadatas"]
            )
            processed = {}
            for meta in results.get("metadatas", []):
                if "file_path" in meta and "file_hash" in meta:
                    processed[meta["file_path"]] = meta["file_hash"]
            return processed
        except Exception as e:
            logger.warning(f"Could not load processed files: {e}")
            return {}

    def _get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file for change detection."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash for {file_path}: {e}")
            return ""

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF with fallback to PyPDF2."""
        text = ""

        # Try PyMuPDF first
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf_path}: {e}")

        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            if text.strip():
                return text
        except Exception as e:
            logger.error(f"PyPDF2 also failed for {pdf_path}: {e}")

        return ""

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap

            if start >= len(text):
                break

        return [chunk for chunk in chunks if chunk.strip()]

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API with error handling."""
        if not self.client:
            logger.warning("OpenAI client not available for embeddings")
            return []

        try:
            # Truncate text if too long
            text = text[:8000]  # OpenAI token limits
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    # ==================== PDF Paper Methods ====================

    def add_pdf_paper(self, pdf_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a PDF paper to the database with comprehensive error handling."""
        if not self.collection:
            logger.error("Database collection not available")
            return False

        try:
            pdf_path = str(Path(pdf_path).resolve())

            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False

            # Check if already processed
            current_hash = self._get_file_hash(pdf_path)
            if not current_hash:
                logger.error(f"Could not generate hash for {pdf_path}")
                return False

            if pdf_path in self.processed_files and self.processed_files[pdf_path] == current_hash:
                logger.info(f"PDF already processed: {pdf_path}")
                return True

            # Extract and chunk text
            text = self._extract_text_from_pdf(pdf_path)
            if not text.strip():
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                return False

            chunks = self._chunk_text(text, self.chunk_size, self.chunk_overlap)
            logger.info(f"Created {len(chunks)} chunks from {pdf_path}")

            # Prepare metadata
            base_metadata = {
                "content_type": ContentType.PDF_PAPER.value,
                "file_path": pdf_path,
                "file_name": os.path.basename(pdf_path),
                "file_hash": current_hash,
                "total_chunks": len(chunks),
                "processed_at": datetime.now().isoformat()
            }

            if metadata:
                base_metadata.update(metadata)

            # Process chunks
            embeddings, documents, metadatas, ids = [], [], [], []

            for i, chunk in enumerate(chunks):
                embedding = self.get_embedding(chunk)
                if not embedding:
                    logger.warning(f"Could not generate embedding for chunk {i}")
                    continue

                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_length": len(chunk)
                })

                chunk_id = f"pdf_{hashlib.md5(pdf_path.encode()).hexdigest()}_{i}"

                embeddings.append(embedding)
                documents.append(chunk)
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)

            # Add to database
            if embeddings:
                self.collection.upsert(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                self.processed_files[pdf_path] = current_hash
                logger.info(f"Successfully added {len(embeddings)} chunks from {pdf_path}")
                return True
            else:
                logger.warning(f"No embeddings generated for {pdf_path}")
                return False

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return False

    def add_pdf_directory(self, folder_path: str) -> None:
        """Add all PDF files from a folder to the vector database."""
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            logger.error(f"Folder not found: {folder_path}")
            return

        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in: {folder_path}")
            return

        success_count = 0
        for pdf_file in pdf_files:
            if self.add_pdf_paper(str(pdf_file)):
                success_count += 1

        logger.info(f"Successfully processed {success_count}/{len(pdf_files)} PDFs from {folder_path}")

    # ==================== Agent Memory Methods ====================

    def store_agent_memory(self,
                           agent_name: str,
                           iteration: int,
                           content: str,
                           content_type: ContentType,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store agent memory/output in the database."""
        if not self.collection:
            logger.error("Database collection not available")
            return ""

        try:
            # Generate unique ID
            memory_id = f"{agent_name}_{iteration}_{content_type.value}_{uuid.uuid4().hex[:8]}"

            # Create embedding
            embedding = self.get_embedding(content)
            if not embedding:
                logger.error(f"Could not generate embedding for {agent_name} memory")
                return ""

            # Prepare metadata
            memory_metadata = {
                "content_type": content_type.value,
                "agent_name": agent_name,
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "content_length": len(content)
            }

            if metadata:
                memory_metadata.update(metadata)

            # Store in database
            self.collection.upsert(
                embeddings=[embedding],
                documents=[content],
                metadatas=[memory_metadata],
                ids=[memory_id]
            )

            logger.info(f"Stored {content_type.value} from {agent_name} (iteration {iteration})")
            return memory_id

        except Exception as e:
            logger.error(f"Error storing agent memory: {e}")
            return ""

    def store_extracted_parameter(self,
                                  agent_name: str,
                                  iteration: int,
                                  parameter_name: str,
                                  parameter_value: Union[str, float, int],
                                  parameter_type: str,
                                  source_paper: Optional[str] = None,
                                  confidence: Optional[float] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store an extracted parameter with structured metadata."""
        # Create structured content
        content_text = f"Parameter: {parameter_name}\nValue: {parameter_value}\nType: {parameter_type}"
        if source_paper:
            content_text += f"\nSource: {source_paper}"
        if confidence:
            content_text += f"\nConfidence: {confidence}"

        # Additional metadata
        param_metadata = {
            "parameter_name": parameter_name,
            "parameter_value": str(parameter_value),
            "parameter_type": parameter_type,
            "source_paper": source_paper,
            "confidence": confidence
        }

        if metadata:
            param_metadata.update(metadata)

        return self.store_agent_memory(
            agent_name=agent_name,
            iteration=iteration,
            content=content_text,
            content_type=ContentType.EXTRACTED_PARAMETER,
            metadata=param_metadata
        )

    # ==================== Search and Retrieval Methods ====================

    def search_content(self,
                       query: str,
                       top_k: int = 10,
                       content_types: Optional[List[ContentType]] = None,
                       agent_filter: Optional[str] = None,
                       iteration_filter: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search across all content types with optional filtering."""
        if not self.collection:
            logger.error("Database collection not available")
            return []

        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                logger.error("Could not generate embedding for query")
                return []

            # Build where clause
            where_clause = {}
            if content_types:
                where_clause["content_type"] = {"$in": [ct.value for ct in content_types]}
            if agent_filter:
                where_clause["agent_name"] = agent_filter
            if iteration_filter is not None:
                where_clause["iteration"] = iteration_filter

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, meta, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                )):
                    formatted_results.append({
                        "content": doc,
                        "metadata": meta,
                        "similarity_score": 1 - distance,
                        "rank": i + 1
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []

    def search_parameters(self,
                          parameter_name: Optional[str] = None,
                          parameter_type: Optional[str] = None,
                          source_paper: Optional[str] = None,
                          min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for extracted parameters with specific filters."""
        if not self.collection:
            return []

        where_clause = {"content_type": ContentType.EXTRACTED_PARAMETER.value}

        if parameter_name:
            where_clause["parameter_name"] = parameter_name
        if parameter_type:
            where_clause["parameter_type"] = parameter_type
        if source_paper:
            where_clause["source_paper"] = source_paper
        if min_confidence:
            where_clause["confidence"] = {"$gte": min_confidence}

        try:
            results = self.collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )

            formatted_results = []
            if results['documents']:
                for doc, meta in zip(results['documents'], results['metadatas']):
                    formatted_results.append({
                        "content": doc,
                        "metadata": meta
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching parameters: {e}")
            return []

    def get_agent_history(self,
                          agent_name: str,
                          content_type: Optional[ContentType] = None,
                          max_iterations: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the history of an agent's activities."""
        if not self.collection:
            return []

        where_clause = {"agent_name": agent_name}

        if content_type:
            where_clause["content_type"] = content_type.value

        try:
            results = self.collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )

            history = []
            if results['documents']:
                for doc, meta in zip(results['documents'], results['metadatas']):
                    history.append({
                        "content": doc,
                        "metadata": meta
                    })

            # Sort by iteration and timestamp
            history.sort(key=lambda x: (x['metadata'].get('iteration', 0), x['metadata'].get('timestamp', '')))

            if max_iterations:
                history = history[-max_iterations:]

            return history

        except Exception as e:
            logger.error(f"Error getting agent history: {e}")
            return []

    # ==================== Database Management ====================

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        if not self.collection:
            return {"error": "Database collection not available"}

        try:
            results = self.collection.get(include=["metadatas"])

            stats = {
                "total_entries": 0,
                "content_types": {},
                "agents": {},
                "parameters": {
                    "total": 0,
                    "by_type": {},
                    "by_paper": {}
                },
                "papers": {
                    "total": 0,
                    "total_chunks": 0
                }
            }

            for meta in results.get("metadatas", []):
                stats["total_entries"] += 1

                content_type = meta.get("content_type", "Unknown")
                stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1

                agent_name = meta.get("agent_name")
                if agent_name:
                    stats["agents"][agent_name] = stats["agents"].get(agent_name, 0) + 1

                if content_type == ContentType.EXTRACTED_PARAMETER.value:
                    stats["parameters"]["total"] += 1

                    param_type = meta.get("parameter_type", "Unknown")
                    stats["parameters"]["by_type"][param_type] = stats["parameters"]["by_type"].get(param_type, 0) + 1

                    source_paper = meta.get("source_paper")
                    if source_paper:
                        stats["parameters"]["by_paper"][source_paper] = stats["parameters"]["by_paper"].get(
                            source_paper, 0) + 1

                elif content_type == ContentType.PDF_PAPER.value:
                    if meta.get("chunk_index") == 0:  # Count only first chunk of each paper
                        stats["papers"]["total"] += 1
                    stats["papers"]["total_chunks"] += 1

            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}

    def search_top_paper_texts(self, query: str, top_k: int = 3,
                               content_types: Optional[List[ContentType]] = None,
                               return_titles: bool = False) -> List[Dict[str, Any]]:
        """
        Search for top papers and return their full text content.

        Args:
            query: Search query
            top_k: Number of top papers to return
            content_types: Filter by content types
            return_titles: Whether to include paper titles in results

        Returns:
            List of papers with full text content
        """
        if not self.collection:
            logger.error("Database collection not available")
            return []

        try:
            # First get search results to identify top papers
            search_results = self.search_content(
                query=query,
                top_k=top_k * 3,  # Get more chunks to identify top papers
                content_types=content_types or [ContentType.PDF_PAPER]
            )

            if not search_results:
                return []

            # Group by paper and calculate relevance
            papers_data = {}
            for result in search_results:
                file_path = result['metadata'].get('file_path', '')
                file_name = result['metadata'].get('file_name', 'Unknown')

                if file_path not in papers_data:
                    papers_data[file_path] = {
                        'file_path': file_path,
                        'file_name': file_name,
                        'scores': [],
                        'chunks': [],
                        'metadata': result['metadata']
                    }

                papers_data[file_path]['scores'].append(result['similarity_score'])
                papers_data[file_path]['chunks'].append(result['content'])

            # Calculate average relevance for each paper
            for paper_data in papers_data.values():
                scores = paper_data['scores']
                paper_data['avg_relevance'] = sum(scores) / len(scores) if scores else 0.0

            # Sort by relevance and take top_k
            sorted_papers = sorted(
                papers_data.values(),
                key=lambda x: x['avg_relevance'],
                reverse=True
            )[:top_k]

            # Get full text for each paper
            result_papers = []
            for paper_data in sorted_papers:
                file_path = paper_data['file_path']

                # Try to get full text from file
                if file_path and os.path.exists(file_path):
                    full_text = self._extract_text_from_pdf(file_path)
                    if full_text:
                        content = full_text
                    else:
                        # Fallback to concatenated chunks
                        content = '\n\n'.join(paper_data['chunks'])
                else:
                    # Use concatenated chunks
                    content = '\n\n'.join(paper_data['chunks'])

                result_papers.append({
                    'content': content,
                    'metadata': paper_data['metadata'],
                    'similarity_score': paper_data['avg_relevance'],
                    'file_name': paper_data['file_name'],
                    'file_path': file_path
                })

            return result_papers

        except Exception as e:
            logger.error(f"Error searching top paper texts: {e}")
            return []

    def export_parameters(self, output_file: str) -> bool:
        """Export all extracted parameters to a JSON file."""
        try:
            parameters = self.search_parameters()

            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_parameters": len(parameters),
                "parameters": []
            }

            for param in parameters:
                meta = param["metadata"]
                export_data["parameters"].append({
                    "parameter_name": meta.get("parameter_name"),
                    "parameter_value": meta.get("parameter_value"),
                    "parameter_type": meta.get("parameter_type"),
                    "source_paper": meta.get("source_paper"),
                    "confidence": meta.get("confidence"),
                    "extracted_by": meta.get("agent_name"),
                    "iteration": meta.get("iteration"),
                    "timestamp": meta.get("timestamp")
                })

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Exported {len(parameters)} parameters to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error exporting parameters: {e}")
            return False


# Example usage for multi-agent system
if __name__ == "__main__":
    # Initialize the shared database
    db = MultiAgentSharedDatabase(
        db_dir="./multi_agent_db",
        collection_name="shared_memory"
    )

    # Add all papers from the target folder
    if os.path.exists("../parameter/my_papers"):
        db.add_pdf_directory("../parameter/my_papers")
    else:
        print("Paper directory not found")