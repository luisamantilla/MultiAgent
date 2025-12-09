# tools/tools.py
"""
Advanced tools module for biological modeling with RAG integration.
Includes PubMed API, ArXiv access, PaperQA2 integration, and modern RAG pipeline.
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import re
import hashlib

# Scientific paper processing
try:
    import arxiv

    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

try:
    from Bio import Entrez

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# Enhanced document processing
try:
    from llama_parse import LlamaParse

    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False

try:
    import paperqa

    PAPERQA_AVAILABLE = True
except ImportError:
    PAPERQA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Enhanced paper metadata structure."""
    title: str
    authors: List[str]
    abstract: str
    doi: Optional[str] = None
    pubmed_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    url: Optional[str] = None
    confidence_score: float = 1.0
    source: str = "unknown"


@dataclass
class ProcessedChunk:
    """Processed document chunk with enhanced metadata."""
    content: str
    chunk_id: str
    source_paper: PaperMetadata
    chunk_index: int
    embedding: Optional[List[float]] = None
    processing_method: str = "standard"
    confidence: float = 1.0
    section: Optional[str] = None


class RAGPipeline:
    """
    State-of-the-art RAG pipeline for biological literature.
    Integrates multiple data sources and processing methods.
    """

    def __init__(self,
                 email: Optional[str] = None,
                 llamaparse_api_key: Optional[str] = None,
                 cache_dir: str = "./cache"):
        self.email = email
        self.llamaparse_api_key = llamaparse_api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.session: Optional[aiohttp.ClientSession] = None
        self._setup_entrez()
        self._setup_llamaparse()
        self._setup_paperqa()

        # Cache for processed papers
        self.paper_cache: Dict[str, PaperMetadata] = {}
        self.chunk_cache: Dict[str, List[ProcessedChunk]] = {}

    def _setup_entrez(self):
        """Setup Entrez for PubMed access."""
        if BIOPYTHON_AVAILABLE and self.email:
            Entrez.email = self.email
            logger.info("Entrez configured for PubMed access")

    def _setup_llamaparse(self):
        """Setup LlamaParse for advanced document processing."""
        if LLAMAPARSE_AVAILABLE and self.llamaparse_api_key:
            self.llamaparse = LlamaParse(api_key=self.llamaparse_api_key)
            logger.info("LlamaParse initialized")
        else:
            self.llamaparse = None

    def _setup_paperqa(self):
        """Setup PaperQA2 for advanced RAG."""
        if PAPERQA_AVAILABLE:
            try:
                self.paperqa = paperqa.Docs()
                logger.info("PaperQA2 initialized")
            except Exception as e:
                logger.warning(f"PaperQA2 initialization failed: {e}")
                self.paperqa = None
        else:
            self.paperqa = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    # STEP 1: INGEST - Get documents from multiple sources
    async def search_pubmed(self,
                            query: str,
                            max_results: int = 20,
                            filters: Optional[Dict[str, Any]] = None) -> List[PaperMetadata]:
        """Search PubMed with advanced filtering."""
        if not BIOPYTHON_AVAILABLE:
            logger.warning("BioPython not available for PubMed search")
            return []

        try:
            # Build search query with filters
            search_query = query
            if filters:
                if filters.get('date_range'):
                    search_query += f" AND {filters['date_range']}"
                if filters.get('species'):
                    search_query += f" AND {filters['species']}[MeSH Terms]"
                if filters.get('article_types'):
                    for article_type in filters['article_types']:
                        search_query += f" AND {article_type}[ptyp]"

            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=search_query,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()

            if not search_results['IdList']:
                return []

            # Fetch detailed information
            handle = Entrez.efetch(
                db="pubmed",
                id=search_results['IdList'],
                rettype="xml",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()

            papers = []
            for record in records['PubmedArticle']:
                try:
                    paper = self._parse_pubmed_record(record)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Error parsing PubMed record: {e}")
                    continue

            logger.info(f"Retrieved {len(papers)} papers from PubMed")
            return papers

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    def _parse_pubmed_record(self, record) -> PaperMetadata:
        """Parse PubMed XML record into structured metadata."""
        article = record['MedlineCitation']['Article']

        # Extract basic info
        title = article.get('ArticleTitle', 'Unknown Title')
        abstract = ""
        if 'Abstract' in article and 'AbstractText' in article['Abstract']:
            abstract_parts = article['Abstract']['AbstractText']
            if isinstance(abstract_parts, list):
                abstract = ' '.join(str(part) for part in abstract_parts)
            else:
                abstract = str(abstract_parts)

        # Extract authors
        authors = []
        if 'AuthorList' in article:
            for author in article['AuthorList']:
                if 'LastName' in author and 'ForeName' in author:
                    authors.append(f"{author['ForeName']} {author['LastName']}")

        # Extract journal info
        journal = article.get('Journal', {}).get('Title', 'Unknown Journal')

        # Extract publication date
        pub_date = ""
        if 'Journal' in article and 'JournalIssue' in article['Journal']:
            issue = article['Journal']['JournalIssue']
            if 'PubDate' in issue:
                pub_date_info = issue['PubDate']
                year = pub_date_info.get('Year', '')
                month = pub_date_info.get('Month', '')
                if year:
                    pub_date = f"{year}" + (f"-{month}" if month else "")

        # Extract DOI and PMID
        doi = None
        pmid = record['MedlineCitation']['PMID']

        if 'ELocationID' in article:
            for elocation in article['ELocationID']:
                if elocation.attributes.get('EIdType') == 'doi':
                    doi = str(elocation)

        return PaperMetadata(
            title=title,
            authors=authors,
            abstract=abstract,
            doi=doi,
            pubmed_id=str(pmid),
            journal=journal,
            publication_date=pub_date,
            source="pubmed",
            confidence_score=0.9
        )

    async def search_arxiv(self,
                           query: str,
                           max_results: int = 10,
                           categories: Optional[List[str]] = None) -> List[PaperMetadata]:
        """Search ArXiv for preprints."""
        if not ARXIV_AVAILABLE:
            logger.warning("ArXiv library not available")
            return []

        try:
            # Build search with categories
            search_query = query
            if categories:
                category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
                search_query = f"({query}) AND ({category_filter})"

            # Search ArXiv
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in search.results():
                paper = PaperMetadata(
                    title=result.title,
                    authors=[str(author) for author in result.authors],
                    abstract=result.summary,
                    arxiv_id=result.entry_id.split('/')[-1],
                    publication_date=result.published.strftime('%Y-%m-%d'),
                    url=result.entry_id,
                    keywords=[str(cat) for cat in result.categories],
                    source="arxiv",
                    confidence_score=0.8  # Slightly lower as preprints
                )
                papers.append(paper)

            logger.info(f"Retrieved {len(papers)} papers from ArXiv")
            return papers

        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []

    async def download_paper(self, paper: PaperMetadata, output_dir: str) -> Optional[str]:
        """Download paper PDF if available."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        safe_title = re.sub(r'[^\w\s-]', '', paper.title)[:50]
        filename = f"{safe_title}_{paper.source}.pdf"
        file_path = output_path / filename

        try:
            if paper.source == "arxiv" and paper.arxiv_id:
                # Download from ArXiv
                if ARXIV_AVAILABLE:
                    paper_obj = next(arxiv.Search(id_list=[paper.arxiv_id]).results())
                    paper_obj.download_pdf(dirpath=str(output_path), filename=filename)
                    return str(file_path)

            # For other sources, try DOI-based download (placeholder)
            elif paper.doi:
                # This would need integration with publisher APIs
                logger.info(f"DOI-based download not implemented for {paper.doi}")

        except Exception as e:
            logger.error(f"Failed to download paper: {e}")

        return None

    # STEP 2: CHUNK - Split text into passages
    async def process_document_llamaparse(self, file_path: str) -> List[ProcessedChunk]:
        """Process document using LlamaParse for superior extraction."""
        if not self.llamaparse:
            logger.warning("LlamaParse not available, falling back to basic processing")
            return await self.process_document_basic(file_path)

        try:
            documents = self.llamaparse.load_data(file_path)
            chunks = []

            for i, doc in enumerate(documents):
                chunk_id = f"llamaparse_{hashlib.md5(file_path.encode()).hexdigest()}_{i}"

                # Create paper metadata from filename
                paper_metadata = PaperMetadata(
                    title=Path(file_path).stem,
                    authors=[],
                    abstract="",
                    source="local_file"
                )

                chunk = ProcessedChunk(
                    content=doc.text,
                    chunk_id=chunk_id,
                    source_paper=paper_metadata,
                    chunk_index=i,
                    processing_method="llamaparse",
                    confidence=0.95
                )
                chunks.append(chunk)

            logger.info(f"Processed {len(chunks)} chunks with LlamaParse from {file_path}")
            return chunks

        except Exception as e:
            logger.error(f"LlamaParse processing failed: {e}")
            return await self.process_document_basic(file_path)

    async def process_document_basic(self, file_path: str) -> List[ProcessedChunk]:
        """Basic document processing fallback."""
        # This would integrate with your existing PDFVectorDB processing
        # For now, placeholder implementation
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple chunking
            chunks = self._chunk_text(content)
            processed_chunks = []

            paper_metadata = PaperMetadata(
                title=Path(file_path).stem,
                authors=[],
                abstract="",
                source="local_file"
            )

            for i, chunk_text in enumerate(chunks):
                chunk_id = f"basic_{hashlib.md5(file_path.encode()).hexdigest()}_{i}"
                chunk = ProcessedChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    source_paper=paper_metadata,
                    chunk_index=i,
                    processing_method="basic",
                    confidence=0.7
                )
                processed_chunks.append(chunk)

            return processed_chunks

        except Exception as e:
            logger.error(f"Basic document processing failed: {e}")
            return []

    def _chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """Intelligent text chunking with section awareness."""
        # Look for section headers
        sections = re.split(r'\n\n(?=[A-Z][^a-z]*\n)', text)

        chunks = []
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section.strip())
            else:
                # Split large sections
                words = section.split()
                current_chunk = []
                current_length = 0

                for word in words:
                    if current_length + len(word) > chunk_size and current_chunk:
                        chunks.append(' '.join(current_chunk))
                        # Keep overlap
                        overlap_words = current_chunk[-overlap // 10:] if overlap // 10 < len(
                            current_chunk) else current_chunk
                        current_chunk = overlap_words + [word]
                        current_length = sum(len(w) for w in current_chunk)
                    else:
                        current_chunk.append(word)
                        current_length += len(word) + 1

                if current_chunk:
                    chunks.append(' '.join(current_chunk))

        return [chunk for chunk in chunks if chunk.strip()]

    # STEP 3 & 4: EMBED & STORE - Integration with existing vector DB
    async def embed_and_store_chunks(self,
                                     chunks: List[ProcessedChunk],
                                     vector_db) -> bool:
        """Embed chunks and store in vector database."""
        try:
            # This integrates with your existing PDFVectorDB
            success_count = 0

            for chunk in chunks:
                # Get embedding
                if hasattr(vector_db, 'get_embedding'):
                    embedding = vector_db.get_embedding(chunk.content)
                    chunk.embedding = embedding

                # Store in database with enhanced metadata
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "source_title": chunk.source_paper.title,
                    "source": chunk.source_paper.source,
                    "processing_method": chunk.processing_method,
                    "confidence": chunk.confidence,
                    "chunk_index": chunk.chunk_index
                }

                # Add to vector DB (this would need integration with your existing PDFVectorDB)
                if hasattr(vector_db, 'add_chunk'):
                    success = vector_db.add_chunk(chunk.content, metadata, embedding)
                    if success:
                        success_count += 1

            logger.info(f"Stored {success_count}/{len(chunks)} chunks in vector database")
            return success_count == len(chunks)

        except Exception as e:
            logger.error(f"Failed to embed and store chunks: {e}")
            return False

    # STEP 5: RETRIEVE - Find relevant chunks
    async def retrieve_relevant_chunks(self,
                                       query: str,
                                       vector_db,
                                       top_k: int = 5,
                                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks with PaperQA2 integration."""
        try:
            # Use PaperQA2 if available
            if self.paperqa:
                # This would need proper PaperQA2 integration
                # For now, use basic search
                pass

            # Fallback to existing vector search
            if hasattr(vector_db, 'search_content'):
                results = vector_db.search_content(query, top_k=top_k)
                return results
            else:
                logger.warning("Vector database search not available")
                return []

        except Exception as e:
            logger.error(f"Chunk retrieval failed: {e}")
            return []

    # STEP 6: AUGMENT - Inject into LLM prompt
    def augment_prompt_with_context(self,
                                    base_prompt: str,
                                    retrieved_chunks: List[Dict[str, Any]],
                                    max_context_length: int = 4000) -> str:
        """Create augmented prompt with retrieved context."""
        if not retrieved_chunks:
            return base_prompt

        # Build context section
        context_parts = ["RELEVANT RESEARCH CONTEXT:\n"]
        current_length = len(base_prompt)

        for i, chunk in enumerate(retrieved_chunks, 1):
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})

            # Format context entry
            source = metadata.get('source_title', f'Source {i}')
            confidence = metadata.get('confidence', 1.0)

            context_entry = f"\n{i}. [{source}] (Confidence: {confidence:.2f})\n{content}\n"

            # Check length limit
            if current_length + len(context_entry) > max_context_length:
                break

            context_parts.append(context_entry)
            current_length += len(context_entry)

        context_section = "".join(context_parts)

        # Combine with base prompt
        augmented_prompt = f"""{context_section}

TASK:
{base_prompt}

INSTRUCTIONS:
- Use the research context above to inform your response
- Cite sources when making specific claims
- Indicate confidence levels for inferred information
- Prioritize information from higher-confidence sources
"""

        return augmented_prompt

    # Complete RAG pipeline
    async def run_rag_pipeline(self,
                               query: str,
                               vector_db,
                               search_online: bool = True,
                               max_papers: int = 10,
                               chunk_top_k: int = 5) -> Dict[str, Any]:
        """Run complete RAG pipeline for a biological query."""
        pipeline_results = {
            "query": query,
            "papers_found": [],
            "chunks_retrieved": [],
            "augmented_prompt": "",
            "processing_stats": {}
        }

        start_time = time.time()

        try:
            # Step 1: Search for new papers if requested
            if search_online:
                logger.info(f"Searching for papers related to: {query}")

                # Search PubMed and ArXiv in parallel
                tasks = []
                if BIOPYTHON_AVAILABLE and self.email:
                    tasks.append(self.search_pubmed(query, max_results=max_papers // 2))
                if ARXIV_AVAILABLE:
                    tasks.append(self.search_arxiv(query, max_results=max_papers // 2))

                if tasks:
                    paper_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in paper_results:
                        if isinstance(result, list):
                            pipeline_results["papers_found"].extend(result)

                logger.info(f"Found {len(pipeline_results['papers_found'])} new papers")

            # Step 2: Retrieve relevant chunks from existing database
            retrieved_chunks = await self.retrieve_relevant_chunks(
                query, vector_db, top_k=chunk_top_k
            )
            pipeline_results["chunks_retrieved"] = retrieved_chunks

            # Step 3: Create augmented prompt
            pipeline_results["augmented_prompt"] = self.augment_prompt_with_context(
                query, retrieved_chunks
            )

            # Stats
            pipeline_results["processing_stats"] = {
                "total_time": time.time() - start_time,
                "papers_found": len(pipeline_results["papers_found"]),
                "chunks_retrieved": len(retrieved_chunks),
                "search_online": search_online
            }

            logger.info(f"RAG pipeline completed in {pipeline_results['processing_stats']['total_time']:.2f}s")

        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            pipeline_results["error"] = str(e)

        return pipeline_results


# Specialized tools for biological modeling
class BiologyTools:
    """Specialized tools for biological modeling and analysis."""

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.biological_databases = {
            "uniprot": "https://rest.uniprot.org",
            "kegg": "https://rest.kegg.jp",
            "reactome": "https://reactome.org/ContentService",
            "string": "https://string-db.org/api"
        }

    async def search_protein_data(self,
                                  protein_name: str,
                                  organism: str = "homo sapiens") -> Dict[str, Any]:
        """Search protein data from UniProt."""
        try:
            async with aiohttp.ClientSession() as session:
                query = f"{protein_name} AND organism:{organism}"
                url = f"{self.biological_databases['uniprot']}/uniprotkb/search"
                params = {
                    "query": query,
                    "format": "json",
                    "limit": 10
                }

                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data

        except Exception as e:
            logger.error(f"Protein search failed: {e}")
            return {}

    async def get_pathway_data(self, pathway_id: str) -> Dict[str, Any]:
        """Get pathway data from KEGG."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.biological_databases['kegg']}/get/{pathway_id}"

                async with session.get(url) as response:
                    data = await response.text()
                    return {"pathway_id": pathway_id, "data": data}

        except Exception as e:
            logger.error(f"Pathway search failed: {e}")
            return {}

    async def analyze_protein_interactions(self,
                                           protein_list: List[str],
                                           organism_id: str = "9606") -> Dict[str, Any]:
        """Analyze protein-protein interactions using STRING."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.biological_databases['string']}/json/network"
                params = {
                    "identifiers": "%0d".join(protein_list),
                    "species": organism_id,
                    "required_score": 400
                }

                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data

        except Exception as e:
            logger.error(f"Protein interaction analysis failed: {e}")
            return {}

    def extract_biological_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract biological entities from text using regex patterns."""
        entities = {
            "proteins": [],
            "genes": [],
            "organisms": [],
            "pathways": [],
            "compounds": []
        }

        # Simple regex patterns (could be enhanced with NER models)
        patterns = {
            "proteins": r'\b[A-Z][a-z]+\d*\b',  # Simple protein pattern
            "genes": r'\b[A-Z]{2,6}\d*\b',  # Gene pattern
            "organisms": r'\b[A-Z]\. [a-z]+\b',  # Species pattern
        }

        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            entities[entity_type] = list(set(matches))

        return entities


# Integration functions for existing codebase
def create_enhanced_rag_pipeline(email: str,
                                 llamaparse_key: Optional[str] = None) -> RAGPipeline:
    """Factory function to create enhanced RAG pipeline."""
    return RAGPipeline(email=email, llamaparse_api_key=llamaparse_key)


def create_biology_tools(rag_pipeline: Optional[RAGPipeline] = None) -> BiologyTools:
    """Factory function to create biology tools."""
    return BiologyTools(rag_pipeline)


# Async context manager for tool sessions
class ToolSession:
    """Context manager for tool sessions with resource management."""

    def __init__(self, email: str, llamaparse_key: Optional[str] = None):
        self.rag_pipeline = RAGPipeline(email, llamaparse_key)
        self.biology_tools = BiologyTools(self.rag_pipeline)

    async def __aenter__(self):
        await self.rag_pipeline.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.rag_pipeline.__aexit__(exc_type, exc_val, exc_tb)

    async def search_and_process_papers(self,
                                        query: str,
                                        vector_db,
                                        download_papers: bool = False,
                                        output_dir: str = "./papers") -> Dict[str, Any]:
        """Complete pipeline: search, download, process papers."""
        results = await self.rag_pipeline.run_rag_pipeline(
            query, vector_db, search_online=True
        )

        if download_papers and results.get("papers_found"):
            downloaded_papers = []
            for paper in results["papers_found"]:
                file_path = await self.rag_pipeline.download_paper(paper, output_dir)
                if file_path:
                    downloaded_papers.append(file_path)

                    # Process downloaded paper
                    chunks = await self.rag_pipeline.process_document_llamaparse(file_path)
                    await self.rag_pipeline.embed_and_store_chunks(chunks, vector_db)

            results["downloaded_papers"] = downloaded_papers

        return results