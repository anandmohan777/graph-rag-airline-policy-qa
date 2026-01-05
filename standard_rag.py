import os
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

# ============================================================================
# CONFIGURATION
# ============================================================================
CHROMA_CHUNK_DIR = "chroma_chunk_db"
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
LLM_NAME = "gemma3:27b"
RETRIEVER_K_RAG = 4

KNOWN_AIRLINES = ['lufthansa', 'britishairways', 'airindia', 'singaporeairlines']

def natural_sort_key(s):
    """Split string into text and number parts for natural sorting"""
    return [int(text) if text.isdigit() else text
            for text in re.split('([0-9]+)', s)]


# ============================================================================
# STANDARD RAG CLASS
# ============================================================================
class StandardRAG:
    """Standard RAG implementation using ChromaDB vector store."""

    def __init__(self):
        self.embedder = None
        self.chunk_db = None
        self.retriever = None
        self.llm = None
        self.initialized = False

    def initialize(self) -> Tuple[bool, str]:
        """Initialize RAG components."""
        try:
            self.embedder = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_NAME,
                model_kwargs={"trust_remote_code": True}
            )

            if not Path(CHROMA_CHUNK_DIR).exists():
                return False, f"❌ ChromaDB not found at {CHROMA_CHUNK_DIR}. Please build index first."

            self.chunk_db = Chroma(
                persist_directory=CHROMA_CHUNK_DIR,
                embedding_function=self.embedder
            )

            self.retriever = self.chunk_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVER_K_RAG}
            )

            self.llm = OllamaLLM(model=LLM_NAME, temperature=0)
            self.initialized = True

            return True, "✅ Standard RAG initialized successfully"
        except Exception as e:
            return False, f"❌ Error initializing Standard RAG: {str(e)}"

    def _llm_extract_section_id(self, content: str) -> Optional[str]:
        """Extract section identifier from chunk content using LLM. Returns section ID like BA11, BA12 etc."""
        prompt = f"""Extract the section identifier from this airline policy text.

        IDENTIFIER AUTO-DETECTION:
        Automatically detect the section identifier pattern used in the document:
        - british airways: BA1, BA2, BA3, BA11, BA12
        - airindia: AI1, AI2, AI3, AI23, AI45
        - lufthansa: LH1, LH2, LH3
        - singaporeairlines: SIA1, SIA2, SIA3

        TEXT EXCERPT:
        {content}

        INSTRUCTIONS:
        1. Look for section identifiers in the text
        2. Return ONLY the identifier (e.g., "BA11")
        3. If no clear identifier found, return "NONE"
        4. Do not include any other text or explanation

        IDENTIFIER:"""
        try:
            response = self.llm.invoke(prompt).strip()
            # Clean up response
            response = response.replace('"', '').replace("'", "").strip()

            if response and response.upper() != "NONE" and response in content:
                return response.upper()
            return "UNKNOWN"
        except Exception as e:
            print(f"Error extracting section ID: {e}")
            return "UNKNOWN"

    def rebuild_index(self, folder_path: str = "data/policies") -> Tuple[bool, str]:
        """Rebuild ChromaDB index from documents."""
        try:
            # Load documents
            loader = DirectoryLoader(
                folder_path,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={'autodetect_encoding': True}
            )
            raw_docs = loader.load()

            # Markdown splitter
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2")
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False
            )

            # Split and extract section IDs
            split_docs = []
            for doc in raw_docs:
                source_path = Path(doc.metadata.get("source", ""))
                filename = source_path.name
                airline_name = filename.replace('.md', '')
                doc.metadata.update({
                    "doc_id": airline_name,
                    "airline": airline_name,
                    "source": str(source_path),
                    "document_type": "booking_policy",
                    "section_id": "UNKNOWN",
                    "original_filename": filename
                })

                header_splits = markdown_splitter.split_text(doc.page_content)

                print(f"\nProcessing {airline_name} document...")
                for i, split in enumerate(header_splits):
                    split.metadata.update(doc.metadata)

                    header_text = split.metadata.get('Header 2', '').strip()

                    # EXTRACT SECTION ID USING LLM
                    section_id = self._llm_extract_section_id(header_text)
                    split.metadata['section_id'] = section_id

                    if section_id:
                        print(f"  [{i + 1}/{len(header_splits)}] ✓ {section_id}")

                    split_docs.append(split)

            print(f"\n✅ Total chunks: {len(split_docs)}")

            # Create embeddings and persist
            self.chunk_db = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embedder,
                persist_directory=CHROMA_CHUNK_DIR,
                collection_metadata={"hnsw:space": "cosine"}
            )

            self.retriever = self.chunk_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVER_K_RAG}
            )

            return True, f"✅ Index rebuilt with {len(split_docs)} chunks from {len(raw_docs)} documents"
        except Exception as e:
            return False, f"❌ Error rebuilding index: {str(e)}"

    # ========================================================================
    # RAG.py
    # ========================================================================

    def _llm_extract_airlines(self, query: str) -> List[str]:
        """Use LLM to extract airlines from query. Returns standardized airline names."""
        prompt = f"""Extract all airline company names mentioned in the query. 
        Return ONLY the standardized airline names, comma-separated, with no extra words, 
        no explanations, no labels, no quotes. If none, return: none.

        Standardized airline names (pick closest match): {', '.join(KNOWN_AIRLINES)}

        Examples:
        Query: "Compare airindia, British Airways, lufthansa and SingaporeAirlines airlines baggage fees" -> airindia,britishairways,lufthansa,singaporeairlines
        Query: "How to get refund from airindia airline" -> airindia
        Query: "Tell me about flight changes on britishairways" -> britishairways
        Query: "My visa application was denied for SingaporeAirlines" -> singaporeairlines
        Query: "What is lufthansa refund policy?" -> lufthansa
        Query: "What are the cancellation policies?" -> none

        Now analyze this query:
        Query: "{query}"
        """

        try:
            response = self.llm.invoke(prompt).strip().lower()

            if response == "none" or not response:
                return []

            # Parse and clean the response
            airlines = []
            for airline in response.split(','):
                airline = airline.strip()
                if airline and airline != "none":
                    airlines.append(airline)

            # Validate against known airlines
            validated_airlines = []
            for airline in airlines:
                if airline in KNOWN_AIRLINES:
                    validated_airlines.append(airline)

            return list(set(validated_airlines))  # Remove duplicates
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return []

    def _classify_intent(self, airlines: List[str]) -> str:
        """Classify query intent based on extracted airlines."""
        if len(airlines) == 1:
            return "targeted"
        elif len(airlines) > 1:
            return "comparative"
        else:
            return "ambiguous"

    def _retrieve_with_filter(self, query: str, airline: str, k: int = RETRIEVER_K_RAG) -> List[Document]:
        """Retrieve documents with metadata filtering for a specific airline."""
        retriever = self.chunk_db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": {"airline": airline}
            }
        )
        return retriever.get_relevant_documents(query)

    async def query_async(self, question: str, airline_filter: Optional[str] = None) -> Dict[str, Any]:
        """Query RAG system asynchronously."""
        start_time = time.time()

        try:
            # Extract airlines
            mentioned_airlines = self._llm_extract_airlines(question)

            # Classify intent
            intent = self._classify_intent(mentioned_airlines)

            print("Airline RAG: ", mentioned_airlines)
            # Retrieve documents based on intent
            source_documents = []
            if intent in ["targeted", "comparative"]:
                for airline in mentioned_airlines:
                    airline_docs = self._retrieve_with_filter(question, airline, k=RETRIEVER_K_RAG)
                    source_documents.extend(airline_docs)
            else:
                # Ambiguous query - retrieve for all known airlines
                for airline in KNOWN_AIRLINES:
                    airline_docs = self._retrieve_with_filter(question, airline, k=RETRIEVER_K_RAG)
                    source_documents.extend(airline_docs)

            # Build context
            if not mentioned_airlines:
                mentioned_airlines = KNOWN_AIRLINES

            context = "\n\n".join([
                f"=== Airline: {doc.metadata.get('airline', 'Unknown')}\n=== Policy Heading Content:\n{doc.page_content}"
                for doc in source_documents
            ])

            # Generate answer using answer prompt structure
            answer_prompt = f"""
            You are an expert in {mentioned_airlines} policy interpretation.

            You must first decide an internal RESPONSE MODE based on the user question and the provided context:
            - MODE=DIRECT  → Use when the question is a simple lookup, list, or definition (e.g., “What are the fare classes?”, “Define Economy Saver”). 
              *Only answer from the most directly relevant section(s).* Do NOT pull in referenced/indirect sections, exceptions, or appendices unless the user explicitly asks.
            - MODE=MULTIHOP → Use when the question involves rules, refunds, exceptions, eligibility, timing, fees, documentation, or when a section says “see Section X / Appendix Y”.
              Follow cross-references and merge their contents to produce a complete, end-to-end answer.

            ALWAYS follow these rules:
            - Use ONLY the information in the context.
            - Be accurate and concise.
            - Cite sections you actually used. 
              - In MODE=DIRECT, cite only the minimal section(s) used.
              - In MODE=MULTIHOP, cite all sections/appendices used.
            - If multiple sections interact, state how they connect (briefly).
            - Do NOT include reasoning steps or mention the mode in the final answer.

            -----------------------
            RELATED POLICY CONTEXT:
            {context}
            -----------------------

            User Query: {question}

            Now produce the final answer only (no preamble, no mode). Keep it concise and well-structured. 
            If MODE=MULTIHOP, clearly merge the linked rules and any needed exceptions.
            Answer:
            """

            prompt = ChatPromptTemplate.from_template(answer_prompt)
            chain = prompt | self.llm | StrOutputParser()

            answer = chain.invoke({
                "airline": mentioned_airlines,
                "context": context,
                "question": question
            })

            # answer = "test answer"

            # Prepare chunks with metadata
            chunks_with_scores = []
            for i, doc in enumerate(source_documents[:4]):
                chunks_with_scores.append({
                    "chunk_number": i + 1,
                    "airline": doc.metadata.get("airline", "Unknown"),
                    "section_id": doc.metadata.get("section_id", "Unknown"),
                    "section": f"{doc.metadata.get('Header 1', 'N/A')} > {doc.metadata.get('Header 2', 'N/A')}",
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "similarity_score": 0.85 - (i * 0.05),  # Mock score
                    "source": doc.metadata.get("source", "N/A")
                })

            section_ids = [
                chunk_with_scores.get('section_id')
                for chunk_with_scores in chunks_with_scores
            ]

            print("Section IDs RAG: ", sorted(section_ids, key=natural_sort_key))
            response_time = time.time() - start_time

            return {
                "success": True,
                "answer": answer,
                "chunks": chunks_with_scores,
                "response_time": response_time,
                "section_ids": list(set(section_ids)),
                "num_chunks": len(source_documents),
                "airlines": mentioned_airlines,
                "intent": intent
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time
            }

