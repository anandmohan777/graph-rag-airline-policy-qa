import os
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import re
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.chat_models import ChatOllama

# ============================================================================
# CONFIGURATION
# ============================================================================
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
LLM_NAME = "gemma3:27b"
RETRIEVER_K_GRAPHRAG = 4

os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

KNOWN_AIRLINES = ['lufthansa', 'britishairways', 'airindia', 'singaporeairlines']

def natural_sort_key(s):
    """Split string into text and number parts for natural sorting"""
    return [int(text) if text.isdigit() else text
            for text in re.split('([0-9]+)', s)]


# ============================================================================
# GRAPHRAG CLASS
# ============================================================================
class GraphRAG:
    """GraphRAG implementation using Neo4j knowledge graph."""

    def __init__(self):
        self.graph = None
        self.llm_json = None
        self.llm_chat = None
        self.embeddings = None
        self.vector_store = None
        self.initialized = False

    def initialize(self) -> Tuple[bool, str]:
        """Initialize GraphRAG components."""
        try:
            self.graph = Neo4jGraph(
                username=os.environ["NEO4J_USERNAME"],
                password=os.environ["NEO4J_PASSWORD"],
                url=os.environ["NEO4J_URI"]
            )

            self.llm_json = ChatOllama(
                model=LLM_NAME,
                temperature=0,
                format="json",
                num_ctx=16384
            )

            self.llm_chat = ChatOllama(
                model=LLM_NAME,
                temperature=0,
                num_ctx=16384
            )

            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_NAME,
                model_kwargs={"trust_remote_code": True},
                encode_kwargs={'normalize_embeddings': True}
            )

            # Check if vector store exists
            try:
                self.vector_store = Neo4jVector(
                    embedding=self.embeddings,
                    url=os.environ["NEO4J_URI"],
                    username=os.environ["NEO4J_USERNAME"],
                    password=os.environ["NEO4J_PASSWORD"],
                    index_name="policy_content",
                    node_label="PolicyContent",
                    text_node_property="content",
                    embedding_node_property="embedding"
                )
            except Exception:
                self.vector_store = None

            self.initialized = True
            return True, "✅ GraphRAG initialized successfully"

        except Exception as e:
            return False, f"❌ Error initializing GraphRAG: {str(e)}"

    def knowledge_graph_exists(self) -> bool:
        """Check if knowledge graph exists."""
        try:
            result = self.graph.query("MATCH (a:Airline) RETURN count(a) as count")
            return result[0]["count"] > 0
        except Exception:
            return False

    def rebuild_graph(self, folder_path: str = "data/policies") -> Tuple[bool, str]:
        """Rebuild Neo4j knowledge graph from documents."""
        try:
            # Clear existing graph
            self.graph.query("MATCH (n) DETACH DELETE n")
            self.graph.query("DROP INDEX policy_content IF EXISTS")

            # Load documents using YOUR existing function structure
            docs, msg = self._load_documents_from_dir(folder_path)
            if not docs:
                return False, msg

            # Group by airline
            docs_by_airline = defaultdict(list)
            for doc in docs:
                docs_by_airline[doc.metadata["airline"]].append(doc)

            # Build graph for each airline using YOUR existing setup_system logic
            success, message = self._setup_system(docs_by_airline)

            if success:
                total_sections = sum(len(docs) for docs in docs_by_airline.values())
                return True, f"✅ Graph rebuilt with {len(docs_by_airline)} airlines, ~{total_sections} sections"
            else:
                return False, message

        except Exception as e:
            return False, f"❌ Error rebuilding graph: {str(e)}"

    # ========================================================================
    # YOUR EXISTING FUNCTIONS FROM graphRAG.py - PRESERVED AS-IS
    # ========================================================================

    def _load_documents_from_dir(self, folder_path: str) -> Tuple[List[Document], str]:
        """Load documents from directory."""
        try:
            loader = DirectoryLoader(
                folder_path,
                glob="**/*.md",
                loader_cls=TextLoader
            )
            documents = loader.load()

            for doc in documents:
                source_path = Path(doc.metadata.get("source", ""))
                filename = source_path.name
                airline_name = filename.replace('.md', '')
                doc.metadata["airline"] = airline_name

            return documents, f"Loaded {len(documents)} documents from directory"
        except Exception as e:
            return [], f"Error loading documents: {str(e)}"

    def _extract_document_structure(self, documents: List[Document], airline_name: str) -> Dict[str, Any]:
        """Extract document structure using YOUR existing LLM prompt."""
        structure_prompt = """
        You are a document structure analyst. Extract the airline information, structural hierarchy with FULL CONTENT, and cross-references for ANY airline policy document.

        CRITICAL INSTRUCTION FOR CONTENT FIELD:
        - The "content" field MUST contain the complete, actual text from each section
        - DO NOT use placeholder text like "Full text content..." or "..."
        - Extract ALL text content word-for-word from the section
        - Include all subsections, bullet points, and paragraphs

        EXTRACTION FOCUS:
        1. Airline basic information (name, version, effective date)
        2. Section hierarchy with identifiers and COMPLETE ACTUAL CONTENT
        3. EXPLICIT CROSS-REFERENCES between sections

        IDENTIFIER AUTO-DETECTION:
        Automatically detect the section identifier pattern used in the document:
        - british airways: BA1, BA2, BA3
        - airindia: AI1, AI2, AI3
        - lufthansa: LH1, LH2, LH3
        - singaporeairlines: SIA1, SIA2, SIA3

        Preserve the EXACT identifier format found in the document.

        CROSS-REFERENCE EXTRACTION RULES:
        - Extract ONLY explicit references like "see Section X", "refer to Section Y", "as per Section Z"
        - Normalize references to consistent identifiers based on detected pattern:
            Examples for British Airways:
              - "Section BA1" → "BA1"
              - "Section BA2.1" → "BA2.1"
              - "Sections BA3, BA4" → ["BA3", "BA4"]

            Examples for airindia:
              - "Section AI1" → "AI1"
              - "Sections AI5, AI6" → ["AI5", "AI6"]

            Examples for Lufthansa:
              - "Section LH1" → "LH1"
              
            Examples for SingaporeAirlines:
              - "Section SIA1" → "SIA1"
        - Ignore implicit references like "above section", "following clause", "see below"

        RETURN STRUCTURE:
        {{
          "airline": {{
            "name": "{airline_name}",
            "document_version": "Extracted from document if available",
            "effective_date": "Extracted from document if available"
          }},
          "structural_hierarchy": [
            {{
                "name": "BA1. General Conditions",
                "identifier": "BA1",
                "content": "1.1. All tickets are non-transferable and valid only for the named passenger with valid ID. 1.2. Cancellation refunds depend on fare type (see Section BA2) and cancellation timing (see Section BA3). 1.3. Special circumstances may waive fees (see Section BA4).",
                "cross_references": ["BA2", "BA3", "BA4"]
            }},
            {{
                "name": "AI2. Fare Types",
                "identifier": "AI2",
                "content": "AI2.1. Economy Basic - Fare codes: Q, O, G - Non-refundable under normal circumstances - Refunds only available if Section AI4 exceptions apply",
                "cross_references": ["AI4"]
            }},
            {{
                "name": "Appendix BAA – Economy Basic Cancellation Fees",
                "identifier": "Appendix BAA",
                "content": "More than 7 days before departure: No refund. Within 7 days: No refund. Exception: Refund available only under Section BA4 special circumstances.",
                "cross_references": ["BA4"]
            }}
          ]
        }}

        IMPORTANT:
        - Extract REAL section content, not example placeholders
        - The "content" field must contain verbatim text from the document
        - Do NOT truncate content or use "..."
        - Include complete sentences and all details
        - Automatically adapt to whatever identifier pattern the document uses
        - Preserve subsection numbering (e.g., BA2.1, AI3.2, SIA3, LH4)

        Document:
        {document}
        """

        prompt = ChatPromptTemplate.from_template(structure_prompt)
        chain = prompt | self.llm_json | JsonOutputParser()

        try:
            document_structure = chain.invoke({
                "airline_name": airline_name,
                "document": documents[0].page_content
            })
            return document_structure
        except Exception as e:
            st.error(f"Error in structure extraction: {str(e)}")
            return {}

    def _create_knowledge_graph(self, document_structure: Dict[str, Any]) -> Tuple[bool, str]:
        """Create knowledge graph using YOUR existing Cypher queries."""
        try:
            airline = document_structure.get("airline", {})

            # Create Airline node
            self.graph.query(
                """
                MERGE (a:Airline {
                    name: $name,
                    document_version: $version,
                    effective_date: $date,
                    type: 'Airline'
                })
                """,
                {
                    "name": airline.get("name", ""),
                    "version": airline.get("document_version", "Unknown"),
                    "date": airline.get("effective_date", "Unknown")
                }
            )

            sections = document_structure.get("structural_hierarchy", [])
            section_relationships = []

            # Create PolicySection nodes
            for section in sections:
                self.graph.query(
                    """
                    MERGE (s:PolicySection {
                        identifier: $identifier,
                        name: $name,
                        type: 'PolicySection'
                    })
                    """,
                    {
                        "identifier": section.get("identifier", ""),
                        "name": section.get("name", "")
                    }
                )

                # Link airline to section
                self.graph.query(
                    """
                    MATCH (a:Airline {name: $airline_name})
                    MATCH (s:PolicySection {identifier: $section_id})
                    MERGE (a)-[r:HAS_POLICY_SECTION]->(s)
                    """,
                    {
                        "airline_name": airline.get("name", ""),
                        "section_id": section.get("identifier", "")
                    }
                )

                # Collect cross-references
                cross_refs = section.get("cross_references", [])
                for ref_id in cross_refs:
                    section_relationships.append({
                        "source_id": section.get("identifier", ""),
                        "target_id": ref_id
                    })

            # Create REFERENCES relationships
            successful_rels = 0
            for rel in section_relationships:
                try:
                    # Check if target exists
                    target_exists = self.graph.query(
                        """
                        MATCH (s:PolicySection {identifier: $target_id})
                        RETURN count(s) as count
                        """,
                        {"target_id": rel["target_id"]}
                    )

                    if target_exists[0]["count"] > 0:
                        self.graph.query(
                            """
                            MATCH (source:PolicySection {identifier: $source_id})
                            MATCH (target:PolicySection {identifier: $target_id})
                            MERGE (source)-[r:REFERENCES {relationship_type: 'CROSS_REFERENCE'}]->(target)
                            """,
                            {
                                "source_id": rel["source_id"],
                                "target_id": rel["target_id"]
                            }
                        )
                        successful_rels += 1
                except Exception:
                    pass

            return True, f"Knowledge graph created with {len(sections)} sections and {successful_rels} cross-reference relationships"
        except Exception as e:
            return False, f"Error creating knowledge graph: {str(e)}"

    def _create_vector_store(self, document_structure: Dict[str, Any]) -> Tuple[bool, str]:
        """Create vector store using YOUR existing structure."""
        try:
            airline = document_structure.get("airline", {})
            sections = document_structure.get("structural_hierarchy", [])

            documents = []
            for section in sections:
                doc = Document(
                    page_content=section.get("content", ""),
                    metadata={
                        "airline": airline.get("name", ""),
                        "section_identifier": section.get("identifier", ""),
                        "section_name": section.get("name", ""),
                        "cross_references": section.get("cross_references", [])
                    }
                )
                documents.append(doc)

            self.vector_store = Neo4jVector.from_documents(
                documents=documents,
                embedding=self.embeddings,
                url=os.environ["NEO4J_URI"],
                username=os.environ["NEO4J_USERNAME"],
                password=os.environ["NEO4J_PASSWORD"],
                index_name="policy_content",
                node_label="PolicyContent",
                text_node_property="content",
                embedding_node_property="embedding"
            )

            return True, f"Vector store created with {len(documents)} sections"
        except Exception as e:
            return False, f"Error creating vector store: {str(e)}"

    def _setup_system(self, documents_by_airline: Dict[str, List[Document]]) -> Tuple[bool, str]:
        """Setup system for all airlines using YOUR existing logic."""
        for airline_name, docs_for_airline in documents_by_airline.items():
            success, msg = self._setup_single_airline_system(docs_for_airline, airline_name)
            if not success:
                return False, f"Setup failed for {airline_name}: {msg}"
        return True, "All airline systems built successfully!"

    def _setup_single_airline_system(self, documents: List[Document], airline_name: str) -> Tuple[bool, str]:
        """Setup single airline system."""
        try:
            document_structure = self._extract_document_structure(documents, airline_name)
            if not document_structure:
                return False, f"Failed to extract document structure for {airline_name}"

            kg_success, kg_msg = self._create_knowledge_graph(document_structure)
            if not kg_success:
                return False, kg_msg

            vs_success, vs_msg = self._create_vector_store(document_structure)
            if not vs_success:
                return False, vs_msg

            return True, f"Setup successful for {airline_name}"
        except Exception as e:
            return False, str(e)

    def _llm_extract_airlines(self, query: str) -> List[str]:
        """Use LLM to extract airlines from query - YOUR EXISTING FUNCTION."""
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
            response = self.llm_chat.invoke(prompt).content.strip().lower()

            if response == "none" or not response:
                return []

            airlines = []
            for airline in response.split(','):
                airline = airline.strip()
                if airline and airline != "none":
                    airlines.append(airline)

            validated_airlines = []
            for airline in airlines:
                if airline in KNOWN_AIRLINES:
                    validated_airlines.append(airline)

            return list(set(validated_airlines))
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return []

    def _classify_intent(self, airlines: List[str]) -> str:
        """Classify query intent - YOUR EXISTING FUNCTION."""
        if len(airlines) == 1:
            return "targeted"
        elif len(airlines) > 1:
            return "comparative"
        else:
            return "ambiguous"

    def _retrieve_with_filter(self, question: str, mentioned_airlines: str, k: int = RETRIEVER_K_GRAPHRAG) -> List[Document]:
        """Retrieve with filter - YOUR EXISTING FUNCTION."""
        vector_results = self.vector_store.similarity_search(
            question,
            k=k,
            filter={"airline": mentioned_airlines}
        )
        return vector_results

    def _get_connected_sections(self, section_ids: List[str]) -> List[str]:
        """Get connected sections via graph traversal - YOUR EXISTING FUNCTION."""
        if not section_ids:
            return []

        query = """
        MATCH (start:PolicySection)
        WHERE start.identifier IN $section_ids
        MATCH (start)-[:REFERENCES*1..5]->(connected:PolicySection)
        RETURN DISTINCT connected.identifier as connected_id
        """

        try:
            results = self.graph.query(query, {"section_ids": section_ids})
            return [row["connected_id"] for row in results if row["connected_id"] not in section_ids]
        except Exception:
            return []

    def _get_sections_content(self, section_ids: List[str]) -> str:
        """Get sections content - YOUR EXISTING FUNCTION."""
        if not section_ids:
            return "No content found"

        try:
            query = """
            MATCH (n:PolicyContent)
            WHERE n.section_identifier IN $section_ids
            RETURN n.airline as airline, n.section_name as name,
                   n.section_identifier as identifier, n.content as content
            ORDER BY n.section_identifier
            """

            results = self.graph.query(query, {"section_ids": section_ids})

            content_parts = []
            for row in results:
                content_parts.append(
                    f"=== Airline: {row['airline']}, Policy Heading: {row['name']} ({row['identifier']}) ===\n=== Policy Heading Content:\n{row['content']}\n"
                )

            return "\n".join(content_parts)
        except Exception as e:
            return f"Error retrieving content: {str(e)}"

    def _generate_answer(self, question: str, context: str, section_ids: List[str], airline_name: List[str]) -> str:
        """Generate answer - YOUR EXISTING FUNCTION with YOUR EXISTING PROMPT."""
        if not airline_name:
            airline_name = KNOWN_AIRLINES

        answer_prompt = f"""You are an expert in airline policy interpretation.
        You must first decide an internal RESPONSE MODE based on the user question and the provided context:

        - MODE=DIRECT: Use when the question is a simple lookup, list, or definition (e.g., "What are the fare classes?", "Define Economy Saver").
          Only answer from the most directly relevant sections.
          Do NOT pull in referenced/indirect sections, exceptions, or appendices unless the user explicitly asks.

        - MODE=MULTI-HOP: Use when the question involves rules, refunds, exceptions, eligibility, timing, fees, documentation,
          or when a section says "see Section X / Appendix Y".
          Follow cross-references and merge their contents to produce a complete, end-to-end answer.

        ALWAYS follow these rules:
        - Use ONLY the information in the context.
        - Be accurate and concise.
        - Cite sections you actually used.
        - In MODE=DIRECT, cite only the minimal sections used.
        - In MODE=MULTI-HOP, cite all sections/appendices used.
        - If multiple sections interact, state how they connect (briefly).
        - Do NOT include reasoning steps or mention the mode in the final answer.

        -----------------------
        RELATED POLICY CONTEXT:
        {context}
        -----------------------

        User Query: {question}

        Now produce the final answer only (no preamble, no mode).
        Keep it concise and well-structured.
        If MODE=MULTI-HOP, clearly merge the linked rules and any needed exceptions.

        Answer:
        """

        prompt = ChatPromptTemplate.from_template(answer_prompt)
        chain = prompt | self.llm_chat | StrOutputParser()

        try:
            response = chain.invoke({
                "airline": airline_name,
                "sections": ", ".join(section_ids),
                "context": context,
                "question": question
            })
            return response
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    async def query_async(self, question: str) -> Dict[str, Any]:
        """Query GraphRAG system using YOUR EXISTING GRAPHRAG_QUERY LOGIC."""
        start_time = time.time()

        try:
            # Extract airlines using YOUR existing function
            mentioned_airlines = self._llm_extract_airlines(question)

            # Classify intent using YOUR existing function
            intent = self._classify_intent(mentioned_airlines)

            # Retrieve documents based on intent
            vector_results = []
            if intent in ["targeted", "comparative"]:
                for airline in mentioned_airlines:
                    airline_docs = self._retrieve_with_filter(question, airline, k=RETRIEVER_K_GRAPHRAG)
                    vector_results.extend(airline_docs)
            else:
                # Ambiguous - search all airlines
                for airline in KNOWN_AIRLINES:
                    airline_docs = self._retrieve_with_filter(question, airline, k=RETRIEVER_K_GRAPHRAG)
                    vector_results.extend(airline_docs)

            if not vector_results:
                return {
                    "success": False,
                    "error": "No relevant sections found",
                    "response_time": time.time() - start_time
                }

            # Get initial section IDs
            initial_section_ids = [doc.metadata["section_identifier"] for doc in vector_results]
            print("Section IDs graphRAG: ", sorted(initial_section_ids, key=natural_sort_key))

            # Get connected sections
            connected_sections = self._get_connected_sections(initial_section_ids)

            # Combine all sections
            all_relevant_ids = list(set(initial_section_ids + connected_sections))
            print("All IDs graphRAG: ", sorted(list(set(all_relevant_ids)), key=natural_sort_key))

            # Get content
            all_content = self._get_sections_content(all_relevant_ids)

            # Generate answer using YOUR existing function
            answer = self._generate_answer(question, all_content, all_relevant_ids, mentioned_airlines)
            # answer = "test answer"

            # Build traversal path
            traversal_path = {
                "initial_nodes": initial_section_ids,
                "connected_nodes": connected_sections,
                "relationships_used": len(connected_sections),
                "total_nodes": len(all_relevant_ids)
            }

            response_time = time.time() - start_time

            return {
                "success": True,
                "answer": answer,
                "traversal_path": traversal_path,
                "response_time": response_time,
                "section_ids": list(set(all_relevant_ids)),
                "num_sections": len(all_relevant_ids),
                "intent": intent,
                "airlines": mentioned_airlines
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time
            }

