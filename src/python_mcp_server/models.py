"""Pydantic models for MCP server responses."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class EntityMetadata(BaseModel):
    """Metadata for an entity."""

    entity_type: Optional[str] = Field(None, description="Type of entity")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class SearchResult(BaseModel):
    """Single search result from knowledge graph."""

    id: str = Field(description="Unique identifier for the entity")
    content: str = Field(description="Content of the search result")
    score: Optional[float] = Field(None, description="Relevance score")
    metadata: EntityMetadata = Field(
        default_factory=EntityMetadata, description="Entity metadata"
    )


class SearchResults(BaseModel):
    """Collection of search results from knowledge graph."""

    items: list[SearchResult] = Field(description="List of search results")
    total: int = Field(description="Total number of results")
    source: Literal["knowledge_graph"] = Field(
        default="knowledge_graph",
        description="Source of the results - verified facts from knowledge graph",
    )


class DocumentMetadata(BaseModel):
    """Metadata for a document."""

    document_type: Optional[str] = Field(None, description="Type of document")
    source_url: Optional[str] = Field(None, description="Source URL")
    author: Optional[str] = Field(None, description="Document author")
    book: Optional[str] = Field(None, description="Book domain")
    section_level: Optional[str] = Field(None, description="Section level (h1-h4)")
    analysis_relevance: Optional[str] = Field(
        None, description="Analysis relevance (high/medium/low)"
    )


class Document(BaseModel):
    """Document from RAG vector search."""

    id: str = Field(description="Document identifier")
    title: Optional[str] = Field(None, description="Document title")
    content: str = Field(description="Document content")
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata, description="Document metadata"
    )
    similarity_score: float = Field(description="Vector similarity score")
    source: Literal["documents"] = Field(
        default="documents", description="Source type - supporting documents"
    )


class FactEvidence(BaseModel):
    """Graph evidence relevant to a statement. Caller judges entailment."""

    statement: str = Field(description="The statement the evidence was gathered for")
    evidence: list[SearchResult] = Field(
        description="Related facts from the knowledge graph, ranked by relevance"
    )


class EntityData(BaseModel):
    """Core entity information from graph."""

    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Type of entity")
    properties: EntityMetadata = Field(description="Entity properties")


class Relationship(BaseModel):
    """A relationship between entities."""

    source_id: str = Field(description="Source entity ID")
    target_id: str = Field(description="Target entity ID")
    relationship_type: str = Field(description="Type of relationship")
    properties: EntityMetadata = Field(
        default_factory=EntityMetadata, description="Relationship properties"
    )


class EntityContext(BaseModel):
    """Complete context for an entity from both graph and documents."""

    entity_id: str = Field(description="Entity identifier")
    entity_data: EntityData = Field(description="Core entity information from graph")
    relationships: list[Relationship] = Field(
        description="Related entities and connections"
    )
    supporting_documents: list[Document] = Field(
        description="Relevant documents for context"
    )


class CombinedResults(BaseModel):
    """Combined results from both graph and vector search."""

    graph_results: SearchResults = Field(description="Results from knowledge graph")
    vector_results: list[Document] = Field(description="Results from vector search")
    query: str = Field(description="Original search query")


class UsageExample(BaseModel):
    """Example usage pattern."""

    user_question: str = Field(description="Example user question")
    approach: str = Field(description="Recommended approach")
    reasoning: str = Field(description="Why this approach works")


class KnowledgeInstructions(BaseModel):
    """Instructions for LLM on how to use the knowledge base effectively."""

    title: str = Field(description="Title of the instruction set")
    instructions: list[str] = Field(description="Step-by-step instructions")
    examples: list[UsageExample] = Field(description="Example usage patterns")


class ExampleQuery(BaseModel):
    """Example of how to effectively query the knowledge base."""

    user_question: str = Field(description="Example user question")
    approach: str = Field(description="Recommended approach to answer")
    reasoning: str = Field(description="Why this approach works best")
    tools_used: list[str] = Field(description="Which tools to use in order")


class GraphSchema(BaseModel):
    """Knowledge graph schema information."""

    node_types: list[str] = Field(description="Available node types in the graph")
    relationship_types: list[str] = Field(description="Available relationship types")
    total_nodes: int = Field(description="Total number of nodes")
    total_relationships: int = Field(description="Total number of relationships")
    error: Optional[str] = Field(
        None, description="Error message if schema fetch failed"
    )
