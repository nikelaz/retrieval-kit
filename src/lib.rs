use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use backends::lancedb::{Chunk, DocumentRecord, LanceDbBackend};
use glob::{GlobError, PatternError, glob};
use lancedb::Error as LanceDbError;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

pub mod backends;
pub mod chunking;
pub mod embeddings;
mod node;

pub use chunking::{ChunkingConfig, ChunkingError, chunk_text};
pub use embeddings::{EmbeddingError, EmbeddingsConfig, EmbeddingsProvider, OrtEmbedder};

#[derive(Clone, Debug, Eq, PartialEq)]
/// Storage engine configuration for a retrieval index.
pub enum DbEngine {
    /// Store documents and chunks in a local LanceDB database.
    LanceDb {
        /// LanceDB database directory or URI.
        path: PathBuf,
        /// Number of `f32` values produced by the configured embedder.
        vector_dimensions: i32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Embedding provider configuration.
pub enum EmbeddingsProviderKind {
    /// Use ONNX Runtime with a sentence-transformers style model.
    Ort(EmbeddingsConfig),
}

#[derive(Debug, Eq, PartialEq)]
pub enum RKitConfigError {
    InvalidVectorDimensions(i32),
    InvalidChunkingConfig(ChunkingError),
}

impl fmt::Display for RKitConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVectorDimensions(value) => {
                write!(
                    f,
                    "vector_dimensions must be greater than zero, got {value}"
                )
            }
            Self::InvalidChunkingConfig(error) => {
                write!(f, "invalid chunking config: {error}")
            }
        }
    }
}

impl Error for RKitConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidVectorDimensions(_) => None,
            Self::InvalidChunkingConfig(error) => Some(error),
        }
    }
}

#[derive(Debug)]
pub enum RKitInitError {
    DbEngine(LanceDbError),
    EmbeddingsProvider(EmbeddingError),
    EmbeddingDimensionMismatch {
        db_vector_dimensions: i32,
        embedding_dimensions: usize,
    },
}

impl fmt::Display for RKitInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DbEngine(error) => write!(f, "failed to initialize database engine: {error}"),
            Self::EmbeddingsProvider(error) => {
                write!(f, "failed to initialize embeddings provider: {error}")
            }
            Self::EmbeddingDimensionMismatch {
                db_vector_dimensions,
                embedding_dimensions,
            } => write!(
                f,
                "embedding dimension mismatch: database expects {db_vector_dimensions}, embedder returns {embedding_dimensions}"
            ),
        }
    }
}

impl Error for RKitInitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::DbEngine(error) => Some(error),
            Self::EmbeddingsProvider(error) => Some(error),
            Self::EmbeddingDimensionMismatch { .. } => None,
        }
    }
}

const DEFAULT_TOOL_LIMIT: usize = 10;
const SEMANTIC_SEARCH_TOOL_NAME: &str = "semantic_search";
const KEYWORD_SEARCH_TOOL_NAME: &str = "keyword_search";
const LIST_DOCUMENTS_TOOL_NAME: &str = "list_documents";
const GET_DOCUMENT_TOOL_NAME: &str = "get_document";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Result returned after a document is ingested or upserted.
pub struct IngestDocumentResult {
    /// Stable identifier assigned to or provided for the document.
    pub document_id: String,
    /// Number of chunks written for the document.
    pub chunk_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Lightweight document listing entry.
pub struct DocumentSummary {
    /// Stored document identifier.
    pub document_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Full stored document.
pub struct Document {
    /// Stored document identifier.
    pub document_id: String,
    /// Original full document content.
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// Semantic search match returned from vector search.
pub struct VectorSearchResult {
    /// Identifier of the source document.
    pub document_id: String,
    /// Matching chunk text.
    pub text: String,
    /// LanceDB vector distance; lower values are closer.
    pub distance: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// Full-text search match returned from keyword search.
pub struct KeywordSearchResult {
    /// Identifier of the source document.
    pub document_id: String,
    /// Matching chunk text.
    pub text: String,
    /// LanceDB full-text relevance score.
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// MCP-style tool definition with a JSON Schema input shape.
pub struct ToolDefinition {
    /// Tool name used with `invoke_tool`.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Human-readable tool description.
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    /// JSON Schema object describing accepted arguments.
    pub input_schema: Value,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
/// Optional description overrides for built-in retrieval tools.
pub struct ToolDescriptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_search: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyword_search: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_documents: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub get_document: Option<String>,
}

#[derive(Debug)]
pub enum IngestDocumentError {
    NotInitialized,
    EmptyContent,
    FileRead {
        path: PathBuf,
        source: std::io::Error,
    },
    InvalidGlobPattern(PatternError),
    Glob(GlobError),
    NoFilesMatched {
        pattern: String,
    },
    Chunking(ChunkingError),
    Embeddings(EmbeddingError),
    DbEngine(LanceDbError),
}

impl fmt::Display for IngestDocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInitialized => write!(f, "RKit must be initialized before ingesting"),
            Self::EmptyContent => write!(f, "document content must not be empty"),
            Self::FileRead { path, source } => {
                write!(
                    f,
                    "failed to read document file {}: {source}",
                    path.display()
                )
            }
            Self::InvalidGlobPattern(error) => {
                write!(f, "invalid document file glob pattern: {error}")
            }
            Self::Glob(error) => write!(f, "failed to resolve document file glob: {error}"),
            Self::NoFilesMatched { pattern } => {
                write!(f, "document file glob matched no files: {pattern}")
            }
            Self::Chunking(error) => write!(f, "failed to chunk document: {error}"),
            Self::Embeddings(error) => write!(f, "failed to generate embeddings: {error}"),
            Self::DbEngine(error) => write!(f, "failed to insert document into database: {error}"),
        }
    }
}

impl Error for IngestDocumentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NotInitialized | Self::EmptyContent | Self::NoFilesMatched { .. } => None,
            Self::FileRead { source, .. } => Some(source),
            Self::InvalidGlobPattern(error) => Some(error),
            Self::Glob(error) => Some(error),
            Self::Chunking(error) => Some(error),
            Self::Embeddings(error) => Some(error),
            Self::DbEngine(error) => Some(error),
        }
    }
}

#[derive(Debug)]
pub enum DocumentError {
    NotInitialized,
    DbEngine(LanceDbError),
}

impl fmt::Display for DocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInitialized => write!(f, "RKit must be initialized before reading documents"),
            Self::DbEngine(error) => write!(f, "failed to read documents from database: {error}"),
        }
    }
}

impl Error for DocumentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NotInitialized => None,
            Self::DbEngine(error) => Some(error),
        }
    }
}

#[derive(Debug)]
pub enum VectorSearchError {
    NotInitialized,
    EmptyQuery,
    Embeddings(EmbeddingError),
    DbEngine(LanceDbError),
}

impl fmt::Display for VectorSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInitialized => write!(f, "RKit must be initialized before searching"),
            Self::EmptyQuery => write!(f, "search query must not be empty"),
            Self::Embeddings(error) => write!(f, "failed to generate query embedding: {error}"),
            Self::DbEngine(error) => write!(f, "failed to search database: {error}"),
        }
    }
}

impl Error for VectorSearchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NotInitialized | Self::EmptyQuery => None,
            Self::Embeddings(error) => Some(error),
            Self::DbEngine(error) => Some(error),
        }
    }
}

#[derive(Debug)]
pub enum KeywordSearchError {
    NotInitialized,
    EmptyQuery,
    DbEngine(LanceDbError),
}

impl fmt::Display for KeywordSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInitialized => write!(f, "RKit must be initialized before searching"),
            Self::EmptyQuery => write!(f, "search query must not be empty"),
            Self::DbEngine(error) => write!(f, "failed to search database: {error}"),
        }
    }
}

impl Error for KeywordSearchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NotInitialized | Self::EmptyQuery => None,
            Self::DbEngine(error) => Some(error),
        }
    }
}

#[derive(Debug)]
pub enum InvokeToolError {
    UnknownTool(String),
    InvalidArguments {
        tool_name: String,
        source: serde_json::Error,
    },
    Serialization(serde_json::Error),
    SemanticSearch(VectorSearchError),
    KeywordSearch(KeywordSearchError),
    Document(DocumentError),
}

impl fmt::Display for InvokeToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTool(name) => write!(f, "unknown tool: {name}"),
            Self::InvalidArguments { tool_name, source } => {
                write!(f, "invalid arguments for tool {tool_name}: {source}")
            }
            Self::Serialization(error) => write!(f, "failed to serialize tool result: {error}"),
            Self::SemanticSearch(error) => write!(f, "semantic search tool failed: {error}"),
            Self::KeywordSearch(error) => write!(f, "keyword search tool failed: {error}"),
            Self::Document(error) => write!(f, "document tool failed: {error}"),
        }
    }
}

impl Error for InvokeToolError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::UnknownTool(_) => None,
            Self::InvalidArguments { source, .. } => Some(source),
            Self::Serialization(error) => Some(error),
            Self::SemanticSearch(error) => Some(error),
            Self::KeywordSearch(error) => Some(error),
            Self::Document(error) => Some(error),
        }
    }
}

pub struct RKit {
    db_engine_config: DbEngine,
    embeddings_provider_config: EmbeddingsProviderKind,
    chunking_config: ChunkingConfig,
    db_engine: Option<InitializedDbEngine>,
    embeddings_provider: Option<InitializedEmbeddingsProvider>,
}

impl RKit {
    pub fn new(
        db_engine: DbEngine,
        embeddings_provider: EmbeddingsProviderKind,
    ) -> Result<Self, RKitConfigError> {
        validate_db_engine(&db_engine)?;

        Ok(Self {
            db_engine_config: db_engine,
            embeddings_provider_config: embeddings_provider,
            chunking_config: ChunkingConfig::default(),
            db_engine: None,
            embeddings_provider: None,
        })
    }

    pub async fn init(&mut self) -> Result<(), RKitInitError> {
        let initialized_db_engine = init_db_engine(self.db_engine_config.clone()).await?;
        let initialized_embeddings_provider =
            init_embeddings_provider(self.embeddings_provider_config.clone())?;
        validate_embedding_dimensions(&initialized_db_engine, &initialized_embeddings_provider)?;
        ensure_db_engine_tables(&initialized_db_engine).await?;

        self.db_engine = Some(initialized_db_engine);
        self.embeddings_provider = Some(initialized_embeddings_provider);

        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.db_engine.is_some() && self.embeddings_provider.is_some()
    }

    pub fn db_engine_initialized(&self) -> bool {
        self.db_engine.is_some()
    }

    pub fn embeddings_provider_initialized(&self) -> bool {
        self.embeddings_provider.is_some()
    }

    pub fn lancedb_backend(&self) -> Option<&LanceDbBackend> {
        match self.db_engine.as_ref() {
            Some(InitializedDbEngine::LanceDb(backend)) => Some(backend),
            None => None,
        }
    }

    pub fn ort_embedder(&self) -> Option<&embeddings::OrtEmbedder> {
        match self.embeddings_provider.as_ref() {
            Some(InitializedEmbeddingsProvider::Ort(embedder)) => Some(embedder),
            #[cfg(test)]
            Some(InitializedEmbeddingsProvider::Mock(_)) => None,
            None => None,
        }
    }

    pub fn ort_embedder_mut(&mut self) -> Option<&mut embeddings::OrtEmbedder> {
        match self.embeddings_provider.as_mut() {
            Some(InitializedEmbeddingsProvider::Ort(embedder)) => Some(embedder),
            #[cfg(test)]
            Some(InitializedEmbeddingsProvider::Mock(_)) => None,
            None => None,
        }
    }

    pub fn clear_initialized_state(&mut self) {
        self.db_engine = None;
        self.embeddings_provider = None;
    }

    pub fn db_engine_config(&self) -> &DbEngine {
        &self.db_engine_config
    }

    pub fn embeddings_provider_config(&self) -> &EmbeddingsProviderKind {
        &self.embeddings_provider_config
    }

    pub fn chunking_config(&self) -> ChunkingConfig {
        self.chunking_config
    }

    pub fn set_chunking_config(
        &mut self,
        chunking_config: ChunkingConfig,
    ) -> Result<(), RKitConfigError> {
        validate_chunking_config(chunking_config)?;
        self.chunking_config = chunking_config;
        Ok(())
    }

    pub fn select_db_engine(&mut self, db_engine: DbEngine) -> Result<(), RKitConfigError> {
        validate_db_engine(&db_engine)?;
        self.db_engine_config = db_engine;
        self.db_engine = None;
        Ok(())
    }

    pub fn register_embeddings_provider(
        &mut self,
        embeddings_provider: EmbeddingsProviderKind,
    ) -> Result<(), RKitConfigError> {
        self.embeddings_provider_config = embeddings_provider;
        self.embeddings_provider = None;
        Ok(())
    }

    pub async fn ingest_document(
        &mut self,
        content: String,
    ) -> Result<IngestDocumentResult, IngestDocumentError> {
        let mut results = self.ingest_documents(vec![content]).await?;
        Ok(results
            .pop()
            .expect("single-document ingest should always yield one result"))
    }

    pub async fn ingest_document_file<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<IngestDocumentResult, IngestDocumentError> {
        let content = read_document_file(path.as_ref())?;
        self.ingest_document(content).await
    }

    pub async fn upsert_document(
        &mut self,
        id: String,
        content: String,
    ) -> Result<IngestDocumentResult, IngestDocumentError> {
        let document = self.prepare_document_with_id(id, content)?;
        let db_engine = self
            .db_engine
            .as_ref()
            .ok_or(IngestDocumentError::NotInitialized)?;
        let embeddings_provider = self
            .embeddings_provider
            .as_ref()
            .ok_or(IngestDocumentError::NotInitialized)?;
        let embeddings = embeddings_provider
            .embed_batch(&document.chunks)
            .await
            .map_err(IngestDocumentError::Embeddings)?;

        let mut chunks = Vec::with_capacity(document.chunks.len());
        let mut embedding_iter = embeddings.into_iter();
        let chunk_count = document.chunks.len();

        for text in document.chunks {
            let vector = embedding_iter.next().ok_or_else(|| {
                IngestDocumentError::Embeddings(EmbeddingError::MissingOutput(
                    "fewer embeddings returned than chunks provided".to_string(),
                ))
            })?;
            chunks.push(Chunk {
                document_id: document.document_id.clone(),
                text,
                vector,
            });
        }

        if embedding_iter.next().is_some() {
            return Err(IngestDocumentError::Embeddings(
                EmbeddingError::MissingOutput(
                    "more embeddings returned than chunks provided".to_string(),
                ),
            ));
        }

        match db_engine {
            InitializedDbEngine::LanceDb(backend) => backend
                .upsert_data(
                    &DocumentRecord {
                        document_id: document.document_id.clone(),
                        content: document.content,
                    },
                    &chunks,
                )
                .await
                .map_err(IngestDocumentError::DbEngine)?,
        }

        Ok(IngestDocumentResult {
            document_id: document.document_id,
            chunk_count,
        })
    }

    pub async fn ingest_documents(
        &mut self,
        documents: Vec<String>,
    ) -> Result<Vec<IngestDocumentResult>, IngestDocumentError> {
        if documents.is_empty() {
            return Err(IngestDocumentError::EmptyContent);
        }

        let prepared_documents = documents
            .into_iter()
            .map(|content| self.prepare_document(content))
            .collect::<Result<Vec<_>, _>>()?;
        let flattened_chunks = prepared_documents
            .iter()
            .flat_map(|document| document.chunks.iter().cloned())
            .collect::<Vec<_>>();

        let db_engine = self
            .db_engine
            .as_ref()
            .ok_or(IngestDocumentError::NotInitialized)?;
        let embeddings_provider = self
            .embeddings_provider
            .as_ref()
            .ok_or(IngestDocumentError::NotInitialized)?;
        let embeddings = embeddings_provider
            .embed_batch(&flattened_chunks)
            .await
            .map_err(IngestDocumentError::Embeddings)?;

        let mut documents = Vec::with_capacity(prepared_documents.len());
        let mut chunks = Vec::with_capacity(flattened_chunks.len());
        let mut embedding_iter = embeddings.into_iter();
        let mut results = Vec::with_capacity(prepared_documents.len());

        for document in prepared_documents {
            let chunk_count = document.chunks.len();

            for text in document.chunks {
                let vector = embedding_iter.next().ok_or_else(|| {
                    IngestDocumentError::Embeddings(EmbeddingError::MissingOutput(
                        "fewer embeddings returned than chunks provided".to_string(),
                    ))
                })?;
                chunks.push(Chunk {
                    document_id: document.document_id.clone(),
                    text,
                    vector,
                });
            }

            documents.push(DocumentRecord {
                document_id: document.document_id.clone(),
                content: document.content,
            });
            results.push(IngestDocumentResult {
                document_id: document.document_id,
                chunk_count,
            });
        }

        if embedding_iter.next().is_some() {
            return Err(IngestDocumentError::Embeddings(
                EmbeddingError::MissingOutput(
                    "more embeddings returned than chunks provided".to_string(),
                ),
            ));
        }

        match db_engine {
            InitializedDbEngine::LanceDb(backend) => backend
                .insert_data(&documents, &chunks)
                .await
                .map_err(IngestDocumentError::DbEngine)?,
        }

        Ok(results)
    }

    pub async fn ingest_document_files(
        &mut self,
        pattern: &str,
    ) -> Result<Vec<IngestDocumentResult>, IngestDocumentError> {
        let paths = glob(pattern)
            .map_err(IngestDocumentError::InvalidGlobPattern)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(IngestDocumentError::Glob)?;

        if paths.is_empty() {
            return Err(IngestDocumentError::NoFilesMatched {
                pattern: pattern.to_string(),
            });
        }

        let documents = paths
            .into_iter()
            .map(|path| read_document_file(&path))
            .collect::<Result<Vec<_>, _>>()?;

        self.ingest_documents(documents).await
    }

    pub async fn vector_search(
        &self,
        query: String,
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorSearchError> {
        if query.trim().is_empty() {
            return Err(VectorSearchError::EmptyQuery);
        }
        if !self.is_initialized() {
            return Err(VectorSearchError::NotInitialized);
        }
        if limit == 0 {
            return Ok(Vec::new());
        }

        let embeddings_provider = self
            .embeddings_provider
            .as_ref()
            .expect("initialized embeddings provider");
        let mut embeddings = embeddings_provider
            .embed_batch(&[query])
            .await
            .map_err(VectorSearchError::Embeddings)?;
        let query_vector = embeddings.pop().ok_or_else(|| {
            VectorSearchError::Embeddings(EmbeddingError::MissingOutput(
                "no query embedding returned".to_string(),
            ))
        })?;

        if embeddings.pop().is_some() {
            return Err(VectorSearchError::Embeddings(
                EmbeddingError::MissingOutput("more than one query embedding returned".to_string()),
            ));
        }

        let db_engine = self.db_engine.as_ref().expect("initialized db engine");
        match db_engine {
            InitializedDbEngine::LanceDb(backend) => backend
                .vector_search(query_vector, limit)
                .await
                .map(|results| {
                    results
                        .into_iter()
                        .map(|result| VectorSearchResult {
                            document_id: result.document_id,
                            text: result.text,
                            distance: result.distance,
                        })
                        .collect()
                })
                .map_err(VectorSearchError::DbEngine),
        }
    }

    pub async fn keyword_search(
        &self,
        query: String,
        limit: usize,
    ) -> Result<Vec<KeywordSearchResult>, KeywordSearchError> {
        if query.trim().is_empty() {
            return Err(KeywordSearchError::EmptyQuery);
        }
        if !self.is_initialized() {
            return Err(KeywordSearchError::NotInitialized);
        }
        if limit == 0 {
            return Ok(Vec::new());
        }

        let db_engine = self.db_engine.as_ref().expect("initialized db engine");
        match db_engine {
            InitializedDbEngine::LanceDb(backend) => backend
                .keyword_search(query, limit)
                .await
                .map(|results| {
                    results
                        .into_iter()
                        .map(|result| KeywordSearchResult {
                            document_id: result.document_id,
                            text: result.text,
                            score: result.score,
                        })
                        .collect()
                })
                .map_err(KeywordSearchError::DbEngine),
        }
    }

    pub async fn list_documents(&self) -> Result<Vec<DocumentSummary>, DocumentError> {
        if !self.is_initialized() {
            return Err(DocumentError::NotInitialized);
        }
        let db_engine = self.db_engine.as_ref().expect("initialized db engine");

        match db_engine {
            InitializedDbEngine::LanceDb(backend) => backend
                .list_documents()
                .await
                .map(|documents| {
                    documents
                        .into_iter()
                        .map(|document| DocumentSummary {
                            document_id: document.document_id,
                        })
                        .collect()
                })
                .map_err(DocumentError::DbEngine),
        }
    }

    pub async fn get_document(&self, id: String) -> Result<Option<Document>, DocumentError> {
        if !self.is_initialized() {
            return Err(DocumentError::NotInitialized);
        }
        let db_engine = self.db_engine.as_ref().expect("initialized db engine");

        match db_engine {
            InitializedDbEngine::LanceDb(backend) => backend
                .get_document(&id)
                .await
                .map(|document| {
                    document.map(|document| Document {
                        document_id: document.document_id,
                        content: document.content,
                    })
                })
                .map_err(DocumentError::DbEngine),
        }
    }

    pub async fn delete_document(&self, id: String) -> Result<(), DocumentError> {
        if !self.is_initialized() {
            return Err(DocumentError::NotInitialized);
        }
        let db_engine = self.db_engine.as_ref().expect("initialized db engine");

        match db_engine {
            InitializedDbEngine::LanceDb(backend) => backend
                .delete_document(&id)
                .await
                .map_err(DocumentError::DbEngine),
        }
    }

    pub fn get_tool_definitions(
        &self,
        descriptions: Option<ToolDescriptions>,
    ) -> Vec<ToolDefinition> {
        let descriptions = descriptions.unwrap_or_default();

        vec![
            ToolDefinition {
                name: SEMANTIC_SEARCH_TOOL_NAME.to_string(),
                description: Some(descriptions.semantic_search.unwrap_or_else(|| {
                    "Search ingested documents by semantic similarity to a natural language query."
                        .to_string()
                })),
                input_schema: search_tool_input_schema(),
            },
            ToolDefinition {
                name: KEYWORD_SEARCH_TOOL_NAME.to_string(),
                description: Some(descriptions.keyword_search.unwrap_or_else(|| {
                    "Search ingested documents by exact keyword and full-text relevance."
                        .to_string()
                })),
                input_schema: search_tool_input_schema(),
            },
            ToolDefinition {
                name: LIST_DOCUMENTS_TOOL_NAME.to_string(),
                description: Some(descriptions.list_documents.unwrap_or_else(|| {
                    "List document identifiers currently stored in the retrieval index.".to_string()
                })),
                input_schema: empty_tool_input_schema(),
            },
            ToolDefinition {
                name: GET_DOCUMENT_TOOL_NAME.to_string(),
                description: Some(descriptions.get_document.unwrap_or_else(|| {
                    "Fetch the full stored content for a document by document identifier."
                        .to_string()
                })),
                input_schema: get_document_tool_input_schema(),
            },
        ]
    }

    pub async fn invoke_tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<Value, InvokeToolError> {
        match name {
            SEMANTIC_SEARCH_TOOL_NAME => {
                let arguments: SearchToolArguments = parse_tool_arguments(name, arguments)?;
                let results = self
                    .vector_search(
                        arguments.query,
                        arguments.limit.unwrap_or(DEFAULT_TOOL_LIMIT),
                    )
                    .await
                    .map_err(InvokeToolError::SemanticSearch)?;
                serialize_tool_result(json!({ "results": results }))
            }
            KEYWORD_SEARCH_TOOL_NAME => {
                let arguments: SearchToolArguments = parse_tool_arguments(name, arguments)?;
                let results = self
                    .keyword_search(
                        arguments.query,
                        arguments.limit.unwrap_or(DEFAULT_TOOL_LIMIT),
                    )
                    .await
                    .map_err(InvokeToolError::KeywordSearch)?;
                serialize_tool_result(json!({ "results": results }))
            }
            LIST_DOCUMENTS_TOOL_NAME => {
                let _: EmptyToolArguments = parse_tool_arguments(name, arguments)?;
                let documents = self
                    .list_documents()
                    .await
                    .map_err(InvokeToolError::Document)?;
                serialize_tool_result(json!({ "documents": documents }))
            }
            GET_DOCUMENT_TOOL_NAME => {
                let arguments: GetDocumentToolArguments = parse_tool_arguments(name, arguments)?;
                let document = self
                    .get_document(arguments.document_id)
                    .await
                    .map_err(InvokeToolError::Document)?;
                serialize_tool_result(json!({ "document": document }))
            }
            other => Err(InvokeToolError::UnknownTool(other.to_string())),
        }
    }

    fn prepare_document(&self, content: String) -> Result<PreparedDocument, IngestDocumentError> {
        self.prepare_document_with_id(Uuid::new_v4().to_string(), content)
    }

    fn prepare_document_with_id(
        &self,
        document_id: String,
        content: String,
    ) -> Result<PreparedDocument, IngestDocumentError> {
        if content.trim().is_empty() {
            return Err(IngestDocumentError::EmptyContent);
        }

        let chunks = self
            .chunk_document_content(&content)
            .map_err(IngestDocumentError::Chunking)?;
        if chunks.is_empty() {
            return Err(IngestDocumentError::EmptyContent);
        }

        Ok(PreparedDocument {
            document_id,
            content,
            chunks,
        })
    }

    fn chunk_document_content(&self, content: &str) -> Result<Vec<String>, ChunkingError> {
        match self.embeddings_provider.as_ref() {
            Some(InitializedEmbeddingsProvider::Ort(embedder)) => embedder
                .chunk_text(content, self.chunking_config.overlap_size)
                .map_err(|_| ChunkingError::EmbeddingTokenizer),
            #[cfg(test)]
            Some(InitializedEmbeddingsProvider::Mock(_)) | None => {
                chunk_text(content, self.chunking_config)
            }
            #[cfg(not(test))]
            None => chunk_text(content, self.chunking_config),
        }
    }
}

enum InitializedDbEngine {
    LanceDb(LanceDbBackend),
}

#[cfg(test)]
struct MockEmbedder {
    dimensions: usize,
    calls: Arc<AtomicUsize>,
}

#[cfg(test)]
impl MockEmbedder {
    fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let call_number = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(texts
            .iter()
            .enumerate()
            .map(|(index, _)| vec![(call_number + index) as f32; self.dimensions])
            .collect())
    }
}

enum InitializedEmbeddingsProvider {
    Ort(embeddings::OrtEmbedder),
    #[cfg(test)]
    Mock(MockEmbedder),
}

impl InitializedEmbeddingsProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        match self {
            Self::Ort(embedder) => embedder.embed_batch_shared(texts).await,
            #[cfg(test)]
            Self::Mock(embedder) => embedder.embed_batch(texts).await,
        }
    }

    fn expected_embedding_size(&self) -> Option<usize> {
        match self {
            Self::Ort(embedder) => embedder.expected_embedding_size(),
            #[cfg(test)]
            Self::Mock(embedder) => Some(embedder.dimensions),
        }
    }
}

struct PreparedDocument {
    document_id: String,
    content: String,
    chunks: Vec<String>,
}

fn read_document_file(path: &Path) -> Result<String, IngestDocumentError> {
    std::fs::read_to_string(path).map_err(|source| IngestDocumentError::FileRead {
        path: path.to_path_buf(),
        source,
    })
}

async fn init_db_engine(db_engine: DbEngine) -> Result<InitializedDbEngine, RKitInitError> {
    match db_engine {
        DbEngine::LanceDb {
            path,
            vector_dimensions,
        } => LanceDbBackend::new(path, vector_dimensions)
            .await
            .map(InitializedDbEngine::LanceDb)
            .map_err(RKitInitError::DbEngine),
    }
}

fn init_embeddings_provider(
    embeddings_provider: EmbeddingsProviderKind,
) -> Result<InitializedEmbeddingsProvider, RKitInitError> {
    match embeddings_provider {
        EmbeddingsProviderKind::Ort(config) => OrtEmbedder::new(config)
            .map(InitializedEmbeddingsProvider::Ort)
            .map_err(RKitInitError::EmbeddingsProvider),
    }
}

fn validate_db_engine(db_engine: &DbEngine) -> Result<(), RKitConfigError> {
    match db_engine {
        DbEngine::LanceDb {
            vector_dimensions, ..
        } if *vector_dimensions <= 0 => {
            Err(RKitConfigError::InvalidVectorDimensions(*vector_dimensions))
        }
        DbEngine::LanceDb { .. } => Ok(()),
    }
}

fn validate_chunking_config(chunking_config: ChunkingConfig) -> Result<(), RKitConfigError> {
    chunk_text("validation", chunking_config)
        .map(|_| ())
        .map_err(RKitConfigError::InvalidChunkingConfig)
}

async fn ensure_db_engine_tables(db_engine: &InitializedDbEngine) -> Result<(), RKitInitError> {
    match db_engine {
        InitializedDbEngine::LanceDb(backend) => backend
            .create_tables()
            .await
            .map(|_| ())
            .map_err(RKitInitError::DbEngine),
    }
}

fn validate_embedding_dimensions(
    db_engine: &InitializedDbEngine,
    embeddings_provider: &InitializedEmbeddingsProvider,
) -> Result<(), RKitInitError> {
    let Some(embedding_dimensions) = embeddings_provider.expected_embedding_size() else {
        return Ok(());
    };

    match db_engine {
        InitializedDbEngine::LanceDb(backend)
            if backend.vector_dimensions() as usize != embedding_dimensions =>
        {
            Err(RKitInitError::EmbeddingDimensionMismatch {
                db_vector_dimensions: backend.vector_dimensions(),
                embedding_dimensions,
            })
        }
        InitializedDbEngine::LanceDb(_) => Ok(()),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchToolArguments {
    query: String,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyToolArguments {}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GetDocumentToolArguments {
    document_id: String,
}

fn parse_tool_arguments<T: DeserializeOwned>(
    tool_name: &str,
    arguments: Value,
) -> Result<T, InvokeToolError> {
    serde_json::from_value(arguments).map_err(|source| InvokeToolError::InvalidArguments {
        tool_name: tool_name.to_string(),
        source,
    })
}

fn serialize_tool_result<T: Serialize>(value: T) -> Result<Value, InvokeToolError> {
    serde_json::to_value(value).map_err(InvokeToolError::Serialization)
}

fn search_tool_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query."
            },
            "limit": {
                "type": "integer",
                "minimum": 0,
                "description": "Maximum number of matching chunks to return. Defaults to 10."
            }
        },
        "required": ["query"],
        "additionalProperties": false
    })
}

fn empty_tool_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {},
        "additionalProperties": false
    })
}

fn get_document_tool_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "The document identifier returned by ingestion or list_documents."
            }
        },
        "required": ["document_id"],
        "additionalProperties": false
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ChunkingConfig, DbEngine, Document, DocumentError, DocumentSummary, EmbeddingError,
        EmbeddingsConfig, EmbeddingsProviderKind, IngestDocumentError,
        InitializedEmbeddingsProvider, InvokeToolError, KeywordSearchError, KeywordSearchResult,
        MockEmbedder, RKit, RKitConfigError, RKitInitError, ToolDescriptions, VectorSearchError,
        VectorSearchResult,
    };
    use arrow_array::{Array, FixedSizeListArray, Float32Array, StringArray};
    use futures::TryStreamExt;
    use lancedb::query::ExecutableQuery;
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::{TempDir, tempdir};

    fn demo_engine(path: &str, vector_dimensions: i32) -> DbEngine {
        DbEngine::LanceDb {
            path: PathBuf::from(path),
            vector_dimensions,
        }
    }

    fn missing_local_ort_provider() -> EmbeddingsProviderKind {
        EmbeddingsProviderKind::Ort(EmbeddingsConfig {
            local_model_path: Some(PathBuf::from("/tmp/retrieval-kit-missing-model.onnx")),
            local_tokenizer_path: Some(PathBuf::from("/tmp/retrieval-kit-missing-tokenizer.json")),
            local_pooling_config_path: Some(PathBuf::from(
                "/tmp/retrieval-kit-missing-pooling-config.json",
            )),
            local_transformer_config_path: Some(PathBuf::from(
                "/tmp/retrieval-kit-missing-transformer-config.json",
            )),
            ..EmbeddingsConfig::default()
        })
    }

    async fn table_document_ids(rkit: &RKit) -> Vec<String> {
        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        let rows = table.query().execute().await.unwrap();
        let batches = rows.try_collect::<Vec<_>>().await.unwrap();

        batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("document_id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .flatten()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    async fn table_texts(rkit: &RKit) -> Vec<String> {
        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        let rows = table.query().execute().await.unwrap();
        let batches = rows.try_collect::<Vec<_>>().await.unwrap();

        batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("text")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .flatten()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    async fn table_vectors(rkit: &RKit) -> Vec<Vec<f32>> {
        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        let rows = table.query().execute().await.unwrap();
        let batches = rows.try_collect::<Vec<_>>().await.unwrap();

        batches
            .iter()
            .flat_map(|batch| {
                let vectors = batch
                    .column_by_name("vector")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .unwrap();

                (0..vectors.len())
                    .map(|index| {
                        vectors
                            .value(index)
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .unwrap()
                            .values()
                            .to_vec()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    async fn table_stored_documents(rkit: &RKit) -> Vec<Document> {
        let backend = rkit.lancedb_backend().unwrap();
        let mut documents = backend
            .list_documents()
            .await
            .unwrap()
            .into_iter()
            .map(|document| Document {
                document_id: document.document_id,
                content: document.content,
            })
            .collect::<Vec<_>>();
        documents.sort_by(|left, right| left.document_id.cmp(&right.document_id));
        documents
    }

    async fn initialized_test_rkit(vector_dimensions: i32) -> (TempDir, RKit) {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), vector_dimensions),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(
                temp_dir.path().to_str().unwrap(),
                vector_dimensions,
            ))
            .await
            .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(
            vector_dimensions as usize,
        )));
        (temp_dir, rkit)
    }

    #[test]
    fn get_tool_definitions_returns_default_mcp_compatible_definitions() {
        let rkit = RKit::new(
            demo_engine("/tmp/rkit-a", 384),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let definitions = rkit.get_tool_definitions(None);

        assert_eq!(definitions.len(), 4);
        assert_eq!(definitions[0].name, "semantic_search");
        assert_eq!(definitions[1].name, "keyword_search");
        assert_eq!(definitions[2].name, "list_documents");
        assert_eq!(definitions[3].name, "get_document");
        assert_eq!(
            definitions[0].description.as_deref(),
            Some("Search ingested documents by semantic similarity to a natural language query.")
        );
        assert_eq!(definitions[0].input_schema["required"], json!(["query"]));
        assert_eq!(
            definitions[0].input_schema["additionalProperties"],
            json!(false)
        );
        assert_eq!(definitions[2].input_schema["properties"], json!({}));
        assert_eq!(
            definitions[3].input_schema["required"],
            json!(["document_id"])
        );
    }

    #[test]
    fn get_tool_definitions_applies_description_overrides_selectively() {
        let rkit = RKit::new(
            demo_engine("/tmp/rkit-a", 384),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let definitions = rkit.get_tool_definitions(Some(ToolDescriptions {
            semantic_search: Some("Custom semantic search".to_string()),
            get_document: Some("Custom document fetch".to_string()),
            ..ToolDescriptions::default()
        }));

        assert_eq!(
            definitions[0].description.as_deref(),
            Some("Custom semantic search")
        );
        assert_eq!(
            definitions[1].description.as_deref(),
            Some("Search ingested documents by exact keyword and full-text relevance.")
        );
        assert_eq!(
            definitions[3].description.as_deref(),
            Some("Custom document fetch")
        );
    }

    #[tokio::test]
    async fn invoke_tool_dispatches_document_tools() {
        let (_temp_dir, mut rkit) = initialized_test_rkit(3).await;
        rkit.upsert_document("manual-doc".to_string(), "Document content.".to_string())
            .await
            .unwrap();

        let list_result = rkit.invoke_tool("list_documents", json!({})).await.unwrap();
        let get_result = rkit
            .invoke_tool("get_document", json!({ "document_id": "manual-doc" }))
            .await
            .unwrap();
        let missing_result = rkit
            .invoke_tool("get_document", json!({ "document_id": "missing-doc" }))
            .await
            .unwrap();

        assert_eq!(
            list_result,
            json!({ "documents": [{ "document_id": "manual-doc" }] })
        );
        assert_eq!(
            get_result,
            json!({
                "document": {
                    "document_id": "manual-doc",
                    "content": "Document content."
                }
            })
        );
        assert_eq!(missing_result, json!({ "document": null }));
    }

    #[tokio::test]
    async fn invoke_tool_dispatches_search_tools() {
        let (_temp_dir, mut rkit) = initialized_test_rkit(3).await;
        rkit.upsert_document(
            "first-doc".to_string(),
            "First document content.".to_string(),
        )
        .await
        .unwrap();
        rkit.upsert_document(
            "keyword-doc".to_string(),
            "Rust search database.".to_string(),
        )
        .await
        .unwrap();

        let semantic_result = rkit
            .invoke_tool(
                "semantic_search",
                json!({ "query": "content like the second document", "limit": 1 }),
            )
            .await
            .unwrap();
        let keyword_result = rkit
            .invoke_tool("keyword_search", json!({ "query": "rust", "limit": 1 }))
            .await
            .unwrap();

        let semantic_results: Vec<VectorSearchResult> =
            serde_json::from_value(semantic_result["results"].clone()).unwrap();
        let keyword_results: Vec<KeywordSearchResult> =
            serde_json::from_value(keyword_result["results"].clone()).unwrap();

        assert_eq!(semantic_results.len(), 1);
        assert_eq!(semantic_results[0].document_id, "keyword-doc");
        assert_eq!(keyword_results.len(), 1);
        assert_eq!(keyword_results[0].document_id, "keyword-doc");
        assert_eq!(keyword_results[0].text, "Rust search database.");
    }

    #[tokio::test]
    async fn invoke_tool_uses_default_search_limit() {
        let (_temp_dir, mut rkit) = initialized_test_rkit(3).await;
        for index in 0..11 {
            rkit.upsert_document(format!("doc-{index:02}"), format!("Document {index}."))
                .await
                .unwrap();
        }

        let result = rkit
            .invoke_tool("semantic_search", json!({ "query": "document" }))
            .await
            .unwrap();

        assert_eq!(result["results"].as_array().unwrap().len(), 10);
    }

    #[tokio::test]
    async fn invoke_tool_reports_bad_calls_as_errors() {
        let (_temp_dir, rkit) = initialized_test_rkit(3).await;

        let unknown = rkit
            .invoke_tool("missing_tool", json!({}))
            .await
            .unwrap_err();
        let invalid = rkit
            .invoke_tool("semantic_search", json!({ "limit": 1 }))
            .await
            .unwrap_err();
        let extra_argument = rkit
            .invoke_tool("list_documents", json!({ "unexpected": true }))
            .await
            .unwrap_err();

        assert!(matches!(unknown, InvokeToolError::UnknownTool(name) if name == "missing_tool"));
        assert!(matches!(
            invalid,
            InvokeToolError::InvalidArguments { tool_name, .. } if tool_name == "semantic_search"
        ));
        assert!(matches!(
            extra_argument,
            InvokeToolError::InvalidArguments { tool_name, .. } if tool_name == "list_documents"
        ));
    }

    #[test]
    fn new_stores_both_configs() {
        let db_engine = demo_engine("/tmp/rkit-a", 384);
        let embeddings_provider = EmbeddingsProviderKind::Ort(EmbeddingsConfig::default());

        let rkit = RKit::new(db_engine.clone(), embeddings_provider.clone()).unwrap();

        assert_eq!(rkit.db_engine_config(), &db_engine);
        assert_eq!(rkit.embeddings_provider_config(), &embeddings_provider);
        assert_eq!(rkit.chunking_config(), ChunkingConfig::default());
    }

    #[test]
    fn select_db_engine_accepts_valid_lancedb_config() {
        let mut rkit = RKit::new(
            demo_engine("/tmp/rkit-a", 384),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        let next_engine = demo_engine("/tmp/rkit-b", 768);

        rkit.select_db_engine(next_engine.clone()).unwrap();

        assert_eq!(rkit.db_engine_config(), &next_engine);
    }

    #[test]
    fn select_db_engine_rejects_non_positive_vector_dimensions() {
        let mut rkit = RKit::new(
            demo_engine("/tmp/rkit-a", 384),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let error = rkit
            .select_db_engine(demo_engine("/tmp/rkit-b", 0))
            .unwrap_err();

        assert_eq!(error, RKitConfigError::InvalidVectorDimensions(0));
    }

    #[test]
    fn register_embeddings_provider_accepts_default_ort_config() {
        let mut rkit = RKit::new(
            demo_engine("/tmp/rkit-a", 384),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        let provider = EmbeddingsProviderKind::Ort(EmbeddingsConfig::default());

        rkit.register_embeddings_provider(provider.clone()).unwrap();

        assert_eq!(rkit.embeddings_provider_config(), &provider);
    }

    #[test]
    fn reselecting_provider_replaces_prior_config() {
        let mut rkit = RKit::new(
            demo_engine("/tmp/rkit-a", 384),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        let provider = EmbeddingsProviderKind::Ort(EmbeddingsConfig {
            normalize: false,
            max_length: 64,
            ..EmbeddingsConfig::default()
        });

        rkit.register_embeddings_provider(provider.clone()).unwrap();

        assert_eq!(rkit.embeddings_provider_config(), &provider);
    }

    #[test]
    fn set_chunking_config_validates_values() {
        let mut rkit = RKit::new(
            demo_engine("/tmp/rkit-a", 384),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        let chunking_config = ChunkingConfig {
            chunk_size: 128,
            overlap_size: 16,
        };

        rkit.set_chunking_config(chunking_config).unwrap();

        assert_eq!(rkit.chunking_config(), chunking_config);
    }

    #[test]
    fn set_chunking_config_rejects_invalid_values() {
        let mut rkit = RKit::new(
            demo_engine("/tmp/rkit-a", 384),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let error = rkit
            .set_chunking_config(ChunkingConfig {
                chunk_size: 32,
                overlap_size: 32,
            })
            .unwrap_err();

        assert!(matches!(error, RKitConfigError::InvalidChunkingConfig(_)));
    }

    #[tokio::test]
    async fn init_db_engine_initializes_lancedb_backend() {
        let temp_dir = tempdir().unwrap();
        let initialized_db_engine =
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 384))
                .await
                .unwrap();

        super::ensure_db_engine_tables(&initialized_db_engine)
            .await
            .unwrap();

        match initialized_db_engine {
            super::InitializedDbEngine::LanceDb(backend) => {
                assert_eq!(backend.vector_dimensions(), 384);
                let table = backend
                    .connection()
                    .open_table("chunks")
                    .execute()
                    .await
                    .unwrap();
                let documents_table = backend
                    .connection()
                    .open_table("documents")
                    .execute()
                    .await
                    .unwrap();
                assert_eq!(documents_table.count_rows(None).await.unwrap(), 0);
                assert_eq!(table.count_rows(None).await.unwrap(), 0);
            }
        }
    }

    #[tokio::test]
    async fn init_is_atomic_when_embeddings_provider_initialization_fails() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 384),
            missing_local_ort_provider(),
        )
        .unwrap();

        let error = rkit.init().await.unwrap_err();

        assert!(matches!(
            error,
            RKitInitError::EmbeddingsProvider(EmbeddingError::MissingAsset { .. })
        ));
        assert!(rkit.lancedb_backend().is_none());
        assert!(rkit.ort_embedder().is_none());
    }

    #[tokio::test]
    async fn init_dimension_validation_rejects_mismatched_embedder() {
        let temp_dir = tempdir().unwrap();
        let db_engine = super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 5))
            .await
            .unwrap();
        let embeddings_provider = InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3));

        let error =
            super::validate_embedding_dimensions(&db_engine, &embeddings_provider).unwrap_err();

        assert!(matches!(
            error,
            RKitInitError::EmbeddingDimensionMismatch {
                db_vector_dimensions: 5,
                embedding_dimensions: 3
            }
        ));
    }

    #[tokio::test]
    async fn selecting_new_db_engine_clears_initialized_backend() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 384),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 384))
                .await
                .unwrap(),
        );

        assert!(rkit.db_engine_initialized());

        rkit.select_db_engine(demo_engine("/tmp/rkit-b", 768))
            .unwrap();

        assert!(!rkit.db_engine_initialized());
    }

    #[tokio::test]
    async fn ingest_document_requires_initialization() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let error = rkit
            .ingest_document("A short document".to_string())
            .await
            .unwrap_err();

        assert!(matches!(error, IngestDocumentError::NotInitialized));
    }

    #[tokio::test]
    async fn ingest_document_rejects_empty_content() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let error = rkit.ingest_document("   ".to_string()).await.unwrap_err();

        assert!(matches!(error, IngestDocumentError::EmptyContent));
    }

    #[tokio::test]
    async fn ingest_document_file_requires_initialization_after_reading() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        let file_path = temp_dir.path().join("document.txt");
        fs::write(&file_path, "A short document").unwrap();

        let error = rkit.ingest_document_file(&file_path).await.unwrap_err();

        assert!(matches!(error, IngestDocumentError::NotInitialized));
    }

    #[tokio::test]
    async fn ingest_document_file_reads_and_inserts_text_file() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.set_chunking_config(ChunkingConfig {
            chunk_size: 24,
            overlap_size: 4,
        })
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let content = "First sentence has enough text to force chunking. Second sentence adds more words for another chunk.";
        let file_path = temp_dir.path().join("document.txt");
        fs::write(&file_path, content).unwrap();
        let expected_chunk_count = super::chunk_text(content, rkit.chunking_config())
            .unwrap()
            .len();

        let result = rkit.ingest_document_file(&file_path).await.unwrap();

        assert_eq!(result.chunk_count, expected_chunk_count);
        assert!(!result.document_id.is_empty());
        assert_eq!(
            rkit.get_document(result.document_id.clone()).await.unwrap(),
            Some(Document {
                document_id: result.document_id,
                content: content.to_string(),
            })
        );

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        assert_eq!(
            table.count_rows(None).await.unwrap(),
            expected_chunk_count as usize
        );
    }

    #[tokio::test]
    async fn ingest_document_file_returns_file_read_error_for_missing_path() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        let file_path = temp_dir.path().join("missing.txt");

        let error = rkit.ingest_document_file(&file_path).await.unwrap_err();

        assert!(matches!(
            error,
            IngestDocumentError::FileRead { path, .. } if path == file_path
        ));
    }

    #[tokio::test]
    async fn ingest_document_file_returns_file_read_error_for_invalid_utf8() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        let file_path = temp_dir.path().join("invalid.txt");
        fs::write(&file_path, [0xff, 0xfe, 0xfd]).unwrap();

        let error = rkit.ingest_document_file(&file_path).await.unwrap_err();

        assert!(matches!(
            error,
            IngestDocumentError::FileRead { path, .. } if path == file_path
        ));
    }

    #[tokio::test]
    async fn ingest_document_chunks_embeds_and_inserts_rows() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.set_chunking_config(ChunkingConfig {
            chunk_size: 24,
            overlap_size: 4,
        })
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let content = "First sentence has enough text to force chunking. Second sentence adds more words for another chunk.".to_string();
        let expected_chunk_count = super::chunk_text(&content, rkit.chunking_config())
            .unwrap()
            .len();

        let result = rkit.ingest_document(content.clone()).await.unwrap();

        assert_eq!(result.chunk_count, expected_chunk_count);
        assert!(!result.document_id.is_empty());
        assert_eq!(
            rkit.get_document(result.document_id.clone()).await.unwrap(),
            Some(Document {
                document_id: result.document_id.clone(),
                content,
            })
        );

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        assert_eq!(
            table.count_rows(None).await.unwrap(),
            expected_chunk_count as usize
        );

        let rows = table.query().execute().await.unwrap();
        let batches = rows.try_collect::<Vec<_>>().await.unwrap();
        let document_ids = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("document_id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .flatten()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(document_ids.len(), expected_chunk_count);
        assert!(document_ids.iter().all(|id| id == &result.document_id));
    }

    #[tokio::test]
    async fn list_documents_requires_initialization() {
        let temp_dir = tempdir().unwrap();
        let rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let error = rkit.list_documents().await.unwrap_err();

        assert!(matches!(error, DocumentError::NotInitialized));
    }

    #[tokio::test]
    async fn get_document_requires_initialization() {
        let temp_dir = tempdir().unwrap();
        let rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let error = rkit
            .get_document("missing-doc".to_string())
            .await
            .unwrap_err();

        assert!(matches!(error, DocumentError::NotInitialized));
    }

    #[tokio::test]
    async fn delete_document_requires_initialization() {
        let temp_dir = tempdir().unwrap();
        let rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let error = rkit
            .delete_document("missing-doc".to_string())
            .await
            .unwrap_err();

        assert!(matches!(error, DocumentError::NotInitialized));
    }

    #[tokio::test]
    async fn vector_search_requires_initialization() {
        let temp_dir = tempdir().unwrap();
        let rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let error = rkit
            .vector_search("meaningful query".to_string(), 3)
            .await
            .unwrap_err();

        assert!(matches!(error, VectorSearchError::NotInitialized));
    }

    #[tokio::test]
    async fn vector_search_rejects_blank_query() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let error = rkit.vector_search("   ".to_string(), 3).await.unwrap_err();

        assert!(matches!(error, VectorSearchError::EmptyQuery));
    }

    #[tokio::test]
    async fn vector_search_zero_limit_returns_empty_without_embedding() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let results = rkit
            .vector_search("meaningful query".to_string(), 0)
            .await
            .unwrap();

        assert!(results.is_empty());
        match rkit.embeddings_provider.as_ref().unwrap() {
            InitializedEmbeddingsProvider::Mock(embedder) => assert_eq!(embedder.call_count(), 0),
            InitializedEmbeddingsProvider::Ort(_) => unreachable!(),
        }
    }

    #[tokio::test]
    async fn vector_search_embeds_query_and_returns_limited_ranked_chunks() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));
        let first = rkit
            .upsert_document(
                "first-doc".to_string(),
                "First document content.".to_string(),
            )
            .await
            .unwrap();
        let second = rkit
            .upsert_document(
                "second-doc".to_string(),
                "Second document content.".to_string(),
            )
            .await
            .unwrap();

        let results = rkit
            .vector_search("content like the second document".to_string(), 1)
            .await
            .unwrap();

        assert_eq!(
            results,
            vec![VectorSearchResult {
                document_id: second.document_id,
                text: "Second document content.".to_string(),
                distance: 3.0,
            }]
        );
        assert_eq!(first.chunk_count, 1);
        match rkit.embeddings_provider.as_ref().unwrap() {
            InitializedEmbeddingsProvider::Mock(embedder) => assert_eq!(embedder.call_count(), 3),
            InitializedEmbeddingsProvider::Ort(_) => unreachable!(),
        }
    }

    #[tokio::test]
    async fn keyword_search_requires_initialization() {
        let temp_dir = tempdir().unwrap();
        let rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let error = rkit
            .keyword_search("meaningful query".to_string(), 3)
            .await
            .unwrap_err();

        assert!(matches!(error, KeywordSearchError::NotInitialized));
    }

    #[tokio::test]
    async fn keyword_search_rejects_blank_query() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let error = rkit.keyword_search("   ".to_string(), 3).await.unwrap_err();

        assert!(matches!(error, KeywordSearchError::EmptyQuery));
    }

    #[tokio::test]
    async fn keyword_search_zero_limit_returns_empty_without_embedding() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let results = rkit
            .keyword_search("meaningful query".to_string(), 0)
            .await
            .unwrap();

        assert!(results.is_empty());
        match rkit.embeddings_provider.as_ref().unwrap() {
            InitializedEmbeddingsProvider::Mock(embedder) => assert_eq!(embedder.call_count(), 0),
            InitializedEmbeddingsProvider::Ort(_) => unreachable!(),
        }
    }

    #[tokio::test]
    async fn keyword_search_returns_limited_ranked_chunks() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));
        let result = rkit
            .upsert_document(
                "keyword-doc".to_string(),
                "Rust search database. Plain ranger path.".to_string(),
            )
            .await
            .unwrap();

        let results = rkit.keyword_search("rust".to_string(), 1).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, result.document_id);
        assert_eq!(results[0].text, "Rust search database. Plain ranger path.");
        assert!(results[0].score > 0.0);
        match rkit.embeddings_provider.as_ref().unwrap() {
            InitializedEmbeddingsProvider::Mock(embedder) => assert_eq!(embedder.call_count(), 1),
            InitializedEmbeddingsProvider::Ort(_) => unreachable!(),
        }
    }

    #[tokio::test]
    async fn ingest_document_propagates_embedding_dimension_mismatch() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 5),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 5))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let error = rkit
            .ingest_document("A document with one chunk.".to_string())
            .await
            .unwrap_err();

        assert!(matches!(error, IngestDocumentError::DbEngine(_)));

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 0);
        assert!(table_stored_documents(&rkit).await.is_empty());
    }

    #[tokio::test]
    async fn repeated_ingest_document_calls_append_distinct_document_ids() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let first = rkit
            .ingest_document("First document content.".to_string())
            .await
            .unwrap();
        let second = rkit
            .ingest_document("Second document content.".to_string())
            .await
            .unwrap();

        assert_ne!(first.document_id, second.document_id);

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        assert_eq!(
            table.count_rows(None).await.unwrap(),
            first.chunk_count + second.chunk_count
        );
    }

    #[tokio::test]
    async fn list_and_get_documents_return_stored_full_content() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let first = rkit
            .upsert_document(
                "b-doc".to_string(),
                "First full document content.".to_string(),
            )
            .await
            .unwrap();
        let second_content = "Second full document content.".to_string();
        let second = rkit
            .upsert_document("a-doc".to_string(), second_content.clone())
            .await
            .unwrap();

        assert_eq!(
            rkit.list_documents().await.unwrap(),
            vec![
                DocumentSummary {
                    document_id: second.document_id.clone(),
                },
                DocumentSummary {
                    document_id: first.document_id,
                },
            ]
        );
        assert_eq!(
            rkit.get_document(second.document_id.clone()).await.unwrap(),
            Some(Document {
                document_id: second.document_id,
                content: second_content,
            })
        );
        assert_eq!(
            rkit.get_document("missing-doc".to_string()).await.unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn delete_document_removes_stored_content_and_chunks() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));
        rkit.upsert_document("manual-doc".to_string(), "Document content.".to_string())
            .await
            .unwrap();

        rkit.delete_document("manual-doc".to_string())
            .await
            .unwrap();
        rkit.delete_document("missing-doc".to_string())
            .await
            .unwrap();

        assert_eq!(rkit.list_documents().await.unwrap(), Vec::new());
        assert_eq!(table_document_ids(&rkit).await, Vec::<String>::new());
    }

    #[tokio::test]
    async fn upsert_document_inserts_when_document_id_is_new() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let result = rkit
            .upsert_document(
                "manual-doc".to_string(),
                "New document content.".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(result.document_id, "manual-doc");
        assert_eq!(result.chunk_count, 1);
        assert_eq!(
            rkit.get_document("manual-doc".to_string()).await.unwrap(),
            Some(Document {
                document_id: "manual-doc".to_string(),
                content: "New document content.".to_string(),
            })
        );
        assert_eq!(
            table_document_ids(&rkit).await,
            vec!["manual-doc".to_string()]
        );
    }

    #[tokio::test]
    async fn upsert_document_replaces_existing_chunks_for_document_id() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.set_chunking_config(ChunkingConfig {
            chunk_size: 24,
            overlap_size: 4,
        })
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        rkit.upsert_document(
            "manual-doc".to_string(),
            "First sentence has enough text to force chunking. Second sentence adds more words."
                .to_string(),
        )
        .await
        .unwrap();
        let replacement = "Short replacement.".to_string();
        let result = rkit
            .upsert_document("manual-doc".to_string(), replacement.clone())
            .await
            .unwrap();

        assert_eq!(result.document_id, "manual-doc");
        assert_eq!(result.chunk_count, 1);
        assert_eq!(
            table_document_ids(&rkit).await,
            vec!["manual-doc".to_string()]
        );
        assert_eq!(table_texts(&rkit).await, vec![replacement.clone()]);
        assert_eq!(
            rkit.get_document("manual-doc".to_string()).await.unwrap(),
            Some(Document {
                document_id: "manual-doc".to_string(),
                content: replacement,
            })
        );
    }

    #[tokio::test]
    async fn upsert_document_regenerates_embeddings_for_replacement_content() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        rkit.upsert_document("manual-doc".to_string(), "Original content.".to_string())
            .await
            .unwrap();
        assert_eq!(table_vectors(&rkit).await, vec![vec![1.0, 1.0, 1.0]]);

        rkit.upsert_document("manual-doc".to_string(), "Replacement content.".to_string())
            .await
            .unwrap();

        assert_eq!(table_vectors(&rkit).await, vec![vec![2.0, 2.0, 2.0]]);
        match rkit.embeddings_provider.as_ref().unwrap() {
            InitializedEmbeddingsProvider::Mock(embedder) => assert_eq!(embedder.call_count(), 2),
            InitializedEmbeddingsProvider::Ort(_) => unreachable!(),
        }
    }

    #[tokio::test]
    async fn upsert_document_rejects_empty_content_without_mutating_rows() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));
        rkit.upsert_document("manual-doc".to_string(), "Original content.".to_string())
            .await
            .unwrap();

        let error = rkit
            .upsert_document("manual-doc".to_string(), "   ".to_string())
            .await
            .unwrap_err();

        assert!(matches!(error, IngestDocumentError::EmptyContent));
        assert_eq!(
            table_texts(&rkit).await,
            vec!["Original content.".to_string()]
        );
        assert_eq!(
            rkit.get_document("manual-doc".to_string()).await.unwrap(),
            Some(Document {
                document_id: "manual-doc".to_string(),
                content: "Original content.".to_string(),
            })
        );
    }

    #[tokio::test]
    async fn upsert_document_preserves_existing_rows_when_vectors_are_invalid() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 5),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 5))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(5)));
        rkit.upsert_document("manual-doc".to_string(), "Original content.".to_string())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let error = rkit
            .upsert_document("manual-doc".to_string(), "Replacement content.".to_string())
            .await
            .unwrap_err();

        assert!(matches!(error, IngestDocumentError::DbEngine(_)));
        assert_eq!(
            table_texts(&rkit).await,
            vec!["Original content.".to_string()]
        );
        assert_eq!(
            rkit.get_document("manual-doc".to_string()).await.unwrap(),
            Some(Document {
                document_id: "manual-doc".to_string(),
                content: "Original content.".to_string(),
            })
        );
    }

    #[tokio::test]
    async fn ingest_documents_requires_initialization() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let error = rkit
            .ingest_documents(vec!["A short document".to_string()])
            .await
            .unwrap_err();

        assert!(matches!(error, IngestDocumentError::NotInitialized));
    }

    #[tokio::test]
    async fn ingest_documents_rejects_empty_batch() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let error = rkit.ingest_documents(Vec::new()).await.unwrap_err();

        assert!(matches!(error, IngestDocumentError::EmptyContent));
    }

    #[tokio::test]
    async fn ingest_document_files_reads_multiple_files_from_glob() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.set_chunking_config(ChunkingConfig {
            chunk_size: 24,
            overlap_size: 4,
        })
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let first_content =
            "First sentence has enough text to force chunking. Second sentence adds more words.";
        let second_content = "Another document with enough content to create at least one chunk.";
        fs::write(temp_dir.path().join("a.txt"), first_content).unwrap();
        fs::write(temp_dir.path().join("b.txt"), second_content).unwrap();
        let expected_first = super::chunk_text(first_content, rkit.chunking_config())
            .unwrap()
            .len();
        let expected_second = super::chunk_text(second_content, rkit.chunking_config())
            .unwrap()
            .len();
        let pattern = format!("{}/*.txt", temp_dir.path().display());

        let results = rkit.ingest_document_files(&pattern).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_count, expected_first);
        assert_eq!(results[1].chunk_count, expected_second);
        assert_ne!(results[0].document_id, results[1].document_id);

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        assert_eq!(
            table.count_rows(None).await.unwrap(),
            expected_first + expected_second
        );
    }

    #[tokio::test]
    async fn ingest_document_files_rejects_invalid_glob_pattern() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();

        let error = rkit.ingest_document_files("[").await.unwrap_err();

        assert!(matches!(error, IngestDocumentError::InvalidGlobPattern(_)));
    }

    #[tokio::test]
    async fn ingest_document_files_rejects_when_glob_matches_nothing() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        let pattern = format!("{}/*.txt", temp_dir.path().display());

        let error = rkit.ingest_document_files(&pattern).await.unwrap_err();

        assert!(matches!(
            error,
            IngestDocumentError::NoFilesMatched { pattern: actual } if actual == pattern
        ));
    }

    #[tokio::test]
    async fn ingest_document_files_prevalidates_all_matched_files_before_inserting() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        fs::write(temp_dir.path().join("valid.txt"), "valid document").unwrap();
        fs::create_dir(temp_dir.path().join("invalid.txt")).unwrap();
        let pattern = format!("{}/*.txt", temp_dir.path().display());

        let error = rkit.ingest_document_files(&pattern).await.unwrap_err();

        assert!(matches!(error, IngestDocumentError::FileRead { .. }));

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 0);
        assert!(table_stored_documents(&rkit).await.is_empty());
    }

    #[tokio::test]
    async fn ingest_documents_prevalidates_all_inputs_before_inserting() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let error = rkit
            .ingest_documents(vec!["valid document".to_string(), "   ".to_string()])
            .await
            .unwrap_err();

        assert!(matches!(error, IngestDocumentError::EmptyContent));

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 0);
        assert!(table_stored_documents(&rkit).await.is_empty());
    }

    #[tokio::test]
    async fn ingest_documents_returns_one_result_per_input_document() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 3),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.set_chunking_config(ChunkingConfig {
            chunk_size: 24,
            overlap_size: 4,
        })
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 3))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let first_content =
            "First sentence has enough text to force chunking. Second sentence adds more words."
                .to_string();
        let second_content =
            "Another document with enough content to create at least one chunk.".to_string();
        let expected_first = super::chunk_text(&first_content, rkit.chunking_config())
            .unwrap()
            .len();
        let expected_second = super::chunk_text(&second_content, rkit.chunking_config())
            .unwrap()
            .len();

        let results = rkit
            .ingest_documents(vec![first_content.clone(), second_content.clone()])
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_count, expected_first);
        assert_eq!(results[1].chunk_count, expected_second);
        assert_ne!(results[0].document_id, results[1].document_id);
        assert_eq!(
            rkit.get_document(results[0].document_id.clone())
                .await
                .unwrap(),
            Some(Document {
                document_id: results[0].document_id.clone(),
                content: first_content,
            })
        );
        assert_eq!(
            rkit.get_document(results[1].document_id.clone())
                .await
                .unwrap(),
            Some(Document {
                document_id: results[1].document_id.clone(),
                content: second_content,
            })
        );

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        assert_eq!(
            table.count_rows(None).await.unwrap(),
            expected_first + expected_second
        );

        let rows = table.query().execute().await.unwrap();
        let batches = rows.try_collect::<Vec<_>>().await.unwrap();
        let document_ids = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("document_id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .flatten()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let first_count = document_ids
            .iter()
            .filter(|id| *id == &results[0].document_id)
            .count();
        let second_count = document_ids
            .iter()
            .filter(|id| *id == &results[1].document_id)
            .count();

        assert_eq!(first_count, expected_first);
        assert_eq!(second_count, expected_second);
    }

    #[tokio::test]
    async fn ingest_documents_propagates_embedding_dimension_mismatch_without_inserting() {
        let temp_dir = tempdir().unwrap();
        let mut rkit = RKit::new(
            demo_engine(temp_dir.path().to_str().unwrap(), 5),
            EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
        )
        .unwrap();
        rkit.db_engine = Some(
            super::init_db_engine(demo_engine(temp_dir.path().to_str().unwrap(), 5))
                .await
                .unwrap(),
        );
        super::ensure_db_engine_tables(rkit.db_engine.as_ref().unwrap())
            .await
            .unwrap();
        rkit.embeddings_provider = Some(InitializedEmbeddingsProvider::Mock(MockEmbedder::new(3)));

        let error = rkit
            .ingest_documents(vec![
                "First document content.".to_string(),
                "Second document content.".to_string(),
            ])
            .await
            .unwrap_err();

        assert!(matches!(error, IngestDocumentError::DbEngine(_)));

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("chunks")
            .execute()
            .await
            .unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 0);
    }
}
