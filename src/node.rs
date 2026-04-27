use std::path::PathBuf;

use napi::bindgen_prelude::Result;
use napi::{Error, Status};
use napi_derive::napi;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::{
    ChunkingConfig, DbEngine, Document, DocumentSummary, EmbeddingsConfig, EmbeddingsProviderKind,
    IngestDocumentResult, KeywordSearchResult, RKit as InnerRKit, ToolDefinition, ToolDescriptions,
    VectorSearchResult,
};

#[napi(object)]
pub struct RKitOptions {
    pub lance_db_path: String,
    pub vector_dimensions: i32,
    pub embedding: Option<OrtEmbeddingOptions>,
    pub chunking: Option<NodeChunkingOptions>,
}

#[napi(object)]
pub struct OrtEmbeddingOptions {
    pub model_repo: Option<String>,
    pub model_revision: Option<String>,
    pub model_file: Option<String>,
    pub tokenizer_file: Option<String>,
    pub pooling_config_file: Option<String>,
    pub transformer_config_file: Option<String>,
    pub max_length: Option<u32>,
    pub normalize: Option<bool>,
    pub intra_threads: Option<u32>,
    pub cache_dir: Option<String>,
    pub local_model_path: Option<String>,
    pub local_tokenizer_path: Option<String>,
    pub local_pooling_config_path: Option<String>,
    pub local_transformer_config_path: Option<String>,
    pub input_ids_name: Option<String>,
    pub attention_mask_name: Option<String>,
    pub token_type_ids_name: Option<String>,
    pub output_name: Option<String>,
}

#[napi(object)]
pub struct NodeChunkingOptions {
    pub chunk_size: Option<u32>,
    pub overlap_size: Option<u32>,
}

#[napi(object)]
pub struct NodeIngestDocumentResult {
    pub document_id: String,
    pub chunk_count: u32,
}

#[napi(object)]
pub struct NodeDocumentSummary {
    pub document_id: String,
}

#[napi(object)]
pub struct NodeDocument {
    pub document_id: String,
    pub content: String,
}

#[napi(object)]
pub struct NodeVectorSearchResult {
    pub document_id: String,
    pub text: String,
    pub distance: f64,
}

#[napi(object)]
pub struct NodeKeywordSearchResult {
    pub document_id: String,
    pub text: String,
    pub score: f64,
}

#[napi(object)]
pub struct NodeToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

#[napi(object)]
pub struct NodeToolDescriptions {
    pub semantic_search: Option<String>,
    pub keyword_search: Option<String>,
    pub list_documents: Option<String>,
    pub get_document: Option<String>,
}

#[napi(js_name = "RKit")]
pub struct NodeRKit {
    inner: Mutex<InnerRKit>,
}

#[napi]
impl NodeRKit {
    #[napi(constructor)]
    pub fn new(options: RKitOptions) -> Result<Self> {
        let db_engine = DbEngine::LanceDb {
            path: PathBuf::from(options.lance_db_path),
            vector_dimensions: options.vector_dimensions,
        };
        let embeddings_provider =
            EmbeddingsProviderKind::Ort(embedding_config_from_options(options.embedding));
        let mut inner = InnerRKit::new(db_engine, embeddings_provider).map_err(to_napi_error)?;

        if let Some(chunking) = options.chunking {
            inner
                .set_chunking_config(ChunkingConfig {
                    chunk_size: option_u32_to_usize(chunking.chunk_size)
                        .unwrap_or_else(|| inner.chunking_config().chunk_size),
                    overlap_size: option_u32_to_usize(chunking.overlap_size)
                        .unwrap_or_else(|| inner.chunking_config().overlap_size),
                })
                .map_err(to_napi_error)?;
        }

        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[napi]
    pub async fn init(&self) -> Result<()> {
        self.inner.lock().await.init().await.map_err(to_napi_error)
    }

    #[napi]
    pub async fn ingest_document(&self, content: String) -> Result<NodeIngestDocumentResult> {
        self.inner
            .lock()
            .await
            .ingest_document(content)
            .await
            .map(NodeIngestDocumentResult::from)
            .map_err(to_napi_error)
    }

    #[napi]
    pub async fn ingest_document_file(&self, path: String) -> Result<NodeIngestDocumentResult> {
        self.inner
            .lock()
            .await
            .ingest_document_file(path)
            .await
            .map(NodeIngestDocumentResult::from)
            .map_err(to_napi_error)
    }

    #[napi]
    pub async fn ingest_document_files(
        &self,
        pattern: String,
    ) -> Result<Vec<NodeIngestDocumentResult>> {
        self.inner
            .lock()
            .await
            .ingest_document_files(&pattern)
            .await
            .map(|results| {
                results
                    .into_iter()
                    .map(NodeIngestDocumentResult::from)
                    .collect()
            })
            .map_err(to_napi_error)
    }

    #[napi]
    pub async fn upsert_document(
        &self,
        id: String,
        content: String,
    ) -> Result<NodeIngestDocumentResult> {
        self.inner
            .lock()
            .await
            .upsert_document(id, content)
            .await
            .map(NodeIngestDocumentResult::from)
            .map_err(to_napi_error)
    }

    #[napi]
    pub async fn vector_search(
        &self,
        query: String,
        limit: u32,
    ) -> Result<Vec<NodeVectorSearchResult>> {
        self.inner
            .lock()
            .await
            .vector_search(query, limit as usize)
            .await
            .map(|results| {
                results
                    .into_iter()
                    .map(NodeVectorSearchResult::from)
                    .collect()
            })
            .map_err(to_napi_error)
    }

    #[napi]
    pub async fn keyword_search(
        &self,
        query: String,
        limit: u32,
    ) -> Result<Vec<NodeKeywordSearchResult>> {
        self.inner
            .lock()
            .await
            .keyword_search(query, limit as usize)
            .await
            .map(|results| {
                results
                    .into_iter()
                    .map(NodeKeywordSearchResult::from)
                    .collect()
            })
            .map_err(to_napi_error)
    }

    #[napi]
    pub async fn list_documents(&self) -> Result<Vec<NodeDocumentSummary>> {
        self.inner
            .lock()
            .await
            .list_documents()
            .await
            .map(|documents| {
                documents
                    .into_iter()
                    .map(NodeDocumentSummary::from)
                    .collect()
            })
            .map_err(to_napi_error)
    }

    #[napi]
    pub async fn get_document(&self, id: String) -> Result<Option<NodeDocument>> {
        self.inner
            .lock()
            .await
            .get_document(id)
            .await
            .map(|document| document.map(NodeDocument::from))
            .map_err(to_napi_error)
    }

    #[napi]
    pub async fn delete_document(&self, id: String) -> Result<()> {
        self.inner
            .lock()
            .await
            .delete_document(id)
            .await
            .map_err(to_napi_error)
    }

    #[napi]
    pub async fn get_tool_definitions(
        &self,
        descriptions: Option<NodeToolDescriptions>,
    ) -> Result<Vec<NodeToolDefinition>> {
        Ok(self
            .inner
            .lock()
            .await
            .get_tool_definitions(descriptions.map(ToolDescriptions::from))
            .into_iter()
            .map(NodeToolDefinition::from)
            .collect())
    }

    #[napi]
    pub async fn invoke_tool(&self, name: String, arguments: Value) -> Result<Value> {
        self.inner
            .lock()
            .await
            .invoke_tool(&name, arguments)
            .await
            .map_err(to_napi_error)
    }
}

fn embedding_config_from_options(options: Option<OrtEmbeddingOptions>) -> EmbeddingsConfig {
    let Some(options) = options else {
        return EmbeddingsConfig::default();
    };

    let mut config = EmbeddingsConfig::default();
    apply_option(&mut config.model_repo, options.model_repo);
    apply_option(&mut config.model_revision, options.model_revision);
    apply_option(&mut config.model_file, options.model_file);
    apply_option(&mut config.tokenizer_file, options.tokenizer_file);
    apply_option(&mut config.pooling_config_file, options.pooling_config_file);
    apply_option(
        &mut config.transformer_config_file,
        options.transformer_config_file,
    );
    if let Some(max_length) = option_u32_to_usize(options.max_length) {
        config.max_length = max_length;
    }
    if let Some(normalize) = options.normalize {
        config.normalize = normalize;
    }
    config.intra_threads = option_u32_to_usize(options.intra_threads);
    config.cache_dir = options.cache_dir.map(PathBuf::from);
    config.local_model_path = options.local_model_path.map(PathBuf::from);
    config.local_tokenizer_path = options.local_tokenizer_path.map(PathBuf::from);
    config.local_pooling_config_path = options.local_pooling_config_path.map(PathBuf::from);
    config.local_transformer_config_path = options.local_transformer_config_path.map(PathBuf::from);
    config.input_ids_name = options.input_ids_name;
    config.attention_mask_name = options.attention_mask_name;
    config.token_type_ids_name = options.token_type_ids_name;
    config.output_name = options.output_name;
    config
}

fn apply_option(target: &mut String, value: Option<String>) {
    if let Some(value) = value {
        *target = value;
    }
}

fn option_u32_to_usize(value: Option<u32>) -> Option<usize> {
    value.map(|value| value as usize)
}

fn to_napi_error(error: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, error.to_string())
}

impl From<IngestDocumentResult> for NodeIngestDocumentResult {
    fn from(value: IngestDocumentResult) -> Self {
        Self {
            document_id: value.document_id,
            chunk_count: value.chunk_count as u32,
        }
    }
}

impl From<DocumentSummary> for NodeDocumentSummary {
    fn from(value: DocumentSummary) -> Self {
        Self {
            document_id: value.document_id,
        }
    }
}

impl From<Document> for NodeDocument {
    fn from(value: Document) -> Self {
        Self {
            document_id: value.document_id,
            content: value.content,
        }
    }
}

impl From<VectorSearchResult> for NodeVectorSearchResult {
    fn from(value: VectorSearchResult) -> Self {
        Self {
            document_id: value.document_id,
            text: value.text,
            distance: f64::from(value.distance),
        }
    }
}

impl From<KeywordSearchResult> for NodeKeywordSearchResult {
    fn from(value: KeywordSearchResult) -> Self {
        Self {
            document_id: value.document_id,
            text: value.text,
            score: f64::from(value.score),
        }
    }
}

impl From<ToolDefinition> for NodeToolDefinition {
    fn from(value: ToolDefinition) -> Self {
        Self {
            name: value.name,
            description: value.description,
            input_schema: value.input_schema,
        }
    }
}

impl From<NodeToolDescriptions> for ToolDescriptions {
    fn from(value: NodeToolDescriptions) -> Self {
        Self {
            semantic_search: value.semantic_search,
            keyword_search: value.keyword_search,
            list_documents: value.list_documents,
            get_document: value.get_document,
        }
    }
}
