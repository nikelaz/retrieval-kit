use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use backends::lancedb::{Article, LanceDbBackend};
use glob::{GlobError, PatternError, glob};
use lancedb::Error as LanceDbError;
use uuid::Uuid;

pub mod backends;
pub mod chunking;
pub mod embeddings;

pub use chunking::{ChunkingConfig, ChunkingError, chunk_text};
pub use embeddings::{EmbeddingError, EmbeddingsConfig, EmbeddingsProvider, OrtEmbedder};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DbEngine {
    LanceDb {
        path: PathBuf,
        vector_dimensions: i32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EmbeddingsProviderKind {
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
}

impl fmt::Display for RKitInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DbEngine(error) => write!(f, "failed to initialize database engine: {error}"),
            Self::EmbeddingsProvider(error) => {
                write!(f, "failed to initialize embeddings provider: {error}")
            }
        }
    }
}

impl Error for RKitInitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::DbEngine(error) => Some(error),
            Self::EmbeddingsProvider(error) => Some(error),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IngestDocumentResult {
    pub document_id: String,
    pub chunk_count: usize,
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

    pub fn upsert_document(&self, id: String, content: String) {
        let _ = (id, content);
        panic!("not implemented");
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
            .as_mut()
            .ok_or(IngestDocumentError::NotInitialized)?;
        let embeddings = embeddings_provider
            .embed_batch(&flattened_chunks)
            .await
            .map_err(IngestDocumentError::Embeddings)?;

        let mut articles = Vec::with_capacity(flattened_chunks.len());
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
                articles.push(Article {
                    document_id: document.document_id.clone(),
                    text,
                    vector,
                });
            }

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
                .insert_data(&articles)
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

    pub fn vector_search(&self, query: String) {
        let _ = query;
        panic!("not implemented");
    }

    pub fn keyword_search(&self, query: String) {
        let _ = query;
        panic!("not implemented");
    }

    pub fn list_documents(&self) {
        panic!("not implemented");
    }

    pub fn get_document(&self, id: String) {
        let _ = id;
        panic!("not implemented");
    }

    pub fn delete_document(&self, id: String) {
        let _ = id;
        panic!("not implemented");
    }

    pub fn get_tool_definitions(&self) {
        panic!("not implemented");
    }

    pub fn invoke_tool(&self) {
        panic!("not implemented");
    }

    fn prepare_document(&self, content: String) -> Result<PreparedDocument, IngestDocumentError> {
        if content.trim().is_empty() {
            return Err(IngestDocumentError::EmptyContent);
        }

        let chunks =
            chunk_text(&content, self.chunking_config).map_err(IngestDocumentError::Chunking)?;
        if chunks.is_empty() {
            return Err(IngestDocumentError::EmptyContent);
        }

        Ok(PreparedDocument {
            document_id: Uuid::new_v4().to_string(),
            chunks,
        })
    }
}

enum InitializedDbEngine {
    LanceDb(LanceDbBackend),
}

#[cfg(test)]
struct MockEmbedder {
    dimensions: usize,
}

#[cfg(test)]
impl MockEmbedder {
    fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    async fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        Ok(texts
            .iter()
            .enumerate()
            .map(|(index, _)| vec![index as f32 + 1.0; self.dimensions])
            .collect())
    }
}

enum InitializedEmbeddingsProvider {
    Ort(embeddings::OrtEmbedder),
    #[cfg(test)]
    Mock(MockEmbedder),
}

impl InitializedEmbeddingsProvider {
    async fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        match self {
            Self::Ort(embedder) => embedder.embed_batch(texts).await,
            #[cfg(test)]
            Self::Mock(embedder) => embedder.embed_batch(texts).await,
        }
    }
}

struct PreparedDocument {
    document_id: String,
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
            .create_table()
            .await
            .map(|_| ())
            .map_err(RKitInitError::DbEngine),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChunkingConfig, DbEngine, EmbeddingError, EmbeddingsConfig, EmbeddingsProviderKind,
        IngestDocumentError, InitializedEmbeddingsProvider, MockEmbedder, RKit, RKitConfigError,
        RKitInitError,
    };
    use arrow_array::StringArray;
    use futures::TryStreamExt;
    use lancedb::query::ExecutableQuery;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

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
                    .open_table("articles")
                    .execute()
                    .await
                    .unwrap();
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

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("articles")
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

        let result = rkit.ingest_document(content).await.unwrap();

        assert_eq!(result.chunk_count, expected_chunk_count);
        assert!(!result.document_id.is_empty());

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("articles")
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
            .open_table("articles")
            .execute()
            .await
            .unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 0);
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
            .open_table("articles")
            .execute()
            .await
            .unwrap();
        assert_eq!(
            table.count_rows(None).await.unwrap(),
            first.chunk_count + second.chunk_count
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
            .open_table("articles")
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
            .open_table("articles")
            .execute()
            .await
            .unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 0);
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
            .open_table("articles")
            .execute()
            .await
            .unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 0);
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
            .ingest_documents(vec![first_content, second_content])
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_count, expected_first);
        assert_eq!(results[1].chunk_count, expected_second);
        assert_ne!(results[0].document_id, results[1].document_id);

        let backend = rkit.lancedb_backend().unwrap();
        let table = backend
            .connection()
            .open_table("articles")
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
            .open_table("articles")
            .execute()
            .await
            .unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 0);
    }
}
