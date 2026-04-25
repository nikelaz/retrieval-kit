use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use ndarray::{Array2, ArrayView2, ArrayView3, Axis, Ix2, Ix3};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tokenizers::tokenizer::{PaddingParams, PaddingStrategy, TruncationParams};

const DEFAULT_MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L12-v2";
const DEFAULT_MODEL_REVISION: &str = "main";
const DEFAULT_MODEL_FILE: &str = "onnx/model.onnx";
const DEFAULT_TOKENIZER_FILE: &str = "tokenizer.json";
const DEFAULT_POOLING_CONFIG_FILE: &str = "1_Pooling/config.json";
const DEFAULT_TRANSFORMER_CONFIG_FILE: &str = "config.json";
const DEFAULT_MAX_LENGTH: usize = 128;
type EncodedInputs = (Array2<i64>, Array2<i64>, Option<Array2<i64>>);

#[allow(async_fn_in_trait)]
/// Provider interface for generating embeddings from text batches.
pub trait EmbeddingsProvider {
    async fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    async fn embed(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut embeddings = self.embed_batch(&[text.to_owned()]).await?;
        embeddings.pop().ok_or(EmbeddingError::MissingOutput(
            "no embeddings returned".to_string(),
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Configuration for the built-in ONNX Runtime embedder.
pub struct EmbeddingsConfig {
    /// Hugging Face model repository used when local assets are not supplied.
    pub model_repo: String,
    /// Hugging Face model revision used when local assets are not supplied.
    pub model_revision: String,
    /// Model file path inside the Hugging Face repository.
    pub model_file: String,
    /// Tokenizer file path inside the Hugging Face repository.
    pub tokenizer_file: String,
    /// Pooling config file path inside the Hugging Face repository.
    pub pooling_config_file: String,
    /// Transformer config file path inside the Hugging Face repository.
    pub transformer_config_file: String,
    /// Maximum tokenizer sequence length, capped by model config when known.
    pub max_length: usize,
    /// Whether output embeddings should be L2-normalized.
    pub normalize: bool,
    /// Optional ONNX Runtime intra-op thread count.
    pub intra_threads: Option<usize>,
    /// Optional Hugging Face cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Local ONNX model path. If unset, the model is resolved from Hugging Face.
    pub local_model_path: Option<PathBuf>,
    /// Local tokenizer path. If unset, the tokenizer is resolved from Hugging Face.
    pub local_tokenizer_path: Option<PathBuf>,
    /// Local sentence-transformers pooling config path.
    pub local_pooling_config_path: Option<PathBuf>,
    /// Local transformer config path.
    pub local_transformer_config_path: Option<PathBuf>,
    /// Override for models that use a non-standard input IDs name.
    pub input_ids_name: Option<String>,
    /// Override for models that use a non-standard attention mask name.
    pub attention_mask_name: Option<String>,
    /// Override for models that use a non-standard token type IDs name.
    pub token_type_ids_name: Option<String>,
    /// Optional output tensor name. Defaults to the first model output.
    pub output_name: Option<String>,
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        Self {
            model_repo: DEFAULT_MODEL_REPO.to_string(),
            model_revision: DEFAULT_MODEL_REVISION.to_string(),
            model_file: DEFAULT_MODEL_FILE.to_string(),
            tokenizer_file: DEFAULT_TOKENIZER_FILE.to_string(),
            pooling_config_file: DEFAULT_POOLING_CONFIG_FILE.to_string(),
            transformer_config_file: DEFAULT_TRANSFORMER_CONFIG_FILE.to_string(),
            max_length: DEFAULT_MAX_LENGTH,
            normalize: true,
            intra_threads: None,
            cache_dir: None,
            local_model_path: None,
            local_tokenizer_path: None,
            local_pooling_config_path: None,
            local_transformer_config_path: None,
            input_ids_name: None,
            attention_mask_name: None,
            token_type_ids_name: None,
            output_name: None,
        }
    }
}

#[derive(Debug)]
/// ONNX Runtime sentence embedding provider.
pub struct OrtEmbedder {
    inner: Arc<Mutex<OrtEmbedderInner>>,
    max_length: usize,
}

impl OrtEmbedder {
    pub fn new(config: EmbeddingsConfig) -> Result<Self, EmbeddingError> {
        let assets = resolve_model_assets(&config)?;
        let pooling_config = read_json::<PoolingConfig>(&assets.pooling_config_path)?;
        validate_pooling_config(&pooling_config)?;

        let transformer_config = read_json::<TransformerConfig>(&assets.transformer_config_path)?;
        let expected_embedding_size = pooling_config
            .word_embedding_dimension
            .or(transformer_config.hidden_size);
        let max_length = transformer_config
            .max_position_embeddings
            .map(|value| value.min(config.max_length))
            .unwrap_or(config.max_length);

        let tokenizer = load_tokenizer(&assets.tokenizer_path, max_length)?;
        let session = load_session(&assets.model_path, config.intra_threads)?;
        let input_names = SessionInputNames::from_session(
            &session,
            config.input_ids_name.as_deref(),
            config.attention_mask_name.as_deref(),
            config.token_type_ids_name.as_deref(),
        )?;
        let output_name = select_output_name(&session, config.output_name.as_deref())?;

        Ok(Self {
            inner: Arc::new(Mutex::new(OrtEmbedderInner {
                tokenizer,
                session,
                input_names,
                output_name,
                normalize: config.normalize,
                expected_embedding_size,
            })),
            max_length,
        })
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    pub fn expected_embedding_size(&self) -> Option<usize> {
        self.inner
            .lock()
            .ok()
            .and_then(|inner| inner.expected_embedding_size)
    }

    pub fn chunk_text(
        &self,
        text: &str,
        overlap_tokens: usize,
    ) -> Result<Vec<String>, EmbeddingError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let inner = self
            .inner
            .lock()
            .map_err(|error| EmbeddingError::State(format!("embedder state poisoned: {error}")))?;
        inner.chunk_text(text, self.max_length, overlap_tokens)
    }
}

#[derive(Debug)]
struct OrtEmbedderInner {
    tokenizer: Tokenizer,
    session: Session,
    input_names: SessionInputNames,
    output_name: Option<String>,
    normalize: bool,
    expected_embedding_size: Option<usize>,
}

impl OrtEmbedderInner {
    fn chunk_text(
        &self,
        text: &str,
        max_length: usize,
        overlap_tokens: usize,
    ) -> Result<Vec<String>, EmbeddingError> {
        chunk_text_with_tokenizer(&self.tokenizer, text, max_length, overlap_tokens)
    }

    fn encode_inputs(&self, texts: &[String]) -> Result<EncodedInputs, EmbeddingError> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.iter().map(String::as_str).collect(), true)
            .map_err(EmbeddingError::Tokenizer)?;

        let batch_size = encodings.len();
        let sequence_length = encodings
            .first()
            .map(|encoding| encoding.get_ids().len())
            .unwrap_or(0);

        let mut input_ids = Array2::<i64>::zeros((batch_size, sequence_length));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, sequence_length));
        let mut token_type_ids = self
            .input_names
            .token_type_ids
            .as_ref()
            .map(|_| Array2::<i64>::zeros((batch_size, sequence_length)));

        for (row_index, encoding) in encodings.iter().enumerate() {
            for (column_index, token_id) in encoding.get_ids().iter().enumerate() {
                input_ids[(row_index, column_index)] = i64::from(*token_id);
            }

            for (column_index, mask) in encoding.get_attention_mask().iter().enumerate() {
                attention_mask[(row_index, column_index)] = i64::from(*mask);
            }

            if let Some(token_type_ids) = token_type_ids.as_mut() {
                for (column_index, token_type_id) in encoding.get_type_ids().iter().enumerate() {
                    token_type_ids[(row_index, column_index)] = i64::from(*token_type_id);
                }
            }
        }

        Ok((input_ids, attention_mask, token_type_ids))
    }

    fn run_inference(
        &mut self,
        input_ids: Array2<i64>,
        attention_mask: Array2<i64>,
        token_type_ids: Option<Array2<i64>>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut inputs = vec![
            (
                self.input_names.input_ids.clone(),
                Tensor::from_array(input_ids)
                    .map_err(|error| EmbeddingError::Ort(error.to_string()))?,
            ),
            (
                self.input_names.attention_mask.clone(),
                Tensor::from_array(attention_mask.clone())
                    .map_err(|error| EmbeddingError::Ort(error.to_string()))?,
            ),
        ];

        if let (Some(input_name), Some(token_type_ids)) =
            (self.input_names.token_type_ids.as_ref(), token_type_ids)
        {
            inputs.push((
                input_name.clone(),
                Tensor::from_array(token_type_ids)
                    .map_err(|error| EmbeddingError::Ort(error.to_string()))?,
            ));
        }

        let outputs = self
            .session
            .run(inputs)
            .map_err(|error| EmbeddingError::Ort(error.to_string()))?;
        let output_value = match self.output_name.as_deref() {
            Some(output_name) => &outputs[output_name],
            None => {
                if outputs.len() == 0 {
                    return Err(EmbeddingError::MissingOutput(
                        "model returned no outputs".to_string(),
                    ));
                }

                &outputs[0]
            }
        };

        let output_array = match output_value.try_extract_array::<f32>() {
            Ok(array) => array,
            Err(error) => return Err(EmbeddingError::Ort(error.to_string())),
        };

        let embeddings = match output_array.ndim() {
            2 => collect_sentence_embeddings(
                output_array
                    .into_dimensionality::<Ix2>()
                    .map_err(|_| EmbeddingError::InvalidOutputShape(vec![]))?,
                self.normalize,
            ),
            3 => mean_pool_embeddings(
                output_array
                    .into_dimensionality::<Ix3>()
                    .map_err(|_| EmbeddingError::InvalidOutputShape(vec![]))?,
                attention_mask.view(),
                self.normalize,
            )?,
            _ => {
                return Err(EmbeddingError::InvalidOutputShape(
                    output_array.shape().to_vec(),
                ));
            }
        };

        if let Some(expected_embedding_size) = self.expected_embedding_size {
            for embedding in &embeddings {
                if embedding.len() != expected_embedding_size {
                    return Err(EmbeddingError::EmbeddingDimensionMismatch {
                        expected: expected_embedding_size,
                        actual: embedding.len(),
                    });
                }
            }
        }

        Ok(embeddings)
    }
}

fn chunk_text_with_tokenizer(
    tokenizer: &Tokenizer,
    text: &str,
    max_length: usize,
    overlap_tokens: usize,
) -> Result<Vec<String>, EmbeddingError> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let max_content_tokens = max_length.saturating_sub(2).max(1);
    let overlap_tokens = overlap_tokens.min(max_content_tokens.saturating_sub(1));
    let encoding = tokenizer
        .encode(text, false)
        .map_err(EmbeddingError::Tokenizer)?;
    let offsets = encoding
        .get_offsets()
        .iter()
        .copied()
        .filter(|(start, end)| start < end)
        .collect::<Vec<_>>();

    if offsets.is_empty() {
        return Ok(Vec::new());
    }

    let mut chunks = Vec::new();
    let mut start_token = 0;

    while start_token < offsets.len() {
        let end_token = (start_token + max_content_tokens).min(offsets.len());
        let start_byte = offsets[start_token].0;
        let end_byte = offsets[end_token - 1].1;
        let chunk = text[start_byte..end_byte].trim();

        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }

        if end_token >= offsets.len() {
            break;
        }

        let next_start = end_token.saturating_sub(overlap_tokens);
        start_token = if next_start <= start_token {
            end_token
        } else {
            next_start
        };
    }

    Ok(chunks)
}

impl EmbeddingsProvider for OrtEmbedder {
    async fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.embed_batch_shared(texts).await
    }
}

impl OrtEmbedder {
    pub async fn embed_batch_shared(
        &self,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let inner = Arc::clone(&self.inner);
        let texts = texts.to_vec();
        tokio::task::spawn_blocking(move || {
            let mut inner = inner.lock().map_err(|error| {
                EmbeddingError::State(format!("embedder state poisoned: {error}"))
            })?;
            let (input_ids, attention_mask, token_type_ids) = inner.encode_inputs(&texts)?;
            inner.run_inference(input_ids, attention_mask, token_type_ids)
        })
        .await
        .map_err(|error| EmbeddingError::BlockingTask(error.to_string()))?
    }
}

#[derive(Debug)]
pub enum EmbeddingError {
    InvalidConfig(&'static str),
    MissingAsset { asset: &'static str, path: PathBuf },
    MissingModelInput(&'static str),
    MissingOutput(String),
    UnsupportedPooling(String),
    InvalidOutputShape(Vec<usize>),
    EmbeddingDimensionMismatch { expected: usize, actual: usize },
    Hub(hf_hub::api::sync::ApiError),
    Io(std::io::Error),
    Json(serde_json::Error),
    Ort(String),
    State(String),
    BlockingTask(String),
    Tokenizer(tokenizers::Error),
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "{message}"),
            Self::MissingAsset { asset, path } => {
                write!(f, "missing {asset} asset at {}", path.display())
            }
            Self::MissingModelInput(input_name) => {
                write!(f, "model is missing required input `{input_name}`")
            }
            Self::MissingOutput(output_name) => write!(f, "model output not found: {output_name}"),
            Self::UnsupportedPooling(message) => write!(f, "{message}"),
            Self::InvalidOutputShape(shape) => {
                write!(f, "unexpected model output shape: {shape:?}")
            }
            Self::EmbeddingDimensionMismatch { expected, actual } => write!(
                f,
                "embedding dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::Hub(error) => write!(f, "{error}"),
            Self::Io(error) => write!(f, "{error}"),
            Self::Json(error) => write!(f, "{error}"),
            Self::Ort(error) => write!(f, "{error}"),
            Self::State(error) => write!(f, "{error}"),
            Self::BlockingTask(error) => write!(f, "embedding task failed: {error}"),
            Self::Tokenizer(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

impl From<hf_hub::api::sync::ApiError> for EmbeddingError {
    fn from(value: hf_hub::api::sync::ApiError) -> Self {
        Self::Hub(value)
    }
}

impl From<std::io::Error> for EmbeddingError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for EmbeddingError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Debug)]
struct SessionInputNames {
    input_ids: String,
    attention_mask: String,
    token_type_ids: Option<String>,
}

impl SessionInputNames {
    fn from_session(
        session: &Session,
        input_ids_name: Option<&str>,
        attention_mask_name: Option<&str>,
        token_type_ids_name: Option<&str>,
    ) -> Result<Self, EmbeddingError> {
        let inputs = session.inputs();
        let input_ids = resolve_required_name(inputs, input_ids_name, "input_ids")?;
        let attention_mask = resolve_required_name(inputs, attention_mask_name, "attention_mask")?;
        let token_type_ids = resolve_optional_name(inputs, token_type_ids_name, "token_type_ids")?;

        Ok(Self {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }
}

#[derive(Debug)]
struct ModelAssets {
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    pooling_config_path: PathBuf,
    transformer_config_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct PoolingConfig {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    word_embedding_dimension: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct TransformerConfig {
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
}

fn resolve_model_assets(config: &EmbeddingsConfig) -> Result<ModelAssets, EmbeddingError> {
    let use_hub = config.local_model_path.is_none()
        || config.local_tokenizer_path.is_none()
        || config.local_pooling_config_path.is_none()
        || config.local_transformer_config_path.is_none();

    let api = if use_hub {
        let builder = match config.cache_dir.clone() {
            Some(cache_dir) => ApiBuilder::new().with_cache_dir(cache_dir),
            None => ApiBuilder::from_env(),
        };
        Some(builder.with_progress(false).build()?)
    } else {
        None
    };

    let repo = api.as_ref().map(|api| {
        api.repo(Repo::with_revision(
            config.model_repo.clone(),
            RepoType::Model,
            config.model_revision.clone(),
        ))
    });

    Ok(ModelAssets {
        model_path: resolve_asset_path(
            config.local_model_path.as_deref(),
            repo.as_ref(),
            &config.model_file,
            "model",
        )?,
        tokenizer_path: resolve_asset_path(
            config.local_tokenizer_path.as_deref(),
            repo.as_ref(),
            &config.tokenizer_file,
            "tokenizer",
        )?,
        pooling_config_path: resolve_asset_path(
            config.local_pooling_config_path.as_deref(),
            repo.as_ref(),
            &config.pooling_config_file,
            "pooling config",
        )?,
        transformer_config_path: resolve_asset_path(
            config.local_transformer_config_path.as_deref(),
            repo.as_ref(),
            &config.transformer_config_file,
            "transformer config",
        )?,
    })
}

fn resolve_asset_path(
    local_path: Option<&Path>,
    repo: Option<&hf_hub::api::sync::ApiRepo>,
    remote_path: &str,
    asset_name: &'static str,
) -> Result<PathBuf, EmbeddingError> {
    if let Some(local_path) = local_path {
        return ensure_existing_path(local_path.to_path_buf(), asset_name);
    }

    let repo = repo.ok_or(EmbeddingError::InvalidConfig(
        "remote model resolution requires a Hugging Face repository",
    ))?;

    let path = repo.get(remote_path)?;
    ensure_existing_path(path, asset_name)
}

fn ensure_existing_path(
    path: PathBuf,
    asset_name: &'static str,
) -> Result<PathBuf, EmbeddingError> {
    if path.exists() {
        Ok(path)
    } else {
        Err(EmbeddingError::MissingAsset {
            asset: asset_name,
            path,
        })
    }
}

fn read_json<T>(path: &Path) -> Result<T, EmbeddingError>
where
    T: for<'de> Deserialize<'de>,
{
    let contents = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents)?)
}

fn validate_pooling_config(pooling_config: &PoolingConfig) -> Result<(), EmbeddingError> {
    if pooling_config.pooling_mode_mean_tokens
        && !pooling_config.pooling_mode_cls_token
        && !pooling_config.pooling_mode_max_tokens
        && !pooling_config.pooling_mode_mean_sqrt_len_tokens
    {
        return Ok(());
    }

    Err(EmbeddingError::UnsupportedPooling(
        "only mean-token pooling is currently supported".to_string(),
    ))
}

fn load_tokenizer(path: &Path, max_length: usize) -> Result<Tokenizer, EmbeddingError> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(EmbeddingError::Tokenizer)?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(EmbeddingError::Tokenizer)?;

    let mut padding = tokenizer.get_padding().cloned().unwrap_or_default();
    padding.strategy = PaddingStrategy::BatchLongest;
    tokenizer.with_padding(Some(PaddingParams { ..padding }));

    Ok(tokenizer)
}

fn load_session(path: &Path, intra_threads: Option<usize>) -> Result<Session, EmbeddingError> {
    let builder = Session::builder().map_err(|error| EmbeddingError::Ort(error.to_string()))?;
    let mut builder = if let Some(intra_threads) = intra_threads {
        builder
            .with_intra_threads(intra_threads)
            .map_err(|error| EmbeddingError::Ort(error.to_string()))?
    } else {
        builder
    };

    builder
        .commit_from_file(path)
        .map_err(|error| EmbeddingError::Ort(error.to_string()))
}

fn resolve_required_name(
    inputs: &[ort::value::Outlet],
    configured_name: Option<&str>,
    default_name: &'static str,
) -> Result<String, EmbeddingError> {
    if let Some(configured_name) = configured_name {
        return inputs
            .iter()
            .find(|input| input.name() == configured_name)
            .map(|input| input.name().to_string())
            .ok_or(EmbeddingError::MissingModelInput(default_name));
    }

    inputs
        .iter()
        .find(|input| input.name() == default_name)
        .map(|input| input.name().to_string())
        .ok_or(EmbeddingError::MissingModelInput(default_name))
}

fn resolve_optional_name(
    inputs: &[ort::value::Outlet],
    configured_name: Option<&str>,
    default_name: &'static str,
) -> Result<Option<String>, EmbeddingError> {
    if let Some(configured_name) = configured_name {
        return inputs
            .iter()
            .find(|input| input.name() == configured_name)
            .map(|input| Some(input.name().to_string()))
            .ok_or(EmbeddingError::MissingModelInput(default_name));
    }

    Ok(inputs
        .iter()
        .find(|input| input.name() == default_name)
        .map(|input| input.name().to_string()))
}

fn select_output_name(
    session: &Session,
    configured_name: Option<&str>,
) -> Result<Option<String>, EmbeddingError> {
    if let Some(configured_name) = configured_name {
        return session
            .outputs()
            .iter()
            .find(|output| output.name() == configured_name)
            .map(|output| Some(output.name().to_string()))
            .ok_or_else(|| EmbeddingError::MissingOutput(configured_name.to_string()));
    }

    Ok(session
        .outputs()
        .first()
        .map(|output| output.name().to_string()))
}

fn mean_pool_embeddings(
    token_embeddings: ArrayView3<'_, f32>,
    attention_mask: ArrayView2<'_, i64>,
    normalize: bool,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let batch_size = token_embeddings.len_of(Axis(0));
    let sequence_length = token_embeddings.len_of(Axis(1));
    let embedding_size = token_embeddings.len_of(Axis(2));

    if attention_mask.shape() != [batch_size, sequence_length] {
        return Err(EmbeddingError::InvalidOutputShape(vec![
            batch_size,
            sequence_length,
            embedding_size,
        ]));
    }

    let mut sentence_embeddings = Vec::with_capacity(batch_size);
    for batch_index in 0..batch_size {
        let mut pooled = vec![0.0_f32; embedding_size];
        let mut token_count = 0.0_f32;

        for token_index in 0..sequence_length {
            let mask = attention_mask[(batch_index, token_index)] as f32;
            if mask <= 0.0 {
                continue;
            }

            token_count += mask;
            for embedding_index in 0..embedding_size {
                pooled[embedding_index] +=
                    token_embeddings[(batch_index, token_index, embedding_index)] * mask;
            }
        }

        if token_count > 0.0 {
            for value in &mut pooled {
                *value /= token_count;
            }
        }

        if normalize {
            l2_normalize(&mut pooled);
        }

        sentence_embeddings.push(pooled);
    }

    Ok(sentence_embeddings)
}

fn collect_sentence_embeddings(embeddings: ArrayView2<'_, f32>, normalize: bool) -> Vec<Vec<f32>> {
    embeddings
        .axis_iter(Axis(0))
        .map(|row| {
            let mut embedding = row.to_vec();
            if normalize {
                l2_normalize(&mut embedding);
            }
            embedding
        })
        .collect()
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_MAX_LENGTH, DEFAULT_MODEL_FILE, DEFAULT_MODEL_REPO, DEFAULT_MODEL_REVISION,
        DEFAULT_POOLING_CONFIG_FILE, DEFAULT_TOKENIZER_FILE, DEFAULT_TRANSFORMER_CONFIG_FILE,
        EmbeddingError, EmbeddingsConfig, TransformerConfig, chunk_text_with_tokenizer,
        collect_sentence_embeddings, ensure_existing_path, mean_pool_embeddings, read_json,
        resolve_asset_path,
    };
    use ahash::AHashMap;
    use ndarray::{Array2, Array3, array};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::processors::bert::BertProcessing;

    #[test]
    fn uses_expected_default_embedding_config() {
        let config = EmbeddingsConfig::default();

        assert_eq!(config.model_repo, DEFAULT_MODEL_REPO);
        assert_eq!(config.model_revision, DEFAULT_MODEL_REVISION);
        assert_eq!(config.model_file, DEFAULT_MODEL_FILE);
        assert_eq!(config.tokenizer_file, DEFAULT_TOKENIZER_FILE);
        assert_eq!(config.pooling_config_file, DEFAULT_POOLING_CONFIG_FILE);
        assert_eq!(
            config.transformer_config_file,
            DEFAULT_TRANSFORMER_CONFIG_FILE
        );
        assert_eq!(config.max_length, DEFAULT_MAX_LENGTH);
        assert!(config.normalize);
        assert!(config.cache_dir.is_none());
    }

    #[test]
    fn prefers_local_asset_override_when_present() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("model.onnx");
        fs::write(&model_path, b"model").unwrap();

        let resolved = resolve_asset_path(Some(&model_path), None, "ignored", "model").unwrap();

        assert_eq!(resolved, model_path);
    }

    #[test]
    fn rejects_missing_local_asset_override() {
        let missing_path = PathBuf::from("/tmp/retrieval-kit-missing-model.onnx");

        let error = ensure_existing_path(missing_path.clone(), "model").unwrap_err();

        match error {
            EmbeddingError::MissingAsset { asset, path } => {
                assert_eq!(asset, "model");
                assert_eq!(path, missing_path);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn mean_pooling_respects_attention_mask() {
        let token_embeddings =
            Array3::from_shape_vec((1, 3, 2), vec![1.0, 0.0, 3.0, 4.0, 100.0, 100.0]).unwrap();
        let attention_mask = array![[1_i64, 1, 0]];

        let embeddings =
            mean_pool_embeddings(token_embeddings.view(), attention_mask.view(), false).unwrap();

        assert_eq!(embeddings, vec![vec![2.0, 2.0]]);
    }

    #[test]
    fn sentence_embeddings_are_normalized_when_requested() {
        let embeddings = Array2::from_shape_vec((1, 2), vec![3.0_f32, 4.0]).unwrap();

        let normalized = collect_sentence_embeddings(embeddings.view(), true);

        assert!((normalized[0][0] - 0.6).abs() < 1e-6);
        assert!((normalized[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn reads_transformer_config_from_local_file() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");
        fs::write(
            &config_path,
            r#"{"hidden_size":384,"max_position_embeddings":256}"#,
        )
        .unwrap();

        let config: TransformerConfig = read_json(&config_path).unwrap();

        assert_eq!(config.hidden_size, Some(384));
        assert_eq!(config.max_position_embeddings, Some(256));
    }

    #[test]
    fn tokenizer_fixture_saves_to_local_json() {
        let temp_dir = tempdir().unwrap();
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        build_test_tokenizer().save(&tokenizer_path, false).unwrap();

        assert!(tokenizer_path.exists());
    }

    #[test]
    fn token_chunking_respects_model_length_with_overlap() {
        let tokenizer = build_test_tokenizer();
        let chunks =
            chunk_text_with_tokenizer(&tokenizer, "hello world hello world hello world", 5, 1)
                .unwrap();

        assert_eq!(
            chunks,
            vec!["hello world hello", "hello world hello", "hello world"]
        );
        for chunk in chunks {
            let encoding = tokenizer.encode(chunk.as_str(), true).unwrap();
            assert!(encoding.len() <= 5);
        }
    }

    fn build_test_tokenizer() -> Tokenizer {
        let vocab = AHashMap::from_iter([
            ("[UNK]".to_string(), 0),
            ("[PAD]".to_string(), 1),
            ("[CLS]".to_string(), 2),
            ("[SEP]".to_string(), 3),
            ("hello".to_string(), 4),
            ("world".to_string(), 5),
        ]);

        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_post_processor(Some(BertProcessing::new(
            ("[SEP]".to_string(), 3),
            ("[CLS]".to_string(), 2),
        )));
        tokenizer
    }
}
