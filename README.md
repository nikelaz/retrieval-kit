# Retrieval Kit

Retrieval Kit is a small Rust library for local document ingestion, vector search,
keyword search, and MCP-style retrieval tool definitions.

It currently ships with:

- LanceDB storage for documents, chunks, vectors, and full-text search
- ONNX Runtime embeddings through `sentence-transformers/all-MiniLM-L12-v2`
- single-document, batch, file, and glob ingestion
- semantic search, keyword search, document list/get/delete APIs
- JSON tool definitions and invocation helpers for retrieval integrations

## Example

```rust
use retrieval_kit::{
    DbEngine, EmbeddingsConfig, EmbeddingsProviderKind, RKit,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rkit = RKit::new(
        DbEngine::LanceDb {
            path: PathBuf::from("./rkit-data"),
            vector_dimensions: 384,
        },
        EmbeddingsProviderKind::Ort(EmbeddingsConfig::default()),
    )?;

    rkit.init().await?;

    let ingested = rkit
        .ingest_document("Rust makes local retrieval systems practical.".to_string())
        .await?;

    let semantic_results = rkit
        .vector_search("local search in Rust".to_string(), 5)
        .await?;
    let keyword_results = rkit.keyword_search("Rust".to_string(), 5).await?;
    let document = rkit.get_document(ingested.document_id).await?;

    println!("{semantic_results:#?}");
    println!("{keyword_results:#?}");
    println!("{document:#?}");

    Ok(())
}
```

## Configuration Notes

The default embedder downloads model assets from Hugging Face unless local paths
are supplied in `EmbeddingsConfig`. To run fully offline, provide all four local
asset paths:

- `local_model_path`
- `local_tokenizer_path`
- `local_pooling_config_path`
- `local_transformer_config_path`

For the default `all-MiniLM-L12-v2` model, set LanceDB `vector_dimensions` to
`384`. Initialization validates known embedder dimensions against the database
schema and fails early on mismatches.

Ingestion uses tokenizer-aware chunking when the ORT embedder is initialized, so
chunks are split before model truncation would drop content. The standalone
`chunk_text` helper remains character based.

## Retrieval Tools

`get_tool_definitions` returns JSON schemas for:

- `semantic_search`
- `keyword_search`
- `list_documents`
- `get_document`

Use `invoke_tool` to dispatch those tools directly from JSON arguments. Search
tools default to a limit of 10 when no limit is provided.

## Current Limits

- LanceDB and ORT are the only built-in backend/provider pair.
- Document metadata and metadata filters are not implemented yet.
- LanceDB writes are prevalidated and partially cleaned up on insert failure, but
  they are not fully transactional.
- Vector indexes are created only after enough rows exist for LanceDB's automatic
  vector index training. Keyword index and later vector index creation errors are
  returned to callers.
