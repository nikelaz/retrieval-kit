use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    UInt64Array, cast::AsArray, types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::database::CreateTableMode;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::index::{Index, scalar::FtsIndexBuilder};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{Connection, Error, Result, Table, connect};

const DOCUMENTS_TABLE_NAME: &str = "documents";
const CHUNKS_TABLE_NAME: &str = "chunks";
const MIN_VECTOR_INDEX_ROWS: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocumentRecord {
    pub document_id: String,
    pub content: String,
}

pub struct Chunk {
    pub document_id: String,
    pub chunk_index: u64,
    pub text: String,
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChunkSearchRecord {
    pub document_id: String,
    pub text: String,
    pub distance: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChunkKeywordSearchRecord {
    pub document_id: String,
    pub text: String,
    pub score: f32,
}

pub struct LanceDbBackend {
    connection: Connection,
    vector_dimensions: i32,
}

impl LanceDbBackend {
    pub async fn new(path: impl AsRef<Path>, vector_dimensions: i32) -> Result<Self> {
        if vector_dimensions <= 0 {
            return Err(Error::InvalidInput {
                message: "vector_dimensions must be greater than zero".to_string(),
            });
        }

        let uri = path.as_ref().to_string_lossy();
        let connection = connect(uri.as_ref()).execute().await?;

        Ok(Self {
            connection,
            vector_dimensions,
        })
    }

    pub async fn create_tables(&self) -> Result<()> {
        self.create_documents_table().await?;
        self.create_chunks_table().await?;
        Ok(())
    }

    pub async fn create_documents_table(&self) -> Result<Table> {
        self.connection
            .create_empty_table(DOCUMENTS_TABLE_NAME, self.documents_schema())
            .mode(CreateTableMode::exist_ok(|request| request))
            .execute()
            .await
    }

    pub async fn create_chunks_table(&self) -> Result<Table> {
        self.connection
            .create_empty_table(CHUNKS_TABLE_NAME, self.chunks_schema())
            .mode(CreateTableMode::exist_ok(|request| request))
            .execute()
            .await
    }

    pub async fn insert_data(&self, documents: &[DocumentRecord], chunks: &[Chunk]) -> Result<()> {
        let documents_batch = self.documents_batch(documents)?;
        let chunks_batch = self.chunks_batch(chunks)?;

        let documents_table = self
            .connection
            .open_table(DOCUMENTS_TABLE_NAME)
            .execute()
            .await?;
        let chunks_table = self
            .connection
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await?;

        chunks_table.add(chunks_batch).execute().await?;
        if let Err(error) = documents_table.add(documents_batch).execute().await {
            self.delete_chunks_for_documents(&chunks_table, documents)
                .await;
            return Err(error);
        }
        self.ensure_chunks_indices(&chunks_table).await?;

        Ok(())
    }

    pub async fn upsert_data(&self, document: &DocumentRecord, chunks: &[Chunk]) -> Result<()> {
        let documents_batch = self.documents_batch(std::slice::from_ref(document))?;
        let chunks_batch = self.chunks_batch(chunks)?;
        let previous_chunks = self.chunks_for_document(&document.document_id).await?;
        let documents_table = self
            .connection
            .open_table(DOCUMENTS_TABLE_NAME)
            .execute()
            .await?;
        let chunks_table = self
            .connection
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await?;

        self.merge_replace_document_chunks(&chunks_table, &document.document_id, chunks_batch)
            .await?;
        if let Err(error) = self
            .merge_upsert_documents(&documents_table, documents_batch)
            .await
        {
            let _ = self
                .restore_document_chunks(&chunks_table, &document.document_id, previous_chunks)
                .await;
            return Err(error);
        }

        self.ensure_chunks_indices(&chunks_table).await?;

        Ok(())
    }

    pub async fn vector_search(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<ChunkSearchRecord>> {
        let table = self
            .connection
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await?;
        let rows = table
            .query()
            .nearest_to(query_vector)?
            .column("vector")
            .limit(limit)
            .select(Select::columns(&["document_id", "text", "_distance"]))
            .execute()
            .await?;
        let batches = rows.try_collect::<Vec<_>>().await?;

        Ok(chunk_search_records_from_batches(&batches))
    }

    pub async fn keyword_search(
        &self,
        query: String,
        limit: usize,
    ) -> Result<Vec<ChunkKeywordSearchRecord>> {
        let table = self
            .connection
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await?;
        let full_text_query = FullTextSearchQuery::new(query).with_column("text".to_string())?;
        let rows = table
            .query()
            .full_text_search(full_text_query)
            .limit(limit)
            .select(Select::columns(&["document_id", "text", "_score"]))
            .execute()
            .await?;
        let batches = rows.try_collect::<Vec<_>>().await?;

        Ok(chunk_keyword_search_records_from_batches(&batches))
    }

    pub async fn list_documents(&self) -> Result<Vec<DocumentRecord>> {
        let table = self
            .connection
            .open_table(DOCUMENTS_TABLE_NAME)
            .execute()
            .await?;
        let rows = table
            .query()
            .select(Select::columns(&["document_id", "content"]))
            .execute()
            .await?;
        let batches = rows.try_collect::<Vec<_>>().await?;
        let mut documents = document_records_from_batches(&batches);

        documents.sort_by(|left, right| left.document_id.cmp(&right.document_id));
        Ok(documents)
    }

    pub async fn get_document(&self, document_id: &str) -> Result<Option<DocumentRecord>> {
        let table = self
            .connection
            .open_table(DOCUMENTS_TABLE_NAME)
            .execute()
            .await?;
        let rows = table
            .query()
            .only_if(document_id_predicate(document_id))
            .select(Select::columns(&["document_id", "content"]))
            .limit(1)
            .execute()
            .await?;
        let batches = rows.try_collect::<Vec<_>>().await?;

        Ok(document_records_from_batches(&batches).into_iter().next())
    }

    pub async fn delete_document(&self, document_id: &str) -> Result<()> {
        let predicate = document_id_predicate(document_id);
        let documents_table = self
            .connection
            .open_table(DOCUMENTS_TABLE_NAME)
            .execute()
            .await?;
        let chunks_table = self
            .connection
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await?;

        documents_table.delete(&predicate).await?;
        chunks_table.delete(&predicate).await?;

        Ok(())
    }

    pub fn connection(&self) -> &Connection {
        &self.connection
    }

    pub fn vector_dimensions(&self) -> i32 {
        self.vector_dimensions
    }

    async fn ensure_chunks_vector_index(&self, chunks_table: &Table) -> Result<()> {
        let indices = chunks_table.list_indices().await?;
        if indices
            .iter()
            .any(|index| index.columns == vec!["vector".to_string()])
        {
            return Ok(());
        }
        if chunks_table.count_rows(None).await? < MIN_VECTOR_INDEX_ROWS {
            return Ok(());
        }

        chunks_table
            .create_index(&["vector"], Index::Auto)
            .execute()
            .await?;
        Ok(())
    }

    async fn ensure_chunks_keyword_index(&self, chunks_table: &Table) -> Result<()> {
        let indices = chunks_table.list_indices().await?;
        if indices
            .iter()
            .any(|index| index.columns == vec!["text".to_string()])
        {
            return Ok(());
        }

        chunks_table
            .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
            .execute()
            .await?;
        Ok(())
    }

    fn documents_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("document_id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]))
    }

    fn chunks_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("document_id", DataType::Utf8, false),
            Field::new("chunk_index", DataType::UInt64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.vector_dimensions,
                ),
                false,
            ),
        ]))
    }

    fn documents_batch(&self, data: &[DocumentRecord]) -> Result<RecordBatch> {
        let document_id_values = Arc::new(StringArray::from_iter_values(
            data.iter().map(|document| document.document_id.as_str()),
        ));
        let content_values = Arc::new(StringArray::from_iter_values(
            data.iter().map(|document| document.content.as_str()),
        ));

        Ok(RecordBatch::try_new(
            self.documents_schema(),
            vec![document_id_values, content_values],
        )?)
    }

    fn chunks_batch(&self, data: &[Chunk]) -> Result<RecordBatch> {
        let expected_dimensions = self.vector_dimensions as usize;
        for chunk in data {
            if chunk.vector.len() != expected_dimensions {
                return Err(Error::InvalidInput {
                    message: format!(
                        "chunk vector has dimension {}, expected {}",
                        chunk.vector.len(),
                        expected_dimensions
                    ),
                });
            }
        }

        let document_id_values = Arc::new(StringArray::from_iter_values(
            data.iter().map(|chunk| chunk.document_id.as_str()),
        ));
        let chunk_index_values = Arc::new(UInt64Array::from_iter_values(
            data.iter().map(|chunk| chunk.chunk_index),
        ));
        let text_values = Arc::new(StringArray::from_iter_values(
            data.iter().map(|chunk| chunk.text.as_str()),
        ));
        let vector_values = Arc::new(
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                data.iter()
                    .map(|chunk| Some(chunk.vector.iter().copied().map(Some))),
                self.vector_dimensions,
            ),
        );

        Ok(RecordBatch::try_new(
            self.chunks_schema(),
            vec![
                document_id_values,
                chunk_index_values,
                text_values,
                vector_values,
            ],
        )?)
    }

    async fn merge_upsert_documents(&self, table: &Table, batch: RecordBatch) -> Result<()> {
        let mut merge = table.merge_insert(&["document_id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(record_batch_reader(batch)).await?;
        Ok(())
    }

    async fn merge_replace_document_chunks(
        &self,
        table: &Table,
        document_id: &str,
        batch: RecordBatch,
    ) -> Result<()> {
        let mut merge = table.merge_insert(&["document_id", "chunk_index"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all()
            .when_not_matched_by_source_delete(Some(document_id_predicate(document_id)));
        merge.execute(record_batch_reader(batch)).await?;
        Ok(())
    }

    async fn restore_document_chunks(
        &self,
        table: &Table,
        document_id: &str,
        chunks: Vec<Chunk>,
    ) -> Result<()> {
        if chunks.is_empty() {
            table.delete(&document_id_predicate(document_id)).await?;
            return Ok(());
        }

        let batch = self.chunks_batch(&chunks)?;
        self.merge_replace_document_chunks(table, document_id, batch)
            .await
    }

    async fn chunks_for_document(&self, document_id: &str) -> Result<Vec<Chunk>> {
        let table = self
            .connection
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await?;
        let rows = table
            .query()
            .only_if(document_id_predicate(document_id))
            .select(Select::columns(&[
                "document_id",
                "chunk_index",
                "text",
                "vector",
            ]))
            .execute()
            .await?;
        let batches = rows.try_collect::<Vec<_>>().await?;

        Ok(chunks_from_batches(&batches))
    }

    async fn delete_chunks_for_documents(&self, table: &Table, documents: &[DocumentRecord]) {
        for document in documents {
            let _ = table
                .delete(&document_id_predicate(&document.document_id))
                .await;
        }
    }

    async fn ensure_chunks_indices(&self, chunks_table: &Table) -> Result<()> {
        self.ensure_chunks_vector_index(chunks_table).await?;
        self.ensure_chunks_keyword_index(chunks_table).await
    }
}

fn record_batch_reader(batch: RecordBatch) -> Box<dyn arrow_array::RecordBatchReader + Send> {
    let schema = batch.schema();
    Box::new(RecordBatchIterator::new(
        vec![Ok(batch)].into_iter(),
        schema,
    ))
}

fn document_records_from_batches(batches: &[RecordBatch]) -> Vec<DocumentRecord> {
    batches
        .iter()
        .flat_map(|batch| {
            let document_ids = batch
                .column_by_name("document_id")
                .expect("documents query should include document_id")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("document_id column should be Utf8");
            let contents = batch
                .column_by_name("content")
                .expect("documents query should include content")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("content column should be Utf8");

            (0..batch.num_rows())
                .map(|index| DocumentRecord {
                    document_id: document_ids.value(index).to_string(),
                    content: contents.value(index).to_string(),
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn chunk_search_records_from_batches(batches: &[RecordBatch]) -> Vec<ChunkSearchRecord> {
    batches
        .iter()
        .flat_map(|batch| {
            let document_ids = batch
                .column_by_name("document_id")
                .expect("chunks query should include document_id")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("document_id column should be Utf8");
            let texts = batch
                .column_by_name("text")
                .expect("chunks query should include text")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("text column should be Utf8");
            let distances = batch
                .column_by_name("_distance")
                .expect("chunks query should include _distance")
                .as_primitive::<Float32Type>();

            (0..batch.num_rows())
                .map(|index| ChunkSearchRecord {
                    document_id: document_ids.value(index).to_string(),
                    text: texts.value(index).to_string(),
                    distance: distances.value(index),
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn chunk_keyword_search_records_from_batches(
    batches: &[RecordBatch],
) -> Vec<ChunkKeywordSearchRecord> {
    batches
        .iter()
        .flat_map(|batch| {
            let document_ids = batch
                .column_by_name("document_id")
                .expect("chunks query should include document_id")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("document_id column should be Utf8");
            let texts = batch
                .column_by_name("text")
                .expect("chunks query should include text")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("text column should be Utf8");
            let scores = batch
                .column_by_name("_score")
                .expect("chunks query should include _score")
                .as_primitive::<Float32Type>();

            (0..batch.num_rows())
                .map(|index| ChunkKeywordSearchRecord {
                    document_id: document_ids.value(index).to_string(),
                    text: texts.value(index).to_string(),
                    score: scores.value(index),
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn chunks_from_batches(batches: &[RecordBatch]) -> Vec<Chunk> {
    batches
        .iter()
        .flat_map(|batch| {
            let document_ids = batch
                .column_by_name("document_id")
                .expect("chunks query should include document_id")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("document_id column should be Utf8");
            let chunk_indices = batch
                .column_by_name("chunk_index")
                .expect("chunks query should include chunk_index")
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("chunk_index column should be UInt64");
            let texts = batch
                .column_by_name("text")
                .expect("chunks query should include text")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("text column should be Utf8");
            let vectors = batch
                .column_by_name("vector")
                .expect("chunks query should include vector")
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .expect("vector column should be FixedSizeList");

            (0..batch.num_rows())
                .map(|index| Chunk {
                    document_id: document_ids.value(index).to_string(),
                    chunk_index: chunk_indices.value(index),
                    text: texts.value(index).to_string(),
                    vector: vectors
                        .value(index)
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("vector item column should be Float32")
                        .values()
                        .to_vec(),
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn document_id_predicate(document_id: &str) -> String {
    format!("document_id = '{}'", document_id.replace('\'', "''"))
}

#[cfg(test)]
mod tests {
    use super::{CHUNKS_TABLE_NAME, Chunk, DOCUMENTS_TABLE_NAME, DocumentRecord, LanceDbBackend};
    use arrow_array::StringArray;
    use futures::TryStreamExt;
    use lancedb::Error;
    use lancedb::query::ExecutableQuery;

    fn demo_document() -> DocumentRecord {
        DocumentRecord {
            document_id: "demo-doc".to_string(),
            content: "knight ranger priest rogue".to_string(),
        }
    }

    fn demo_chunks() -> Vec<Chunk> {
        vec![
            Chunk {
                document_id: "demo-doc".to_string(),
                chunk_index: 0,
                text: "knight".to_string(),
                vector: vec![0.9, 0.4, 0.8],
            },
            Chunk {
                document_id: "demo-doc".to_string(),
                chunk_index: 1,
                text: "ranger".to_string(),
                vector: vec![0.8, 0.4, 0.7],
            },
            Chunk {
                document_id: "demo-doc".to_string(),
                chunk_index: 2,
                text: "priest".to_string(),
                vector: vec![0.6, 0.2, 0.6],
            },
            Chunk {
                document_id: "demo-doc".to_string(),
                chunk_index: 3,
                text: "rogue".to_string(),
                vector: vec![0.7, 0.4, 0.7],
            },
        ]
    }

    fn five_dimensional_chunks() -> Vec<Chunk> {
        vec![
            Chunk {
                document_id: "five-dim-doc".to_string(),
                chunk_index: 0,
                text: "mage".to_string(),
                vector: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            },
            Chunk {
                document_id: "five-dim-doc".to_string(),
                chunk_index: 1,
                text: "paladin".to_string(),
                vector: vec![0.5, 0.4, 0.3, 0.2, 0.1],
            },
        ]
    }

    #[tokio::test]
    async fn initializes_lancedb_from_local_path() {
        let temp_dir = tempfile::tempdir().unwrap();

        let backend = LanceDbBackend::new(temp_dir.path(), 3).await;

        assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn creates_empty_tables() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 5).await.unwrap();

        backend.create_tables().await.unwrap();

        assert_eq!(backend.vector_dimensions(), 5);
        let documents_table = backend
            .connection()
            .open_table(DOCUMENTS_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        let chunks_table = backend
            .connection()
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        assert_eq!(documents_table.count_rows(None).await.unwrap(), 0);
        assert_eq!(chunks_table.count_rows(None).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn inserts_demo_rows_into_tables() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 3).await.unwrap();

        backend.create_tables().await.unwrap();
        backend
            .insert_data(&[demo_document()], &demo_chunks())
            .await
            .unwrap();

        let documents_table = backend
            .connection()
            .open_table(DOCUMENTS_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        let chunks_table = backend
            .connection()
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        assert_eq!(documents_table.count_rows(None).await.unwrap(), 1);
        assert_eq!(chunks_table.count_rows(None).await.unwrap(), 4);
    }

    #[tokio::test]
    async fn create_tables_is_idempotent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 3).await.unwrap();

        backend.create_tables().await.unwrap();
        backend
            .insert_data(&[demo_document()], &demo_chunks())
            .await
            .unwrap();
        backend.create_tables().await.unwrap();

        let chunks_table = backend
            .connection()
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        assert_eq!(chunks_table.count_rows(None).await.unwrap(), 4);
    }

    #[tokio::test]
    async fn inserts_matching_non_default_dimensions() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 5).await.unwrap();

        backend.create_tables().await.unwrap();
        backend
            .insert_data(
                &[DocumentRecord {
                    document_id: "five-dim-doc".to_string(),
                    content: "mage paladin".to_string(),
                }],
                &five_dimensional_chunks(),
            )
            .await
            .unwrap();

        let chunks_table = backend
            .connection()
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        assert_eq!(chunks_table.count_rows(None).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn rejects_mismatched_vector_dimensions() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 5).await.unwrap();

        backend.create_tables().await.unwrap();
        let error = backend
            .insert_data(&[demo_document()], &demo_chunks())
            .await
            .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn vector_search_returns_ranked_chunk_records() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 3).await.unwrap();

        backend.create_tables().await.unwrap();
        backend
            .insert_data(&[demo_document()], &demo_chunks())
            .await
            .unwrap();

        let results = backend.vector_search(vec![0.9, 0.4, 0.8], 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document_id, "demo-doc");
        assert_eq!(results[0].text, "knight");
        assert_eq!(results[0].distance, 0.0);
    }

    #[tokio::test]
    async fn keyword_search_returns_ranked_chunk_records() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 3).await.unwrap();

        backend.create_tables().await.unwrap();
        backend
            .insert_data(
                &[DocumentRecord {
                    document_id: "search-doc".to_string(),
                    content: "rust database rust search ranger".to_string(),
                }],
                &[
                    Chunk {
                        document_id: "search-doc".to_string(),
                        chunk_index: 0,
                        text: "rust database rust search".to_string(),
                        vector: vec![0.1, 0.2, 0.3],
                    },
                    Chunk {
                        document_id: "search-doc".to_string(),
                        chunk_index: 1,
                        text: "ranger path".to_string(),
                        vector: vec![0.4, 0.5, 0.6],
                    },
                ],
            )
            .await
            .unwrap();

        let results = backend.keyword_search("rust".to_string(), 1).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, "search-doc");
        assert_eq!(results[0].text, "rust database rust search");
        assert!(results[0].score > 0.0);
    }

    #[tokio::test]
    async fn keyword_search_returns_empty_for_missing_terms() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 3).await.unwrap();

        backend.create_tables().await.unwrap();
        backend
            .insert_data(&[demo_document()], &demo_chunks())
            .await
            .unwrap();

        let results = backend
            .keyword_search("warlock".to_string(), 10)
            .await
            .unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn upsert_replaces_rows_for_document_id() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 3).await.unwrap();

        backend.create_tables().await.unwrap();
        backend
            .insert_data(&[demo_document()], &demo_chunks())
            .await
            .unwrap();
        backend
            .upsert_data(
                &DocumentRecord {
                    document_id: "demo-doc".to_string(),
                    content: "replacement next".to_string(),
                },
                &[
                    Chunk {
                        document_id: "demo-doc".to_string(),
                        chunk_index: 0,
                        text: "replacement".to_string(),
                        vector: vec![0.1, 0.2, 0.3],
                    },
                    Chunk {
                        document_id: "demo-doc".to_string(),
                        chunk_index: 1,
                        text: "next".to_string(),
                        vector: vec![0.4, 0.5, 0.6],
                    },
                ],
            )
            .await
            .unwrap();

        let table = backend
            .connection()
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 2);

        let rows = table.query().execute().await.unwrap();
        let batches = rows.try_collect::<Vec<_>>().await.unwrap();
        let texts = batches
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
            .collect::<Vec<_>>();

        assert_eq!(texts, vec!["replacement".to_string(), "next".to_string()]);
        assert_eq!(
            backend.get_document("demo-doc").await.unwrap(),
            Some(DocumentRecord {
                document_id: "demo-doc".to_string(),
                content: "replacement next".to_string(),
            })
        );
    }

    #[tokio::test]
    async fn upsert_preserves_repeated_chunk_text_by_index() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 3).await.unwrap();

        backend.create_tables().await.unwrap();
        backend
            .insert_data(&[demo_document()], &demo_chunks())
            .await
            .unwrap();
        backend
            .upsert_data(
                &DocumentRecord {
                    document_id: "demo-doc".to_string(),
                    content: "repeat repeat".to_string(),
                },
                &[
                    Chunk {
                        document_id: "demo-doc".to_string(),
                        chunk_index: 0,
                        text: "repeat".to_string(),
                        vector: vec![0.1, 0.2, 0.3],
                    },
                    Chunk {
                        document_id: "demo-doc".to_string(),
                        chunk_index: 1,
                        text: "repeat".to_string(),
                        vector: vec![0.4, 0.5, 0.6],
                    },
                ],
            )
            .await
            .unwrap();

        let table = backend
            .connection()
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        let rows = table.query().execute().await.unwrap();
        let batches = rows.try_collect::<Vec<_>>().await.unwrap();
        let texts = batches
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
            .collect::<Vec<_>>();

        assert_eq!(texts, vec!["repeat".to_string(), "repeat".to_string()]);
    }

    #[tokio::test]
    async fn upsert_escapes_document_id_predicate() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 3).await.unwrap();

        backend.create_tables().await.unwrap();
        backend
            .insert_data(
                &[DocumentRecord {
                    document_id: "doc-'quoted'".to_string(),
                    content: "old".to_string(),
                }],
                &[Chunk {
                    document_id: "doc-'quoted'".to_string(),
                    chunk_index: 0,
                    text: "old".to_string(),
                    vector: vec![0.1, 0.2, 0.3],
                }],
            )
            .await
            .unwrap();
        backend
            .upsert_data(
                &DocumentRecord {
                    document_id: "doc-'quoted'".to_string(),
                    content: "new".to_string(),
                },
                &[Chunk {
                    document_id: "doc-'quoted'".to_string(),
                    chunk_index: 0,
                    text: "new".to_string(),
                    vector: vec![0.4, 0.5, 0.6],
                }],
            )
            .await
            .unwrap();

        let table = backend
            .connection()
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 1);
        assert_eq!(
            backend.get_document("doc-'quoted'").await.unwrap(),
            Some(DocumentRecord {
                document_id: "doc-'quoted'".to_string(),
                content: "new".to_string(),
            })
        );
    }

    #[tokio::test]
    async fn lists_documents_sorted_by_document_id() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 3).await.unwrap();

        backend.create_tables().await.unwrap();
        backend
            .insert_data(
                &[
                    DocumentRecord {
                        document_id: "b-doc".to_string(),
                        content: "second".to_string(),
                    },
                    DocumentRecord {
                        document_id: "a-doc".to_string(),
                        content: "first".to_string(),
                    },
                ],
                &[
                    Chunk {
                        document_id: "b-doc".to_string(),
                        chunk_index: 0,
                        text: "second".to_string(),
                        vector: vec![0.1, 0.2, 0.3],
                    },
                    Chunk {
                        document_id: "a-doc".to_string(),
                        chunk_index: 0,
                        text: "first".to_string(),
                        vector: vec![0.4, 0.5, 0.6],
                    },
                ],
            )
            .await
            .unwrap();

        let documents = backend.list_documents().await.unwrap();

        assert_eq!(
            documents,
            vec![
                DocumentRecord {
                    document_id: "a-doc".to_string(),
                    content: "first".to_string(),
                },
                DocumentRecord {
                    document_id: "b-doc".to_string(),
                    content: "second".to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn delete_document_removes_document_and_chunks() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 3).await.unwrap();

        backend.create_tables().await.unwrap();
        backend
            .insert_data(&[demo_document()], &demo_chunks())
            .await
            .unwrap();

        backend.delete_document("demo-doc").await.unwrap();

        let documents_table = backend
            .connection()
            .open_table(DOCUMENTS_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        let chunks_table = backend
            .connection()
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        assert_eq!(documents_table.count_rows(None).await.unwrap(), 0);
        assert_eq!(chunks_table.count_rows(None).await.unwrap(), 0);
    }
}
