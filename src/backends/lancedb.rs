use std::path::Path;
use std::sync::Arc;

use arrow_array::{FixedSizeListArray, RecordBatch, StringArray, types::Float32Type};
use arrow_schema::{DataType, Field, Schema};
use lancedb::database::CreateTableMode;
use lancedb::{Connection, Error, Result, Table, connect};

const ARTICLES_TABLE_NAME: &str = "articles";

pub struct Article {
    pub text: String,
    pub vector: Vec<f32>,
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

    pub async fn create_table(&self) -> Result<Table> {
        self.connection
            .create_empty_table(ARTICLES_TABLE_NAME, self.articles_schema())
            .mode(CreateTableMode::Overwrite)
            .execute()
            .await
    }

    pub async fn insert_data(&self, data: &[Article]) -> Result<()> {
        let table = self
            .connection
            .open_table(ARTICLES_TABLE_NAME)
            .execute()
            .await?;

        table.add(self.articles_batch(data)?).execute().await?;

        Ok(())
    }

    pub fn connection(&self) -> &Connection {
        &self.connection
    }

    pub fn vector_dimensions(&self) -> i32 {
        self.vector_dimensions
    }

    fn articles_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
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

    fn articles_batch(&self, data: &[Article]) -> Result<RecordBatch> {
        let expected_dimensions = self.vector_dimensions as usize;
        for article in data {
            if article.vector.len() != expected_dimensions {
                return Err(Error::InvalidInput {
                    message: format!(
                        "article vector has dimension {}, expected {}",
                        article.vector.len(),
                        expected_dimensions
                    ),
                });
            }
        }

        let text_values = Arc::new(StringArray::from_iter_values(
            data.iter().map(|article| article.text.as_str()),
        ));
        let vector_values = Arc::new(
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                data.iter()
                    .map(|article| Some(article.vector.iter().copied().map(Some))),
                self.vector_dimensions,
            ),
        );

        Ok(RecordBatch::try_new(
            self.articles_schema(),
            vec![text_values, vector_values],
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::{ARTICLES_TABLE_NAME, Article, LanceDbBackend};
    use lancedb::Error;

    fn demo_articles() -> Vec<Article> {
        vec![
            Article {
                text: "knight".to_string(),
                vector: vec![0.9, 0.4, 0.8],
            },
            Article {
                text: "ranger".to_string(),
                vector: vec![0.8, 0.4, 0.7],
            },
            Article {
                text: "priest".to_string(),
                vector: vec![0.6, 0.2, 0.6],
            },
            Article {
                text: "rogue".to_string(),
                vector: vec![0.7, 0.4, 0.7],
            },
        ]
    }

    fn five_dimensional_articles() -> Vec<Article> {
        vec![
            Article {
                text: "mage".to_string(),
                vector: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            },
            Article {
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
    async fn creates_empty_articles_table() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 5).await.unwrap();

        let table = backend.create_table().await.unwrap();

        assert_eq!(backend.vector_dimensions(), 5);
        assert_eq!(table.count_rows(None).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn inserts_demo_rows_into_articles_table() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 3).await.unwrap();

        backend.create_table().await.unwrap();
        backend.insert_data(&demo_articles()).await.unwrap();

        let reopened = backend
            .connection()
            .open_table(ARTICLES_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        assert_eq!(reopened.count_rows(None).await.unwrap(), 4);
    }

    #[tokio::test]
    async fn inserts_matching_non_default_dimensions() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 5).await.unwrap();

        backend.create_table().await.unwrap();
        backend
            .insert_data(&five_dimensional_articles())
            .await
            .unwrap();

        let reopened = backend
            .connection()
            .open_table(ARTICLES_TABLE_NAME)
            .execute()
            .await
            .unwrap();
        assert_eq!(reopened.count_rows(None).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn rejects_mismatched_vector_dimensions() {
        let temp_dir = tempfile::tempdir().unwrap();
        let backend = LanceDbBackend::new(temp_dir.path(), 5).await.unwrap();

        backend.create_table().await.unwrap();
        let error = backend.insert_data(&demo_articles()).await.unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
    }
}
