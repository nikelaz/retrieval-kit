pub mod backends;

pub struct RKit;

impl RKit {
    pub fn select_db_engine(db_engine: String) {
        let _ = db_engine;
        panic!("not implemented");
    }

    pub fn register_embeddings_provider(embeddings_provider: String) {
        let _ = embeddings_provider;
        panic!("not implemented");
    }

    pub fn ingest_document(content: String) {
        let _ = content;
        panic!("not implemented");
    }

    pub fn upsert_document(id: String, content: String) {
        let _ = (id, content);
        panic!("not implemented");
    }

    pub fn ingest_documents(glob: String) {
        let _ = glob;
        panic!("not implemented");
    }

    pub fn vector_search(query: String) {
        let _ = query;
        panic!("not implemented");
    }

    pub fn keyword_search(query: String) {
        let _ = query;
        panic!("not implemented");
    }

    pub fn list_documents() {
        panic!("not implemented");
    }

    pub fn get_document(id: String) {
        let _ = id;
        panic!("not implemented");
    }

    pub fn delete_document(id: String) {
        let _ = id;
        panic!("not implemented");
    }

    pub fn get_tool_definitions() {
        panic!("not implemented");
    }

    pub fn invoke_tool() {
        panic!("not implemented");
    }
}
