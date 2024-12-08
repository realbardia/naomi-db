use serde::{Deserialize, Serialize};
use crate::models::general::*;
use crate::models::database::*;
use crate::utils::ollama::Ollama;
use actix_web::*;
use actix_web::web;
use md5;

use chromadb::v2::client::ChromaClient;
use chromadb::v2::collection::{ChromaCollection, CollectionEntries, QueryOptions};

#[derive(Clone, Serialize, Deserialize)]
pub struct PostDatabaseReq {
    pub text: Vec<String>,
    pub model: Option<String>,
    pub collection: String,
    pub translate_to: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PostDatabaseResult {
    pub text: String,
    pub english: Option<String>,
    pub id: String,
}


#[derive(Clone, Serialize, Deserialize)]
pub struct FindDatabaseReq {
    pub text: String,
    pub model: Option<String>,
    pub collection: String,
    pub translate_to: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FindDatabaseResult {
    pub id: String,
    pub text: String,
    pub distance: f32,
}

impl Database {
	pub async fn insert(data: web::Json<PostDatabaseReq>) -> impl Responder {
        let model = data.model.clone().unwrap_or("nomic-embed-text".to_string());

        let mut results: Vec<PostDatabaseResult> = Vec::new();
        let mut embeddings_list: Vec<Vec<f32>> = Vec::new();
        if data.translate_to != None && data.translate_to != Some(String::new()) {
            for t in data.text.clone() {
                let prompt = format!("Please translate below text to {} without any extra explation and type exact text if It already translated:\n{}", data.translate_to.clone().unwrap().as_str(), t.as_str());
                let english = Ollama::generate(prompt, "gemma2".to_string()).await.unwrap();
                let embeddings = Ollama::embedding(english.clone(), model.clone()).await;

                let md5 = md5::compute(t.clone());
                let id_hash: String = format!("{:x}", md5);

                embeddings_list.push(embeddings.unwrap());
                results.push(PostDatabaseResult {
                    id: id_hash,
                    text: t,
                    english: Some(english),
                });
            }
        } else {
            for t in data.text.clone() {
                let embeddings = Ollama::embedding(t.clone(), model.clone()).await;

                let md5 = md5::compute(t.clone());
                let id_hash: String = format!("{:x}", md5);

                embeddings_list.push(embeddings.unwrap());
                results.push(PostDatabaseResult {
                    id: id_hash,
                    text: t,
                    english: None,
                });
            }
        }

        let collection_name: String = data.collection.clone();
        let chroma: ChromaClient = ChromaClient::new(Default::default());
        let collection: ChromaCollection = chroma.get_or_create_collection(collection_name.as_str(), None).await.unwrap();

        let mut ids: Vec<&str> = Vec::new();
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        let mut documents: Vec<&str> = Vec::new();
        let mut idx = 0;
        for r in &results {
            ids.push(r.id.as_str());
            embeddings.push(embeddings_list.get(idx).unwrap().clone());
            documents.push(r.text.as_str());
            idx = idx + 1;
        }

        let collection_entries = CollectionEntries {
            ids: ids,
            embeddings: Some(embeddings),
            metadatas: None,
            documents: Some(documents)
        };
         
        let _result = collection.upsert(collection_entries, None).await;

        HttpResponse::Ok().json(GeneralValueResult{result: results, status: true})
	} 

	pub async fn find(data: web::Json<FindDatabaseReq>) -> impl Responder {
        let model = data.model.clone().unwrap_or("nomic-embed-text".to_string());
        
        let embedding: Vec<f32>;
        if data.translate_to != None && data.translate_to != Some(String::new()) {
            let prompt = format!("Please translate below text to {} without any extra explation and type exact text if It already translated:\n{}", data.translate_to.clone().unwrap().as_str(), data.text.as_str());
            let english = Ollama::generate(prompt, "gemma2".to_string()).await.unwrap();
            let embeddings = Ollama::embedding(english.clone(), model.clone()).await;
            embedding = embeddings.unwrap();
        } else {
            let embeddings = Ollama::embedding(data.text.clone(), model.clone()).await;
            embedding = embeddings.unwrap();
        }

        let collection_name: String = data.collection.clone();
        let chroma: ChromaClient = ChromaClient::new(Default::default());
        let collection: ChromaCollection = chroma.get_or_create_collection(collection_name.as_str(), None).await.unwrap();

        let query = QueryOptions {
            query_texts: None,
            query_embeddings: Some(vec![embedding]),
            where_metadata: None,
            where_document: None,
            n_results: Some(5),
            include: Some(vec!["documents".into(), "distances".into()])
        };
         
        let result = collection.query(query, None).await;
        match result {
            Ok(r) => {
                let ids = r.ids.get(0).unwrap();
                let distances = r.distances.unwrap().get(0).unwrap().clone();
                let documents = r.documents.unwrap().get(0).unwrap().clone();

                let mut list: Vec<FindDatabaseResult> = Vec::new();
                let mut idx = 0;
                for id in ids {
                    list.push(FindDatabaseResult {
                        id: id.clone(),
                        text: documents.get(idx).unwrap().clone(),
                        distance: distances.get(idx).unwrap().clone(),
                    });
                    idx = idx + 1;
                }

                HttpResponse::Ok().json(GeneralValueResult{result: list, status: true})
            },
            Err(_e) => HttpResponse::InternalServerError().json(ErrorResult {status: false, message: None})
        }
	} 
}
