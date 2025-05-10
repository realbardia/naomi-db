use chromadb::v2::client::ChromaClientOptions;
use serde::{Deserialize, Serialize};
use crate::models::general::*;
use crate::models::database::*;
use crate::utils::ollama::Ollama;
use actix_web::*;
use actix_web::web;
use md5;

use serde_json::{Map, Value};
use chromadb::v2::client::ChromaClient;
use chromadb::v2::collection::{ChromaCollection, CollectionEntries, QueryOptions};


#[derive(Clone, Serialize, Deserialize)]
pub struct PostDatabaseItem {
    pub id: Option<String>,
    pub text: String,
    pub metadata: Option<Map<String, Value>>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)] // Allows the enum to match different JSON structures
pub enum DataType {
    StringList(Vec<String>),
    ItemList(Vec<PostDatabaseItem>),
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PostDatabaseReq {
    pub data: DataType,
    pub model: Option<String>,
    pub collection: String,
    pub translate_to: Option<String>,
    pub calculate_nearest: Option<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PostDatabaseResult {
    pub text: String,
    pub metadata: Option<Map<String, Value>>,
    pub english: Option<String>,
    pub embeddings: Vec<f32>,
    pub id: String,
}


#[derive(Clone, Serialize, Deserialize)]
pub struct PostEmbeddingsItem {
    pub id: String,
    pub text: String,
    pub metadata: Option<Map<String, Value>>,
    pub english: Option<String>,
    pub embeddings: Vec<f32>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PostEmbeddingsReq {
    pub data: Vec<PostEmbeddingsItem>,
    pub collection: String,
    pub calculate_nearest: Option<usize>,
}


#[derive(Clone, Serialize, Deserialize)]
pub struct FindDatabaseReq {
    pub text: String,
    pub model: Option<String>,
    pub collection: String,
    pub translate_to: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FindDatabaseResult {
    pub id: String,
    pub text: String,
    pub metadata: Option<Map<String, Value>>,
    pub distance: f32,
}

const DEFAULT_EMBEDDING_MODEL: &str = "bge-m3";
const DEFAULT_PROMPT_MODEL: &str = "gemma2";

fn translate_prompt(prompt: String, lang: String) -> String {
    let res = format!("Please translate below text to {} without any extra explation and type exact text if It already translated:\n{}", lang.as_str(), prompt.as_str());
    res.to_string()
}

impl Database {

    pub fn get_chroma_port() -> i32 {
        use std::env;
        env::var("CHROMA_PORT").unwrap_or("9011".to_string()).parse().unwrap()
    }

	pub async fn insert(data: web::Json<PostDatabaseReq>) -> impl Responder {
        let model = data.model.clone().unwrap_or(DEFAULT_EMBEDDING_MODEL.to_string());

        let mut data_list: Vec<PostDatabaseItem> = Vec::new();
        match &data.data {
            DataType::StringList(list) => {
                for s in list {
                    data_list.push(PostDatabaseItem {
                        id: None,
                        text: s.clone(),
                        metadata: None,
                    });
                }
            }
            DataType::ItemList(list) => {
                data_list = list.clone();
            }
        }

        let mut results: Vec<PostDatabaseResult> = Vec::new();
        let mut embeddings_list: Vec<Vec<f32>> = Vec::new();
        if data.translate_to != None && data.translate_to != Some(String::new()) {
            for item in data_list.clone() {
                let t: String = item.text;
                let prompt = translate_prompt(t.clone(), data.translate_to.clone().unwrap());
                let english = Ollama::generate(prompt, DEFAULT_PROMPT_MODEL.to_string()).await.unwrap();
                let embeddings = Ollama::embedding(english.clone(), model.clone()).await;

                let md5 = md5::compute(t.clone());
                let id_hash: String = format!("{:x}", md5);

                embeddings_list.push(embeddings.clone().unwrap());
                results.push(PostDatabaseResult {
                    id: id_hash,
                    text: t,
                    metadata: item.metadata,
                    english: Some(english),
                    embeddings: embeddings.unwrap(),
                });
            }
        } else {
            for item in data_list.clone() {
                let t: String = item.text;
                let id: String = match item.id {
                    Some(id) => id,
                    None => {
                        let md5 = md5::compute(t.clone());
                        format!("{:x}", md5)
                    }
                };

                let embeddings = Ollama::embedding(t.clone(), model.clone()).await;

                embeddings_list.push(embeddings.clone().unwrap());
                results.push(PostDatabaseResult {
                    id: id,
                    text: t,
                    metadata: item.metadata,
                    english: None,
                    embeddings: embeddings.unwrap(),
                });
            }
        }

        let collection_name: String = data.collection.clone();
        let mut chroma_options: ChromaClientOptions = Default::default();
        chroma_options.url = "http://127.0.0.1:".to_string() + Database::get_chroma_port().to_string().as_str();

        let chroma: ChromaClient = ChromaClient::new(chroma_options);
        let collection: ChromaCollection = chroma.get_or_create_collection(collection_name.as_str(), None).await.unwrap();

        let mut ids: Vec<&str> = Vec::new();
        let mut documents: Vec<&str> = Vec::new();
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        let mut metadatas: Vec<Map<String, Value>> = Vec::new();
        let mut idx = 0;
        for r in &mut results {
            let mut metadata = match r.metadata.clone() {
                Some(map) => map,
                None => Map::new(),
            };

            if data.calculate_nearest != None {
                let mut mid_distance: f32 = 0.0;
                let nearests = Database::find_nearest(collection_name.clone(), embeddings_list.get(idx).unwrap().clone(), data.calculate_nearest).await;
                match nearests {
                    Ok (list) => {
                        let mut len: f32 = 0.0;
                        for n in list {
                            mid_distance = mid_distance + n.distance;
                            len = len + 1.0;
                        }
                        mid_distance = mid_distance / len;
                    },
                    Err(_) => {}
                }
                metadata.insert("mid_distance".to_string(), Value::from(mid_distance));
            }
            r.metadata = Some(metadata.clone());

            ids.push(r.id.as_str());
            documents.push(&r.text.as_str());
            embeddings.push(embeddings_list.get(idx).unwrap().clone());
            metadatas.push(metadata);
            idx = idx + 1;
        }

        let collection_entries = CollectionEntries {
            ids: ids,
            embeddings: Some(embeddings),
            metadatas: Some(metadatas),
            documents: Some(documents),
        };
         
        let _result = collection.upsert(collection_entries, None).await;

        HttpResponse::Ok().json(GeneralValueResult{result: results, status: true})
	} 

	pub async fn insert_embeddings(data: web::Json<PostEmbeddingsReq>) -> impl Responder {
        let collection_name: String = data.collection.clone();
        let mut chroma_options: ChromaClientOptions = Default::default();
        chroma_options.url = "http://127.0.0.1:".to_string() + Database::get_chroma_port().to_string().as_str();

        let chroma: ChromaClient = ChromaClient::new(chroma_options);
        let collection: ChromaCollection = chroma.get_or_create_collection(collection_name.as_str(), None).await.unwrap();

        let mut result: Vec<String> = Vec::new();
        let mut documents_strs: Vec<String> = Vec::new();

        let mut ids: Vec<&str> = Vec::new();
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        let mut documents: Vec<&str> = Vec::new();
        let mut metadatas: Vec<Map<String, Value>> = Vec::new();
        
        for item in data.data.clone() {
            let mut metadata = match item.metadata.clone() {
                Some(map) => map,
                None => Map::new(),
            };

            if data.calculate_nearest != None {
                let mut mid_distance: f32 = 0.0;
                let nearests = Database::find_nearest(collection_name.clone(), item.embeddings.clone(), data.calculate_nearest).await;
                match nearests {
                    Ok (list) => {
                        let mut len: f32 = 0.0;
                        for n in list {
                            mid_distance = mid_distance + n.distance;
                            len = len + 1.0;
                        }
                        mid_distance = mid_distance / len;
                    },
                    Err(_) => {}
                }
                metadata.insert("mid_distance".to_string(), Value::from(mid_distance));
            }

            result.push(item.id);
            documents_strs.push(item.text);
            embeddings.push(item.embeddings.clone());
            metadatas.push(metadata);
        }
        for id in &result {
            ids.push(id.as_str());
        }
        for str in &documents_strs {
            documents.push(str.as_str());
        }

        let collection_entries = CollectionEntries {
            ids: ids,
            embeddings: Some(embeddings),
            metadatas: Some(metadatas),
            documents: Some(documents),
        };
         
        let _result = collection.upsert(collection_entries, None).await;

        HttpResponse::Ok().json(GeneralValueResult{result: result, status: true})
	} 

    async fn find_nearest(collection_name: String, embedding: Vec<f32>, limit: Option<usize>) -> Result<Vec<FindDatabaseResult>, bool> {
        let mut chroma_options: ChromaClientOptions = Default::default();
        chroma_options.url = "http://127.0.0.1:".to_string() + Database::get_chroma_port().to_string().as_str();

        let chroma: ChromaClient = ChromaClient::new(chroma_options);
        let collection: ChromaCollection = chroma.get_or_create_collection(collection_name.as_str(), None).await.unwrap();

        let query = QueryOptions {
            query_texts: None,
            query_embeddings: Some(vec![embedding]),
            where_metadata: None,
            where_document: None,
            n_results: if limit == None { Some(10) } else { limit },
            include: Some(vec!["documents".into(), "distances".into(), "metadatas".into()])
        };

        let result = collection.query(query, None).await;
        match result {
            Ok(r) => {
                let ids = r.ids.get(0).unwrap();
                let distances = r.distances.unwrap().get(0).unwrap().clone();
                let documents = r.documents.unwrap().get(0).unwrap().clone();
                let metadatas = r.metadatas.unwrap().get(0).unwrap().clone();

                let mut list: Vec<FindDatabaseResult> = Vec::new();
                let mut idx = 0;
                for id in ids {
                    list.push(FindDatabaseResult {
                        id: id.clone(),
                        metadata: metadatas.get(idx).unwrap().clone(),
                        text: documents.get(idx).unwrap().clone(),
                        distance: distances.get(idx).unwrap().clone(),
                    });
                    idx = idx + 1;
                }

                Ok(list)
            },
            Err(e) => {
                println!("{}", e);
                Err(false)
            }
        }
    }

	pub async fn find(data: web::Json<FindDatabaseReq>) -> impl Responder {
        let model = data.model.clone().unwrap_or(DEFAULT_EMBEDDING_MODEL.to_string());
        
        let embedding: Vec<f32>;
        if data.translate_to != None && data.translate_to != Some(String::new()) {
            let prompt = translate_prompt(data.text.clone(), data.translate_to.clone().unwrap());
            let english = Ollama::generate(prompt, DEFAULT_PROMPT_MODEL.to_string()).await.unwrap();
            let embeddings = Ollama::embedding(english.clone(), model.clone()).await;
            embedding = embeddings.unwrap();
        } else {
            let embeddings = Ollama::embedding(data.text.clone(), model.clone()).await;
            embedding = embeddings.unwrap();
        }

        let collection_name: String = data.collection.clone();
        let nearests = Database::find_nearest(collection_name, embedding, data.limit).await;
        match nearests {
            Ok(r) => {
                HttpResponse::Ok().json(GeneralValueResult{result: r, status: true})
            },
            Err(_) => {
                HttpResponse::InternalServerError().json(ErrorResult {status: false, message: None})
            }
        }
	} 
}
