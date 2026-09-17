use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use crate::models::general::*;
use crate::models::database::*;
use crate::utils::ollama::Ollama;
use actix_web::*;
use actix_web::web;
use md5;

use serde_json::{json, Map, Value};

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    UpsertPointsBuilder, 
    CreateCollectionBuilder, 
    Distance, 
    PointStruct, 
    VectorParams, 
    VectorsConfig, 
    QueryPointsBuilder, 
    QueryBatchPointsBuilder,
    PointId, 
    Filter,
    Condition,
    Range,
    ScoredPoint,
};
use uuid::Uuid;


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
pub struct FindDatabaseFilterReq {
    pub field: String,
    pub value: Value,
    pub operator: String,
}


#[derive(Clone, Serialize, Deserialize)]
pub struct FindDatabaseReq {
    pub texts: Option<Vec<String>>,
    pub model: Option<String>,
    pub collection: String,
    pub translate_to: Option<String>,
    pub filters: Option<Vec<FindDatabaseFilterReq>>,
    pub filter_by_ids: Option<Vec<String>>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
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

    pub fn get_qdrant_port() -> i32 {
        use std::env;
        env::var("QDRANT__SERVICE__GRPC_PORT").unwrap_or("6188".to_string()).parse().unwrap()
    }

    async fn create_client(collection_name: String) -> Qdrant {
        let client = Qdrant::from_url(&("http://localhost:".to_string() + &Database::get_qdrant_port().to_string())).build().unwrap();
        if !client.collection_exists(&collection_name).await.unwrap() {
            println!("There is no '{}' collection. Creating...", collection_name);
            client
                .create_collection(
                    CreateCollectionBuilder::new(collection_name.clone()).vectors_config(
                        VectorsConfig {
                            config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                                VectorParams {
                                    size: 1024,
                                    distance: Distance::Cosine.into(),
                                    ..Default::default()
                                },
                            )),
                        }
                    )).await.unwrap();
            println!("Collection '{}' created successfully.", collection_name);
        }
 
        client
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
        let client = Database::create_client(collection_name.clone()).await;

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

                let mut input_embeddings: Vec<Vec<f32>> = Vec::new();
                input_embeddings.push(embeddings_list.get(idx).unwrap().clone());

                let nearests = Database::find_nearest(collection_name.clone(), Some(input_embeddings), data.calculate_nearest, None, None, None).await;
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

        let mut points_to_upsert: Vec<PointStruct> = Vec::new();
        for i in 0..ids.len() {
            let mut payload = metadatas[i].clone();
            payload.insert(
                "original_document".to_string(),
                json!(documents[i]),
            );

            let point_id =
                Uuid::new_v5(&Uuid::NAMESPACE_DNS, ids[i].as_bytes()).to_string();

            let point = PointStruct::new(
                point_id,
                embeddings[i].clone(),
                payload,
            );
            points_to_upsert.push(point);
        }

        if !points_to_upsert.is_empty() {
            let _result = client
                .upsert_points(UpsertPointsBuilder::new(collection_name, points_to_upsert))
                .await.unwrap();
        } else {
        }

        HttpResponse::Ok().json(GeneralValueResult{result: results, status: true})
	} 

	pub async fn insert_embeddings(data: web::Json<PostEmbeddingsReq>) -> impl Responder {
        let collection_name: String = data.collection.clone();
        let client = Database::create_client(collection_name.clone()).await;

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

                let mut input_embeddings: Vec<Vec<f32>> = Vec::new();
                input_embeddings.push(item.embeddings.clone());

                let nearests = Database::find_nearest(collection_name.clone(), Some(input_embeddings), data.calculate_nearest, None, None, None).await;
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

        let mut points_to_upsert: Vec<PointStruct> = Vec::new();
        for i in 0..ids.len() {
            let mut payload = metadatas[i].clone();
            payload.insert(
                "original_document".to_string(),
                json!(documents[i]),
            );

            let point_id =
                Uuid::new_v5(&Uuid::NAMESPACE_DNS, ids[i].as_bytes()).to_string();

            let point = PointStruct::new(
                point_id,
                embeddings[i].clone(),
                payload,
            );
            points_to_upsert.push(point);
        }

        if !points_to_upsert.is_empty() {
            let _result = client
                .upsert_points(UpsertPointsBuilder::new(collection_name, points_to_upsert))
                .await.unwrap();
        } else {
        }

        HttpResponse::Ok().json(GeneralValueResult{result: result, status: true})
	} 

    async fn find_nearest(collection_name: String, embeddings: Option<Vec<Vec<f32>>>, limit: Option<usize>, offset: Option<usize>, filters: Option<Vec<FindDatabaseFilterReq>>, filter_by_ids: Option<Vec<String>>) -> Result<Vec<FindDatabaseResult>, bool> {
        let client = Database::create_client(collection_name.clone()).await;
        let mut filter_conditions: Vec<Condition> = Vec::new();

        match filters {
            Some(list) => {
                for l in list {
                    match l.operator.as_str() {
                        ">" => {
                            filter_conditions.push(Condition::range(
                                l.field,
                                Range {
                                    gt: Some(l.value.as_f64().unwrap()),
                                    ..Default::default()
                                },
                            ));
                        },
                        ">=" => {
                            filter_conditions.push(Condition::range(
                                l.field,
                                Range {
                                    gte: Some(l.value.as_f64().unwrap()),
                                    ..Default::default()
                                },
                            ));
                        },
                        "<" => {
                            filter_conditions.push(Condition::range(
                                l.field,
                                Range {
                                    lt: Some(l.value.as_f64().unwrap()),
                                    ..Default::default()
                                },
                            ));
                        },
                        "<=" => {
                            filter_conditions.push(Condition::range(
                                l.field,
                                Range {
                                    lte: Some(l.value.as_f64().unwrap()),
                                    ..Default::default()
                                },
                            ));
                        },
                        "=" => {
                            match l.value.as_i64() {
                                Some(v) => {
                                    filter_conditions.push(Condition::matches(
                                        l.field,
                                        v,
                                    ));
                                },
                                None => {
                                    filter_conditions.push(Condition::matches_text(
                                        l.field,
                                        l.value.as_str().unwrap(),
                                    ));
                                }
                            }
                        },
                        _ => {
                        }
                    }
                }
            },
            None => {}
        }
        
        if let Some(ids) = filter_by_ids {
            if !ids.is_empty() {
                let point_ids: Vec<PointId> = ids.into_iter().map(PointId::from).collect();
                filter_conditions.push(Condition::has_id(point_ids));
            }
        }

        let filter = Filter::must(filter_conditions);

        let limit_val = limit.unwrap_or(10) as u64;
        let offset_val = offset.unwrap_or(0) as u64;

        let queries = match embeddings {
            Some(vecs) if !vecs.is_empty() => {
                vecs.into_iter().map(|vec| {
                    QueryPointsBuilder::new(&collection_name)
                        .query(vec)
                        .filter(filter.clone())
                        .limit(limit_val)
                        .offset(offset_val)
                        .with_payload(true)
                        .with_vectors(true)
                        .build()
                }).collect()
            },
            _ => {
                vec![
                    QueryPointsBuilder::new(&collection_name)
                        .filter(filter)
                        .limit(limit_val)
                        .offset(offset_val)
                        .with_payload(true)
                        .with_vectors(true)
                        .build()
                ]
            }
        };

        let batch_request = QueryBatchPointsBuilder::new(collection_name, queries);

        let search_result = client.query_batch(batch_request).await;
        match search_result {
            Ok(response) => {
                let mut merged_results: HashMap<String, ScoredPoint> = HashMap::new();

                for batch in response.result {
                    for point in batch.result {
                        let id_key = match &point.id {
                            Some(PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) }) => n.to_string(),
                            Some(PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s)) }) => s.clone(),
                            _ => continue,
                        };

                        merged_results
                            .entry(id_key)
                            .and_modify(|existing| {
                                if point.score > existing.score {
                                    *existing = point.clone();
                                }
                            })
                            .or_insert(point);
                    }
                }

                let mut all_points: Vec<ScoredPoint> = merged_results.into_values().collect();
                all_points.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

                let start = offset_val as usize;
                let req_limit = limit_val as usize;
                
                let final_points = if start >= all_points.len() {
                    Vec::new()
                } else {
                    let end = std::cmp::min(start + req_limit, all_points.len());
                    all_points[start..end].to_vec()
                };

                let mut list: Vec<FindDatabaseResult> = Vec::new();

                for scored_point in final_points {
                    let mut payload = scored_point.payload;

                    let text = payload
                        .remove("original_document")
                        .and_then(|v| Some(v.to_string()))
                        .unwrap_or_else(|| "".to_string());

                    let metadata = payload;

                    let id_str = match scored_point.id.unwrap() {
                        PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) } => n.to_string(),
                        PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s)) } => s,
                        _ => "".to_string(),
                    };

                    list.push(FindDatabaseResult {
                        id: id_str,
                        metadata: Some(metadata.into_iter()
                            .map(|(k, v)| {
                                let serialized_value = serde_json::to_value(v).expect("Failed to serialize value");
                                (k, serialized_value)
                            })
                            .collect()),
                        text: text,
                        distance: 1.0 - scored_point.score,
                    });
                }

                Ok(list)
            },
            Err(e) => {
                println!("Qdrant Error: {}", e);
                Err(false)
            }
        }
    }

	pub async fn find(data: web::Json<FindDatabaseReq>, redis_client: web::Data<redis::Client>) -> impl Responder {
        use futures::future::join_all;
        use redis::AsyncCommands;
        use md5;

        let model = data.model.clone().unwrap_or(DEFAULT_EMBEDDING_MODEL.to_string());
        let multiplexed_con = match redis_client.get_multiplexed_async_connection().await {
            Ok(con) => Some(con),
            Err(e) => {
                println!("Redis connection error: {}", e);
                None
            }
        };

        let embeddings: Option<Vec<Vec<f32>>> = match &data.texts {
            Some(texts) => {
                let tasks = texts.iter().map(|text| {
                    let text = text.clone();
                    let model = model.clone();
                    let translate_to = data.translate_to.clone();

                    let mut task_con = multiplexed_con.clone(); 

                    async move {
                        let key_str = format!("{}:{}", model, text);
                        let key_hash = format!("{:x}", md5::compute(key_str));
                        let redis_key = format!("emb:{}", key_hash);

                        if let Some(mut connection) = task_con.clone() {
                            let cached: redis::RedisResult<String> = connection.get(&redis_key).await;
                            if let Ok(json_str) = cached {
                                if let Ok(vec_data) = serde_json::from_str::<Vec<f32>>(&json_str) {
                                    return vec_data;
                                }
                            }
                        }
                        
                        let text_to_embed = if translate_to.is_some() && translate_to != Some(String::new()) {
                            let prompt = translate_prompt(text.clone(), translate_to.unwrap());
                            Ollama::generate(prompt, DEFAULT_PROMPT_MODEL.to_string()).await.unwrap_or(text)
                        } else {
                            text
                        };

                        let embedding = Ollama::embedding(text_to_embed, model).await.unwrap();
                        
                        if !embedding.is_empty() {
                            if let Some(con) = &mut task_con {
                                if let Ok(json_str) = serde_json::to_string(&embedding) {
                                    let _: redis::RedisResult<()> = con.set_ex(&redis_key, json_str, 86400).await;
                                }
                            }
                        }

                        embedding
                    }
                });

                Some(join_all(tasks).await)
            },
            None => None,
        };

        let collection_name: String = data.collection.clone();
        let nearests = Database::find_nearest(collection_name, embeddings, data.limit, data.offset, data.filters.clone(), data.filter_by_ids.clone()).await;
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
