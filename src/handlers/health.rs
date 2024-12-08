use chrono::Utc;
use serde::{Deserialize, Serialize};
use crate::models::general::*;
use crate::models::health::*;
use actix_web::*;
use actix_web::web;


#[derive(Clone, Serialize, Deserialize)]
pub struct GetHealthReq {
}

impl Health {
	pub async fn check(_query: web::Query<GetHealthReq>) -> impl Responder {
		let res = Health {
    		time: Utc::now().naive_utc(),
		};

		HttpResponse::Ok().json(GeneralValueResult{result: res, status: true})
	} 
}