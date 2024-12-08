use actix_cors::Cors;
use std::env;
use crate::models::health;
use crate::models::database;
use dotenvy::dotenv;
use actix_web::{web, App, HttpServer};

mod models;
mod handlers;
mod utils;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();

    let listen_url: String = env::var("LISTEN_ADDRESS").unwrap_or("127.0.0.1".to_string());
    
    HttpServer::new(|| {
        #[allow(deprecated)]
        App::new()
            .wrap(Cors::permissive())
            
            .route("/api/v1/health", web::get().to(health::Health::check))

            .route("/api/v1/database", web::post().to(database::Database::insert))
            .route("/api/v1/database/find", web::post().to(database::Database::find))
    })
    .workers(2)
    .bind(listen_url)?
    .run()
    .await
}
