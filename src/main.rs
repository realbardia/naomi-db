use actix_cors::Cors;
use models::database::Database;
use std::env;
use crate::models::health;
use crate::models::database;
use dotenvy::dotenv;
use actix_web::{web, App, HttpServer};

use std::process::{Command, Stdio, Child};
use std::thread;
use std::sync::{Arc, Mutex};

mod models;
mod handlers;
mod utils;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();

    let child_process: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
    let child_process_clone = Arc::clone(&child_process);

    let listen_url: String = env::var("LISTEN_ADDRESS").unwrap_or("127.0.0.1".to_string());

    thread::spawn(move || loop {
        let mut child = child_process_clone.lock().unwrap();

        if child.is_none() || child.as_mut().unwrap().try_wait().ok().flatten().is_some() {
            println!("Starting chroma process...");
            *child = Some(
                Command::new("chroma")
                    .arg("run")
                    .arg("--port")
                    .arg(Database::get_chroma_port().to_string())
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .expect("Failed to spawn child process"),
            );
        }

        drop(child);
        thread::sleep(std::time::Duration::from_secs(1));
    });
    
    HttpServer::new(|| {
        #[allow(deprecated)]
        App::new()
            .wrap(Cors::permissive())
            
            .route("/api/v1/health", web::get().to(health::Health::check))

            .route("/api/v1/database", web::post().to(database::Database::insert))
            .route("/api/v1/database/embeddings", web::post().to(database::Database::insert_embeddings))
            .route("/api/v1/database/find", web::post().to(database::Database::find))
    })
    .workers(2)
    .bind(listen_url)?
    .run()
    .await
}
