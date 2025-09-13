use ollama_rs::generation::embeddings::request::{EmbeddingsInput, GenerateEmbeddingsRequest};
use ollama_rs::generation::completion::request::GenerationRequest;


pub struct Ollama {
}

impl Ollama {
    pub async fn generate(prompt: String, model: String) -> Result<String, bool> {
        use ollama_rs::Ollama;

        let ollama = Ollama::default();

        let res = ollama.generate(GenerationRequest::new(model, prompt)).await;
        match res {
            Ok(r) => Ok(r.response),
            Err(_) => Err(false),
        }
    }

    pub async fn embedding(prompt: String, model: String) -> Result<Vec<f32>, bool> {
        use ollama_rs::Ollama;

        let ollama = Ollama::default();

        let input: EmbeddingsInput = EmbeddingsInput::Single(prompt.clone());

        let request = GenerateEmbeddingsRequest::new(model.clone(), input);
        let res = ollama.generate_embeddings(request).await;

        match res {
            Ok(r) => {
                match r.embeddings.get(0) {
                    Some(v) => Ok(v.clone()),
                    None => Err(false)
                }
            }
            Err(err) => {
                println!("prompt: {}, model: {}, err: {}", prompt, model, err);
                Err(false)
            },
        }
    }
}
 
