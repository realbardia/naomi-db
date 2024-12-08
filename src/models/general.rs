use serde::Serialize;

#[derive(Serialize)]
pub struct GeneralValueResult<T> {
    pub result: T,
    pub status: bool,
}

#[derive(Serialize)]
pub struct ErrorResult {
    pub status: bool,
    pub message: Option<String>,
}
