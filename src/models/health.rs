#![allow(non_snake_case)]

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use std::*;

#[derive(Clone, Serialize, Deserialize)]
pub struct Health {
    pub time: NaiveDateTime,
}
