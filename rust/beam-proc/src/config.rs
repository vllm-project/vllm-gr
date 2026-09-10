use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub num_workers: usize,
    pub log_level: String,
    pub log_file: String,
    pub enginecore_beamproc_addr: String,
    pub enginecore_beamproc_tx_addr: Option<String>,
    pub enginecore_beamproc_rx_addr: Option<String>,
    pub apiserver_output_addr: String,
    pub ready_path: Option<String>,
    pub queue_capacity: Option<usize>,
    pub zmq_hwm: Option<i32>,
    pub catalog_token_ids_path: Option<String>,
    pub catalog_token_ids_format: Option<String>,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&data)?;
        Ok(config)
    }
}
