use std::path::Path;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

pub fn init(log_level: &str, log_file: &str) -> anyhow::Result<WorkerGuard> {
    let log_path = Path::new(log_file);
    let log_dir = log_path.parent().unwrap_or_else(|| Path::new("."));
    let log_name = log_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("beam-proc.log");

    std::fs::create_dir_all(log_dir)?;

    let file_appender = tracing_appender::rolling::never(log_dir, log_name);
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    let filter = EnvFilter::try_new(log_level).unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(non_blocking).with_ansi(false))
        .init();

    Ok(guard)
}
