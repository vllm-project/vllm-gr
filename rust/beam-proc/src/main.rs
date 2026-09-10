mod catalog;
mod config;
mod logging;
mod runtime;
mod worker;

use catalog::Catalog;
use config::Config;
use crossbeam_channel::bounded;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{info, warn};
use worker::beam_worker::BeamWorker;
use worker::io_worker::IoWorker;

static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);
const DEFAULT_QUEUE_CAPACITY: usize = 64;
const DEFAULT_ZMQ_HWM: i32 = 64;

fn main() -> anyhow::Result<()> {
    install_signal_handlers();

    let config_path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("VLLM_GR_BEAM_PROC_CONFIG").ok())
        .unwrap_or_else(|| "beam-proc/config.json".to_string());
    let config = Config::load(config_path)?;
    let _log_guard = logging::init(&config.log_level, &config.log_file)?;
    info!(
        num_workers = config.num_workers,
        log_level = %config.log_level,
        log_file = %config.log_file,
        enginecore_beamproc_addr = %config.enginecore_beamproc_addr,
        enginecore_beamproc_tx_addr = ?config.enginecore_beamproc_tx_addr,
        enginecore_beamproc_rx_addr = ?config.enginecore_beamproc_rx_addr,
        apiserver_output_addr = %config.apiserver_output_addr,
        "beam-proc starting"
    );

    let catalog = load_catalog(&config)?;
    let queue_capacity = config.queue_capacity.unwrap_or(DEFAULT_QUEUE_CAPACITY);
    let zmq_hwm = config.zmq_hwm.unwrap_or(DEFAULT_ZMQ_HWM);
    anyhow::ensure!(
        queue_capacity > 0,
        "queue_capacity must be greater than zero"
    );
    anyhow::ensure!(zmq_hwm > 0, "zmq_hwm must be greater than zero");
    let mut workers = Vec::with_capacity(config.num_workers);
    let mut plan_rxs = Vec::with_capacity(config.num_workers);
    for worker_id in 0..config.num_workers {
        let (plan_tx, plan_rx) = bounded(queue_capacity);
        workers.push(BeamWorker::init(
            worker_id,
            plan_tx,
            catalog.clone(),
            queue_capacity,
        ));
        plan_rxs.push(plan_rx);
    }

    for worker in &mut workers {
        info!(worker_id = worker.worker_id(), "starting beam worker");
        worker.run()?;
    }

    let worker_queues = workers
        .iter()
        .map(BeamWorker::message_queue)
        .collect::<Vec<_>>();
    let mut io_worker = IoWorker::init(
        config.enginecore_beamproc_addr.clone(),
        config
            .enginecore_beamproc_tx_addr
            .clone()
            .unwrap_or_else(|| config.enginecore_beamproc_addr.clone()),
        config
            .enginecore_beamproc_rx_addr
            .clone()
            .unwrap_or_else(|| config.enginecore_beamproc_addr.clone()),
        config.apiserver_output_addr.clone(),
        worker_queues,
        plan_rxs,
        queue_capacity,
        zmq_hwm,
    )?;
    io_worker.run()?;
    io_worker.connect_zmq()?;

    if let Some(ready_path) = config.ready_path.as_deref() {
        std::fs::write(ready_path, b"READY\n")?;
    }
    info!("beam-proc ready");
    let runtime_result = wait_for_shutdown(&io_worker, &workers);
    info!("beam-proc shutdown requested");

    io_worker.shutdown();
    let io_join_result = io_worker
        .join()
        .map_err(|_| anyhow::anyhow!("I/O worker thread panicked"));

    let mut shutdown_error = None;
    for worker in &workers {
        if let Err(err) = worker.shutdown() {
            warn!(
                worker_id = worker.worker_id(),
                error = %err,
                "failed to signal beam worker shutdown"
            );
            shutdown_error.get_or_insert_with(|| {
                anyhow::anyhow!(
                    "failed to signal beam worker {} shutdown: {err}",
                    worker.worker_id()
                )
            });
        }
    }

    let mut worker_join_error = None;
    for worker in &mut workers {
        let worker_id = worker.worker_id();
        if worker.join().is_err() {
            worker_join_error
                .get_or_insert_with(|| anyhow::anyhow!("beam worker {worker_id} thread panicked"));
        }
    }

    runtime_result?;
    io_join_result?;
    if let Some(err) = shutdown_error {
        return Err(err);
    }
    if let Some(err) = worker_join_error {
        return Err(err);
    }

    info!("beam-proc stopped");
    Ok(())
}

fn load_catalog(config: &Config) -> anyhow::Result<Option<Arc<Catalog>>> {
    let Some(path) = config.catalog_token_ids_path.as_deref() else {
        info!("beam-proc catalog not configured");
        return Ok(None);
    };
    if config.catalog_token_ids_format.as_deref().unwrap_or("json") != "json" {
        anyhow::bail!(
            "unsupported catalog_token_ids_format={:?}",
            config.catalog_token_ids_format
        );
    }
    let catalog = Catalog::load_json(path)?;
    info!(
        catalog_token_ids_path = %path,
        catalog_nodes = catalog.node_count(),
        "beam-proc loaded token-id catalog"
    );
    Ok(Some(Arc::new(catalog)))
}

fn install_signal_handlers() {
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_shutdown_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            handle_shutdown_signal as *const () as libc::sighandler_t,
        );
    }
}

extern "C" fn handle_shutdown_signal(_signal: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
}

fn wait_for_shutdown(io_worker: &IoWorker, workers: &[BeamWorker]) -> anyhow::Result<()> {
    while !SHUTDOWN_REQUESTED.load(Ordering::SeqCst) {
        if io_worker.is_finished() {
            anyhow::bail!("I/O worker exited unexpectedly");
        }
        if let Some(worker) = workers.iter().find(|worker| worker.is_finished()) {
            anyhow::bail!("beam worker {} exited unexpectedly", worker.worker_id());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Ok(())
}
