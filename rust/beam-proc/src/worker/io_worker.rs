use crate::runtime::manager::{Manager, ManagerEvent, ManagerMessage};
use crate::worker::beam_worker::{
    BeamContext, BeamEngineOutput, BeamEngineOutputBatch, BeamWorkerMsg, BeamWorkerPlan,
};
use crossbeam_channel::{Receiver, Select, Sender, TryRecvError, bounded};
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ptr;
use std::sync::OnceLock;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;
use tracing::{debug, info, warn};

const MSG_ENGINE_OUTPUT_BATCH: &str = "ENGINE_OUTPUT_BATCH";
const MSG_BEAM_PLAN: &str = "BEAM_PLAN";
const MSG_BEAM_FINISHED: &str = "BEAM_FINISHED";
const MSG_BEAM_FINAL: &str = "BEAM_FINAL";
const MSG_BEAM_STATE_UPDATE: &str = "BEAM_STATE_UPDATE";
const MSG_ABORT: &str = "ABORT";
const RX_ZMQ_POLL_TIMEOUT_MS: i64 = 10;
const ZMQ_BUFFER_BYTES: i32 = 512 * 1024 * 1024;
const BEAMPROC_ENABLED_ENV: &str = "VLLM_GR_BEAM_PROC_ENABLED";
const BEAMPROC_OUTPUT_POOL_SLOTS_ENV: &str = "VLLM_GR_BEAMPROC_OUTPUT_POOL_SLOTS";
const BEAMPROC_OUTPUT_POOL_CAPACITY_ENV: &str = "VLLM_GR_BEAMPROC_OUTPUT_POOL_CAPACITY";
const DEFAULT_OUTPUT_POOL_SLOTS: usize = 256;
const DEFAULT_OUTPUT_POOL_CAPACITY: usize = 32;
const TX_PLAN_RECV_TIMEOUT_MS: u64 = 10;
const TX_SEND_TIMEOUT_MS: i32 = 5_000;
const IO_READY_TIMEOUT: Duration = Duration::from_secs(10);
const ABORT_ACK_TIMEOUT: Duration = Duration::from_secs(5);
type ReadyAck = Sender<Result<(), String>>;

#[derive(Debug)]
enum IoRxMsg {
    ConnectZmq(ReadyAck),
    Shutdown,
}

impl ManagerMessage for IoRxMsg {
    fn shutdown() -> Self {
        Self::Shutdown
    }

    fn is_shutdown(&self) -> bool {
        matches!(self, Self::Shutdown)
    }
}

#[derive(Debug)]
enum IoTxMsg {
    ConnectZmq(ReadyAck),
    Shutdown,
}

impl ManagerMessage for IoTxMsg {
    fn shutdown() -> Self {
        Self::Shutdown
    }

    fn is_shutdown(&self) -> bool {
        matches!(self, Self::Shutdown)
    }
}

pub struct IoWorker {
    rx_manager: Manager<IoRxMsg>,
    tx_manager: Manager<IoTxMsg>,
}

impl IoWorker {
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        enginecore_beamproc_addr: String,
        enginecore_beamproc_tx_addr: String,
        enginecore_beamproc_rx_addr: String,
        apiserver_output_addr: String,
        worker_queues: Vec<Sender<BeamWorkerMsg>>,
        plan_rxs: Vec<Receiver<BeamWorkerPlan>>,
        queue_capacity: usize,
        zmq_hwm: i32,
    ) -> anyhow::Result<Self> {
        let mut rx_state = IoRxState::new(
            enginecore_beamproc_addr,
            enginecore_beamproc_tx_addr,
            worker_queues,
            queue_capacity,
            zmq_hwm,
        );
        let mut tx_state = IoTxState::new(
            enginecore_beamproc_rx_addr,
            apiserver_output_addr,
            plan_rxs,
            zmq_hwm,
        );
        let rx_manager = Manager::init_nonblocking_with_capacity(
            move |event| match event {
                ManagerEvent::Message(msg) => rx_state.handle(msg),
                ManagerEvent::Idle => rx_state.poll_zmq(),
            },
            queue_capacity,
        );
        let tx_manager = Manager::init_nonblocking_with_capacity(
            move |event| match event {
                ManagerEvent::Message(msg) => tx_state.handle(msg),
                ManagerEvent::Idle => tx_state.poll_zmq(),
            },
            queue_capacity,
        );

        Ok(Self {
            rx_manager,
            tx_manager,
        })
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        self.rx_manager.run()?;
        if let Err(err) = self.tx_manager.run() {
            let _ = self.rx_manager.shutdown();
            return Err(err);
        }
        Ok(())
    }

    pub fn connect_zmq(&self) -> anyhow::Result<()> {
        let (rx_ready_tx, rx_ready_rx) = bounded(1);
        let (tx_ready_tx, tx_ready_rx) = bounded(1);
        self.rx_manager
            .message_queue()
            .send(IoRxMsg::ConnectZmq(rx_ready_tx))
            .map_err(|err| anyhow::anyhow!("failed to signal RX I/O worker startup: {err}"))?;
        self.tx_manager
            .message_queue()
            .send(IoTxMsg::ConnectZmq(tx_ready_tx))
            .map_err(|err| anyhow::anyhow!("failed to signal TX I/O worker startup: {err}"))?;

        rx_ready_rx
            .recv_timeout(IO_READY_TIMEOUT)
            .map_err(|err| anyhow::anyhow!("RX I/O worker did not acknowledge READY: {err}"))?
            .map_err(|err| anyhow::anyhow!("RX I/O worker initialization failed: {err}"))?;
        tx_ready_rx
            .recv_timeout(IO_READY_TIMEOUT)
            .map_err(|err| anyhow::anyhow!("TX I/O worker did not acknowledge READY: {err}"))?
            .map_err(|err| anyhow::anyhow!("TX I/O worker initialization failed: {err}"))?;
        Ok(())
    }

    pub fn is_finished(&self) -> bool {
        self.rx_manager.is_finished() || self.tx_manager.is_finished()
    }

    pub fn shutdown(&self) {
        if let Err(err) = self.rx_manager.shutdown() {
            warn!(error = %err, "failed to signal RX I/O worker shutdown");
        }
        if let Err(err) = self.tx_manager.shutdown() {
            warn!(error = %err, "failed to signal TX I/O worker shutdown");
        }
    }

    pub fn join(&mut self) -> thread::Result<()> {
        self.rx_manager.join()?;
        self.tx_manager.join()
    }
}

struct IoRxState {
    #[cfg(feature = "iceoryx")]
    iceoryx_rx: Option<ce_transport::iceoryx::FrameReceiver>,
    enginecore_beamproc_addr: String,
    enginecore_beamproc_tx_addr: String,
    zmq_context: Option<zmq::Context>,
    enginecore_rx_socket: Option<zmq::Socket>,
    monitor_threads: Vec<JoinHandle<()>>,
    worker_queues: Vec<Sender<BeamWorkerMsg>>,
    output_pools: Vec<BeamOutputPool>,
    zmq_hwm: i32,
}

struct IoTxState {
    enginecore_beamproc_rx_addr: String,
    apiserver_output_addr: String,
    zmq_context: Option<zmq::Context>,
    enginecore_tx_socket: Option<CeSender>,
    apiserver_output_socket: Option<zmq::Socket>,
    monitor_threads: Vec<JoinHandle<()>>,
    plan_rxs: Vec<Receiver<BeamWorkerPlan>>,
    zmq_hwm: i32,
}

enum CeSender {
    Zmq(zmq::Socket),
    #[cfg(feature = "iceoryx")]
    Iceoryx(Box<ce_transport::iceoryx::FrameSender>),
}

impl CeSender {
    fn send_multipart<const N: usize>(&self, frames: [&[u8]; N], flags: i32) -> Result<(), String> {
        match self {
            Self::Zmq(socket) => socket
                .send_multipart(frames, flags)
                .map_err(|e| e.to_string()),
            #[cfg(feature = "iceoryx")]
            Self::Iceoryx(sender) => sender.send(
                &frames
                    .iter()
                    .map(|frame| frame.to_vec())
                    .collect::<Vec<_>>(),
                Duration::from_millis(TX_SEND_TIMEOUT_MS as u64),
                || crate::SHUTDOWN_REQUESTED.load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

struct BeamOutputPool {
    free_tx: Sender<BeamEngineOutputBatch>,
    free_rx: Receiver<BeamEngineOutputBatch>,
}

impl BeamOutputPool {
    fn new(slots: usize, output_capacity: usize) -> Self {
        let (free_tx, free_rx) = bounded(slots);
        for _ in 0..slots {
            if free_tx
                .send(BeamEngineOutputBatch::with_capacity(output_capacity))
                .is_err()
            {
                break;
            }
        }
        Self { free_tx, free_rx }
    }

    fn try_claim(&self) -> Option<BeamEngineOutputBatch> {
        match self.free_rx.try_recv() {
            Ok(buffer) => Some(buffer),
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
        }
    }

    fn return_tx(&self) -> Sender<BeamEngineOutputBatch> {
        self.free_tx.clone()
    }
}

fn configure_enginecore_push_socket(socket: &zmq::Socket, hwm: i32) -> anyhow::Result<()> {
    socket
        .set_sndhwm(hwm)
        .map_err(|err| anyhow::anyhow!("failed to set EngineCore PUSH ZMQ SNDHWM: {err}"))?;
    socket
        .set_sndbuf(ZMQ_BUFFER_BYTES)
        .map_err(|err| anyhow::anyhow!("failed to set EngineCore PUSH ZMQ SNDBUF: {err}"))?;
    socket
        .set_sndtimeo(TX_SEND_TIMEOUT_MS)
        .map_err(|err| anyhow::anyhow!("failed to set EngineCore PUSH ZMQ SNDTIMEO: {err}"))?;
    Ok(())
}

fn configure_enginecore_pull_socket(socket: &zmq::Socket, hwm: i32) -> anyhow::Result<()> {
    socket
        .set_rcvhwm(hwm)
        .map_err(|err| anyhow::anyhow!("failed to set EngineCore PULL ZMQ RCVHWM: {err}"))?;
    socket
        .set_rcvbuf(ZMQ_BUFFER_BYTES)
        .map_err(|err| anyhow::anyhow!("failed to set EngineCore PULL ZMQ RCVBUF: {err}"))?;
    Ok(())
}

fn configure_apiserver_output_socket(socket: &zmq::Socket, hwm: i32) -> anyhow::Result<()> {
    socket
        .set_sndhwm(hwm)
        .map_err(|err| anyhow::anyhow!("failed to set APIServer output PUSH ZMQ SNDHWM: {err}"))?;
    socket
        .set_sndbuf(ZMQ_BUFFER_BYTES)
        .map_err(|err| anyhow::anyhow!("failed to set APIServer output PUSH ZMQ SNDBUF: {err}"))?;
    socket.set_sndtimeo(TX_SEND_TIMEOUT_MS).map_err(|err| {
        anyhow::anyhow!("failed to set APIServer output PUSH ZMQ SNDTIMEO: {err}")
    })?;
    Ok(())
}

impl IoRxState {
    fn new(
        enginecore_beamproc_addr: String,
        enginecore_beamproc_tx_addr: String,
        worker_queues: Vec<Sender<BeamWorkerMsg>>,
        queue_capacity: usize,
        zmq_hwm: i32,
    ) -> Self {
        let output_pool_slots = beamproc_output_pool_slots().min(queue_capacity.max(1));
        let output_pool_capacity = beamproc_output_pool_capacity();
        let output_pools = (0..worker_queues.len())
            .map(|_| BeamOutputPool::new(output_pool_slots, output_pool_capacity))
            .collect();
        Self {
            enginecore_beamproc_addr,
            enginecore_beamproc_tx_addr,
            #[cfg(feature = "iceoryx")]
            iceoryx_rx: None,
            zmq_context: None,
            enginecore_rx_socket: None,
            monitor_threads: Vec::new(),
            worker_queues,
            output_pools,
            zmq_hwm,
        }
    }

    fn handle(&mut self, msg: IoRxMsg) {
        match msg {
            IoRxMsg::ConnectZmq(ready) => {
                let result = self.ensure_connected().map_err(|err| err.to_string());
                let _ = ready.send(result);
            }
            IoRxMsg::Shutdown => {}
        }
    }

    fn ensure_connected(&mut self) -> anyhow::Result<()> {
        if ce_transport::ce_bp_iceoryx_enabled().map_err(anyhow::Error::msg)? {
            #[cfg(feature = "iceoryx")]
            if self.iceoryx_rx.is_none() {
                self.iceoryx_rx = Some(
                    ce_transport::iceoryx::FrameReceiver::new(
                        &self.enginecore_beamproc_tx_addr,
                        self.zmq_hwm as usize,
                    )
                    .map_err(anyhow::Error::msg)?,
                );
                warn!("CE-to-BP RX initialized iceoryx2 shared-memory transport");
            }
            return Ok(());
        }
        if self.zmq_context.is_some() {
            return Ok(());
        }

        let zmq_context = zmq::Context::new();
        let enginecore_rx_socket = zmq_context
            .socket(zmq::PULL)
            .map_err(|err| anyhow::anyhow!("failed to create EngineCore RX PULL socket: {err}"))?;
        enginecore_rx_socket
            .set_linger(0)
            .map_err(|err| anyhow::anyhow!("failed to set EngineCore RX PULL linger: {err}"))?;
        configure_enginecore_pull_socket(&enginecore_rx_socket, self.zmq_hwm)?;
        enginecore_rx_socket
            .bind(&self.enginecore_beamproc_tx_addr)
            .map_err(|err| {
                anyhow::anyhow!(
                    "failed to bind EngineCore RX PULL socket at {}: {err}",
                    self.enginecore_beamproc_tx_addr
                )
            })?;

        if let Some(handle) =
            start_zmq_monitor(&zmq_context, &enginecore_rx_socket, "enginecore-rx")
        {
            self.monitor_threads.push(handle);
        }

        warn!(
            enginecore_beamproc_addr = %self.enginecore_beamproc_addr,
            enginecore_beamproc_tx_addr = %self.enginecore_beamproc_tx_addr,
            worker_queues = self.worker_queues.len(),
            "RX I/O worker initialized ZMQ socket"
        );

        self.enginecore_rx_socket = Some(enginecore_rx_socket);
        self.zmq_context = Some(zmq_context);
        Ok(())
    }

    fn poll_zmq(&mut self) {
        #[cfg(feature = "iceoryx")]
        if let Some(receiver) = self.iceoryx_rx.as_ref() {
            match receiver.receive_wait(ce_transport::iceoryx::RECEIVE_WAIT) {
                Ok(Some(frames)) => self.handle_enginecore_frames(frames),
                Ok(None) => {}
                // A dead RX worker is detected by the existing process supervisor.
                Err(error) => panic!("CE-to-BP iceoryx RX failed: {error}"),
            }
            return;
        }
        let Some(rx_readable) = self.poll_enginecore_socket() else {
            thread::yield_now();
            return;
        };

        if rx_readable {
            self.drain_enginecore_messages();
        }
    }

    fn poll_enginecore_socket(&self) -> Option<bool> {
        let enginecore_rx_socket = self.enginecore_rx_socket.as_ref()?;
        let mut poll_items = [enginecore_rx_socket.as_poll_item(zmq::POLLIN)];

        match zmq::poll(&mut poll_items, RX_ZMQ_POLL_TIMEOUT_MS) {
            Ok(0) => Some(false),
            Ok(_) => Some(poll_items[0].is_readable()),
            Err(err) => {
                warn!(error = %err, "RX I/O worker ZMQ poll failed");
                Some(false)
            }
        }
    }

    fn drain_enginecore_messages(&mut self) {
        let frames = {
            let Some(socket) = self.enginecore_rx_socket.as_ref() else {
                return;
            };
            match socket.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => frames,
                Err(zmq::Error::EAGAIN) => return,
                Err(err) => {
                    warn!(
                        peer = "enginecore",
                        error = %err,
                        "I/O worker failed to receive ZMQ message"
                    );
                    return;
                }
            }
        };

        self.handle_enginecore_frames(frames);
    }

    fn handle_enginecore_frames(&mut self, frames: Vec<Vec<u8>>) {
        let message_type = frames
            .first()
            .and_then(|frame| std::str::from_utf8(frame).ok())
            .unwrap_or("<missing>");
        let payload_bytes = frames.get(1).map(Vec::len).unwrap_or(0);
        debug!(
            peer = "enginecore",
            message_type,
            frames = frames.len(),
            payload_bytes,
            frame_bytes = ?frames.iter().map(Vec::len).collect::<Vec<_>>(),
            "I/O worker received ZMQ message"
        );

        if message_type == MSG_ENGINE_OUTPUT_BATCH {
            let Some(payload) = frames.get(1) else {
                warn!("I/O worker received ENGINE_OUTPUT_BATCH without payload");
                return;
            };
            self.handle_engine_output_batch(
                payload,
                frames.get(2).map(Vec::as_slice).unwrap_or(&[]),
                frames.get(3).map(Vec::as_slice).unwrap_or(&[]),
                frames.get(4).map(Vec::as_slice).unwrap_or(&[]),
                frames.get(5).map(Vec::as_slice).unwrap_or(&[]),
            );
        } else if message_type == MSG_BEAM_STATE_UPDATE {
            let Some(payload) = frames.get(1) else {
                warn!("I/O worker received BEAM_STATE_UPDATE without payload");
                return;
            };
            self.handle_beam_state_update(payload);
        } else if message_type == MSG_ABORT {
            let Some(payload) = frames.get(1) else {
                warn!("I/O worker received ABORT without payload");
                return;
            };
            self.handle_abort(payload);
        }
    }

    fn handle_engine_output_batch(
        &mut self,
        payload: &[u8],
        token_id_bytes: &[u8],
        logprob_bytes: &[u8],
        rank_bytes: &[u8],
        new_token_id_bytes: &[u8],
    ) {
        match self.try_handle_engine_output_batch_v3(
            payload,
            token_id_bytes,
            logprob_bytes,
            rank_bytes,
            new_token_id_bytes,
        ) {
            Ok(true) => return,
            Ok(false) => {}
            Err(_) => {
                warn!(
                    payload_bytes = payload.len(),
                    "I/O worker rejected malformed ENGINE_OUTPUT_BATCH"
                );
                return;
            }
        }

        let batch = match rmp_serde::from_slice::<EngineOutputBatchPayload>(payload) {
            Ok(batch) => batch,
            Err(err) => {
                warn!(
                    error = %err,
                    payload_bytes = payload.len(),
                    "I/O worker failed to decode ENGINE_OUTPUT_BATCH"
                );
                return;
            }
        };

        if batch.version != 1 && batch.version != 2 {
            warn!(
                version = batch.version,
                "I/O worker received unsupported ENGINE_OUTPUT_BATCH version"
            );
            return;
        }

        if !validate_binary_frame::<i32>(token_id_bytes, "candidate_token_ids")
            || !validate_binary_frame::<f32>(logprob_bytes, "candidate_logprobs")
            || !validate_binary_frame::<i32>(rank_bytes, "sampled_token_ranks")
        {
            return;
        }

        if batch.outputs.iter().any(|output| {
            output
                .validate_binary_ranges(token_id_bytes, logprob_bytes, rank_bytes)
                .is_err()
        }) {
            warn!("I/O worker rejected malformed ENGINE_OUTPUT_BATCH binary ranges");
            return;
        }

        let mut by_request: HashMap<String, EngineOutputRequestGroup> = HashMap::new();
        for output in batch.outputs {
            let (request_id, correlation_id) =
                output.route_ids(batch.correlation_base, &batch.contexts);
            if !by_request.contains_key(&request_id) {
                let group = self.claim_engine_output_group(&request_id, correlation_id);
                by_request.insert(request_id.clone(), group);
            }
            if let Some(group) = by_request.get_mut(&request_id)
                && output
                    .append_routed_output(
                        batch.correlation_base,
                        &batch.contexts,
                        token_id_bytes,
                        logprob_bytes,
                        rank_bytes,
                        &mut group.batch,
                    )
                    .is_err()
            {
                warn!("I/O worker rejected malformed ENGINE_OUTPUT_BATCH binary ranges");
                return;
            }
        }

        let request_count = by_request.len();
        let output_count = by_request
            .values()
            .map(|group| group.batch.output_count())
            .sum::<usize>();
        for (request_id, group) in by_request {
            self.route_engine_output_batch(
                batch.engine_index,
                batch.client_index,
                request_id,
                group,
            );
        }
        debug!(
            engine_index = batch.engine_index,
            client_index = batch.client_index,
            requests = request_count,
            outputs = output_count,
            "I/O worker decoded ENGINE_OUTPUT_BATCH"
        );
    }

    fn try_handle_engine_output_batch_v3(
        &mut self,
        payload: &[u8],
        token_id_bytes: &[u8],
        logprob_bytes: &[u8],
        rank_bytes: &[u8],
        new_token_id_bytes: &[u8],
    ) -> Result<bool, FastDecodeError> {
        let mut decoder = MsgpackDecoder::new(payload);
        let version = decoder.engine_output_batch_version()?;
        if version != Some(6) {
            return Ok(false);
        }

        if !validate_binary_frame::<i32>(token_id_bytes, "candidate_token_ids")
            || !validate_binary_frame::<f32>(logprob_bytes, "candidate_logprobs")
            || !validate_binary_frame::<i32>(rank_bytes, "sampled_token_ranks")
            || (version == Some(6)
                && !validate_binary_frame::<i32>(new_token_id_bytes, "new_token_ids"))
        {
            return Ok(true);
        }

        let mut decoder = MsgpackDecoder::new(payload);
        let decoded = decoder.decode_v3_engine_output_batch(
            self,
            token_id_bytes,
            logprob_bytes,
            rank_bytes,
            new_token_id_bytes,
        )?;

        let request_count = decoded.by_request.len();
        let output_count = decoded
            .by_request
            .values()
            .map(|group| group.batch.output_count())
            .sum::<usize>();
        for (request_id, group) in decoded.by_request {
            self.route_engine_output_batch(
                decoded.engine_index,
                decoded.client_index,
                request_id,
                group,
            );
        }
        debug!(
            engine_index = decoded.engine_index,
            client_index = decoded.client_index,
            requests = request_count,
            outputs = output_count,
            "I/O worker fast-decoded ENGINE_OUTPUT_BATCH"
        );
        Ok(true)
    }

    fn handle_beam_state_update(&mut self, payload: &[u8]) {
        let update = match rmp_serde::from_slice::<BeamStateUpdatePayload>(payload) {
            Ok(update) => update,
            Err(err) => {
                warn!(
                    error = %err,
                    payload_bytes = payload.len(),
                    "I/O worker failed to decode BEAM_STATE_UPDATE"
                );
                return;
            }
        };

        if update.version != 1 {
            warn!(
                version = update.version,
                "I/O worker received unsupported BEAM_STATE_UPDATE version"
            );
            return;
        }

        let root_request_id = update.root_request_id.to_string_value();
        let worker_index = self.worker_index_for_request(&root_request_id);
        let active_count = update.active_parent_ids.len();
        let send_result = self.worker_queues[worker_index].send(BeamWorkerMsg::BeamStateUpdate {
            root_request_id: root_request_id.clone(),
            beam_width: update.beam_width,
            active_parent_ids: update
                .active_parent_ids
                .into_iter()
                .map(ScalarValue::into_string)
                .collect(),
            active_parent_cum_logprobs: update.active_parent_cum_logprobs,
            active_parent_valid_token_ids: update
                .active_parent_valid_token_ids
                .into_iter()
                .map(|tokens| {
                    tokens
                        .into_iter()
                        .filter_map(ScalarValue::into_i32)
                        .collect::<Vec<_>>()
                })
                .collect(),
        });

        match send_result {
            Ok(()) => debug!(
                root_request_id = %root_request_id,
                worker_index,
                beam_width = update.beam_width,
                active = active_count,
                "I/O worker routed BEAM_STATE_UPDATE"
            ),
            Err(err) => warn!(
                root_request_id = %root_request_id,
                worker_index,
                error = %err,
                "I/O worker failed to route BEAM_STATE_UPDATE"
            ),
        }
    }

    fn handle_abort(&mut self, payload: &[u8]) {
        let abort = match rmp_serde::from_slice::<BeamAbortPayload>(payload) {
            Ok(abort) => abort,
            Err(err) => {
                warn!(
                    error = %err,
                    payload_bytes = payload.len(),
                    "I/O worker failed to decode ABORT"
                );
                return;
            }
        };

        for root_request_id in abort.root_request_ids {
            let root_request_id = root_request_id.into_string();
            let worker_index = self.worker_index_for_request(&root_request_id);
            let (ack_tx, ack_rx) = bounded(1);
            match self.worker_queues[worker_index].send(BeamWorkerMsg::Abort {
                root_request_id: root_request_id.clone(),
                ack: Some(ack_tx),
            }) {
                Ok(()) => match ack_rx.recv_timeout(ABORT_ACK_TIMEOUT) {
                    Ok(()) => debug!(
                        root_request_id = %root_request_id,
                        worker_index,
                        "I/O worker received terminal ABORT acknowledgement"
                    ),
                    Err(err) => warn!(
                        root_request_id = %root_request_id,
                        worker_index,
                        error = %err,
                        "I/O worker timed out waiting for terminal ABORT acknowledgement"
                    ),
                },
                Err(err) => warn!(
                    root_request_id = %root_request_id,
                    worker_index,
                    error = %err,
                    "I/O worker failed to route ABORT"
                ),
            }
        }
    }

    fn route_engine_output_batch(
        &mut self,
        engine_index: usize,
        client_index: usize,
        request_id: String,
        group: EngineOutputRequestGroup,
    ) {
        if self.worker_queues.is_empty() {
            warn!(
                request_id = %request_id,
                correlation_id = %group.correlation_id,
                "I/O worker has no beam workers to route engine output batch"
            );
            return;
        }

        let EngineOutputRequestGroup {
            correlation_id,
            worker_index,
            batch,
            pool_return_tx,
        } = group;
        let output_count = batch.output_count();
        let (message, used_pool) = if let Some(pool_return_tx) = pool_return_tx {
            (
                BeamWorkerMsg::EngineOutputBatchPooled {
                    engine_index,
                    client_index,
                    request_id,
                    correlation_id,
                    batch,
                    pool_return_tx,
                },
                true,
            )
        } else {
            (
                BeamWorkerMsg::EngineOutputBatch {
                    engine_index,
                    client_index,
                    request_id,
                    correlation_id,
                    batch,
                },
                false,
            )
        };

        match self.worker_queues[worker_index].send(message) {
            Ok(()) => debug!(
                engine_index,
                client_index,
                worker_index,
                outputs = output_count,
                used_pool,
                "I/O worker routed engine output batch"
            ),
            Err(err) => {
                let (request_id, correlation_id) = match &err.0 {
                    BeamWorkerMsg::EngineOutputBatch {
                        request_id,
                        correlation_id,
                        ..
                    }
                    | BeamWorkerMsg::EngineOutputBatchPooled {
                        request_id,
                        correlation_id,
                        ..
                    } => (request_id, correlation_id),
                    _ => unreachable!("engine output routing created an engine output message"),
                };
                warn!(
                    engine_index,
                    client_index,
                    request_id = %request_id,
                    correlation_id = %correlation_id,
                    worker_index,
                    outputs = output_count,
                    used_pool,
                    error = %err,
                    "I/O worker failed to route engine output batch"
                );
            }
        }
    }

    fn claim_engine_output_group(
        &mut self,
        request_id: &str,
        correlation_id: String,
    ) -> EngineOutputRequestGroup {
        let worker_index = self.worker_index_for_request(request_id);
        let pool = &self.output_pools[worker_index];
        if let Some(mut batch) = pool.try_claim() {
            batch.clear();
            EngineOutputRequestGroup {
                correlation_id,
                worker_index,
                batch,
                pool_return_tx: Some(pool.return_tx()),
            }
        } else {
            EngineOutputRequestGroup {
                correlation_id,
                worker_index,
                batch: BeamEngineOutputBatch::with_capacity(0),
                pool_return_tx: None,
            }
        }
    }

    fn worker_index_for_request(&self, root_request_id: &str) -> usize {
        hash_worker_index(root_request_id, self.worker_queues.len())
    }
}

impl IoTxState {
    fn new(
        enginecore_beamproc_rx_addr: String,
        apiserver_output_addr: String,
        plan_rxs: Vec<Receiver<BeamWorkerPlan>>,
        zmq_hwm: i32,
    ) -> Self {
        Self {
            enginecore_beamproc_rx_addr,
            apiserver_output_addr,
            zmq_context: None,
            enginecore_tx_socket: None,
            apiserver_output_socket: None,
            monitor_threads: Vec::new(),
            plan_rxs,
            zmq_hwm,
        }
    }

    fn handle(&mut self, msg: IoTxMsg) {
        match msg {
            IoTxMsg::ConnectZmq(ready) => {
                let result = self.ensure_connected().map_err(|err| err.to_string());
                let _ = ready.send(result);
            }
            IoTxMsg::Shutdown => {}
        }
    }

    fn ensure_connected(&mut self) -> anyhow::Result<()> {
        if self.zmq_context.is_some() {
            return Ok(());
        }

        let zmq_context = zmq::Context::new();
        let use_iceoryx = ce_transport::ce_bp_iceoryx_enabled().map_err(anyhow::Error::msg)?;
        let apiserver_output_socket = zmq_context.socket(zmq::PUSH).map_err(|err| {
            anyhow::anyhow!("failed to create APIServer output PUSH socket: {err}")
        })?;
        apiserver_output_socket
            .set_linger(0)
            .map_err(|err| anyhow::anyhow!("failed to set APIServer output PUSH linger: {err}"))?;
        configure_apiserver_output_socket(&apiserver_output_socket, self.zmq_hwm)?;

        apiserver_output_socket
            .connect(&self.apiserver_output_addr)
            .map_err(|err| {
                anyhow::anyhow!(
                    "failed to connect APIServer output PUSH socket at {}: {err}",
                    self.apiserver_output_addr
                )
            })?;

        let sender = if use_iceoryx {
            #[cfg(feature = "iceoryx")]
            {
                CeSender::Iceoryx(Box::new(
                    ce_transport::iceoryx::FrameSender::new(
                        &self.enginecore_beamproc_rx_addr,
                        self.zmq_hwm as usize,
                    )
                    .map_err(anyhow::Error::msg)?,
                ))
            }
            #[cfg(not(feature = "iceoryx"))]
            anyhow::bail!("rebuild BeamProc with --features iceoryx");
        } else {
            let socket = zmq_context.socket(zmq::PUSH).map_err(|err| {
                anyhow::anyhow!("failed to create EngineCore TX PUSH socket: {err}")
            })?;
            socket
                .set_linger(0)
                .map_err(|err| anyhow::anyhow!("failed to set EngineCore TX PUSH linger: {err}"))?;
            configure_enginecore_push_socket(&socket, self.zmq_hwm)?;
            socket
                .bind(&self.enginecore_beamproc_rx_addr)
                .map_err(|err| {
                    anyhow::anyhow!(
                        "failed to bind EngineCore TX PUSH socket at {}: {err}",
                        self.enginecore_beamproc_rx_addr
                    )
                })?;
            if let Some(handle) = start_zmq_monitor(&zmq_context, &socket, "enginecore-tx") {
                self.monitor_threads.push(handle);
            }
            CeSender::Zmq(socket)
        };

        if let Some(handle) =
            start_zmq_monitor(&zmq_context, &apiserver_output_socket, "apiserver-output")
        {
            self.monitor_threads.push(handle);
        }

        warn!(
            enginecore_beamproc_rx_addr = %self.enginecore_beamproc_rx_addr,
            apiserver_output_addr = %self.apiserver_output_addr,
            plan_receiver = true,
            ce_transport = if use_iceoryx { "iceoryx2" } else { "zmq" },
            "TX I/O worker initialized CE transport and APIServer ZMQ socket"
        );

        self.enginecore_tx_socket = Some(sender);
        self.apiserver_output_socket = Some(apiserver_output_socket);
        self.zmq_context = Some(zmq_context);
        Ok(())
    }

    fn poll_zmq(&mut self) {
        if self.enginecore_tx_socket.is_none() || self.apiserver_output_socket.is_none() {
            thread::yield_now();
            return;
        }

        self.wait_for_worker_plan();
    }

    fn wait_for_worker_plan(&mut self) {
        if self.plan_rxs.is_empty() {
            thread::yield_now();
            return;
        }

        let first = {
            let mut select = Select::new();
            for plan_rx in &self.plan_rxs {
                select.recv(plan_rx);
            }
            let operation =
                match select.select_timeout(Duration::from_millis(TX_PLAN_RECV_TIMEOUT_MS)) {
                    Ok(operation) => operation,
                    Err(_) => return,
                };
            let worker_index = operation.index();
            match operation.recv(&self.plan_rxs[worker_index]) {
                Ok(plan) => Some((worker_index, plan)),
                Err(_) => {
                    debug!(worker_index, "TX I/O worker plan receiver disconnected");
                    None
                }
            }
        };

        let Some((first_worker_index, first_plan)) = first else {
            return;
        };
        self.send_beam_plan(first_plan);

        // Drain at most one plan per worker per round so a busy worker cannot
        // starve the other workers' TX queues.
        let worker_count = self.plan_rxs.len();
        let mut round_start = (first_worker_index + 1) % worker_count;
        loop {
            let mut received_any = false;
            for offset in 0..worker_count {
                let worker_index = (round_start + offset) % worker_count;
                match self.plan_rxs[worker_index].try_recv() {
                    Ok(plan) => {
                        received_any = true;
                        self.send_beam_plan(plan);
                    }
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => {
                        debug!(worker_index, "TX I/O worker plan receiver disconnected");
                    }
                }
            }
            if !received_any {
                break;
            }
            round_start = (round_start + 1) % worker_count;
        }
    }

    fn send_beam_plan(&mut self, plan: BeamWorkerPlan) {
        if !plan.finished {
            self.send_beam_plan_v4(plan);
            return;
        }

        // EngineCore cleans scheduled final steps before forwarding their
        // outputs. Only an early EOS/catalog finish needs a cleanup notice.
        if !plan.scheduled_final {
            self.send_beam_finished(&plan);
        }

        let mut final_token_offsets = Vec::with_capacity(plan.final_token_ids.len() + 1);
        let final_token_count = plan.final_token_ids.iter().map(Vec::len).sum::<usize>();
        let mut final_token_ids = Vec::with_capacity(final_token_count);
        final_token_offsets.push(0);
        for tokens in &plan.final_token_ids {
            final_token_ids.extend_from_slice(tokens);
            final_token_offsets.push(final_token_ids.len());
        }

        let plan_meta = BeamFinalPayloadV3 {
            beam_timing: plan.timing,
            num_processed_tokens: plan.num_processed_tokens,
            version: 3,
            engine_index: plan.engine_index,
            client_index: plan.client_index,
            request_id: &plan.request_id,
            root_request_id: &plan.root_request_id,
            correlation_id: &plan.correlation_id,
            action: &plan.action,
            beam_width: plan.beam_width,
            completed_count: plan.completed_count,
            finished: plan.finished,
            final_token_ids_include_prompt: plan.final_token_ids_include_prompt,
            final_finish_reasons: &plan.final_finish_reasons,
            final_stop_token_ids: &plan.final_stop_token_ids,
            final_beams: plan.final_token_ids.len(),
            final_token_ids_len: final_token_ids.len(),
            final_token_offsets_len: final_token_offsets.len(),
            final_scores_len: plan.final_scores.len(),
            final_logprob_token_ids_len: plan.final_logprob_token_ids.len(),
            final_logprob_values_len: plan.final_logprob_values.len(),
            final_logprob_step_offsets_len: plan.final_logprob_step_offsets.len(),
            final_logprob_beam_offsets_len: plan.final_logprob_beam_offsets.len(),
        };
        let payload = match rmp_serde::to_vec_named(&plan_meta) {
            Ok(payload) => payload,
            Err(err) => {
                warn!(
                    error = %err,
                    correlation_id = %plan.correlation_id,
                    request_id = %plan.request_id,
                    "TX I/O worker failed to encode BEAM_PLAN"
                );
                return;
            }
        };
        let final_token_offset_bytes = usize_slice_as_u64_le_bytes(&final_token_offsets);

        if plan.finished {
            if let Some(apiserver_output_socket) = self.apiserver_output_socket.as_ref() {
                let final_token_bytes = typed_slice_as_bytes(&final_token_ids);
                let final_score_bytes = typed_slice_as_bytes(&plan.final_scores);
                let final_logprob_token_bytes = typed_slice_as_bytes(&plan.final_logprob_token_ids);
                let final_logprob_value_bytes = typed_slice_as_bytes(&plan.final_logprob_values);
                let final_logprob_step_offset_bytes =
                    usize_slice_as_u64_le_bytes(&plan.final_logprob_step_offsets);
                let final_logprob_beam_offset_bytes =
                    usize_slice_as_u64_le_bytes(&plan.final_logprob_beam_offsets);
                let frame_bytes = payload.len()
                    + final_token_bytes.len()
                    + final_token_offset_bytes.len()
                    + final_score_bytes.len()
                    + final_logprob_token_bytes.len()
                    + final_logprob_value_bytes.len()
                    + final_logprob_step_offset_bytes.len()
                    + final_logprob_beam_offset_bytes.len();
                let frames = [
                    MSG_BEAM_FINAL.as_bytes(),
                    payload.as_slice(),
                    final_token_bytes,
                    final_token_offset_bytes.as_slice(),
                    final_score_bytes,
                    final_logprob_token_bytes,
                    final_logprob_value_bytes,
                    final_logprob_step_offset_bytes.as_slice(),
                    final_logprob_beam_offset_bytes.as_slice(),
                ];
                match apiserver_output_socket.send_multipart(frames, 0) {
                    Ok(()) => {
                        warn!(
                            engine_index = plan.engine_index,
                            client_index = plan.client_index,
                            correlation_id = %plan.correlation_id,
                            request_id = %plan.request_id,
                            root_request_id = %plan.root_request_id,
                            apiserver_output_addr = %self.apiserver_output_addr,
                            action = %plan.action,
                            beam_width = plan.beam_width,
                            parent_ids = plan.parent_ids.len(),
                            token_ids = plan.token_ids.len(),
                            final_token_ids = plan.final_token_ids.len(),
                            first_final_len = plan
                                .final_token_ids
                                .first()
                                .map(|tokens| tokens.len())
                                .unwrap_or(0),
                            payload_bytes = frame_bytes,
                            "TX I/O worker sent APIServer BEAM_FINAL"
                        );
                    }
                    Err(err) => panic!(
                        "failed to deliver APIServer BEAM_FINAL for request {} within {}ms: {}",
                        plan.request_id, TX_SEND_TIMEOUT_MS, err
                    ),
                }
            } else {
                warn!(
                    engine_index = plan.engine_index,
                    client_index = plan.client_index,
                    correlation_id = %plan.correlation_id,
                    request_id = %plan.request_id,
                    "TX I/O worker has no APIServer route for BEAM_FINAL"
                );
            }
            return;
        }

        let Some(socket) = self.enginecore_tx_socket.as_ref() else {
            return;
        };
        let frames = [MSG_BEAM_PLAN.as_bytes(), payload.as_slice()];

        match socket.send_multipart(frames, 0) {
            Ok(()) => {
                debug!(
                    engine_index = plan.engine_index,
                    client_index = plan.client_index,
                    correlation_id = %plan.correlation_id,
                    request_id = %plan.request_id,
                    root_request_id = %plan.root_request_id,
                    action = %plan.action,
                    beam_width = plan.beam_width,
                    scored_parent_ids = plan.scored_parent_ids.len(),
                    parent_ids = plan.parent_ids.len(),
                    child_ids = plan.child_ids.len(),
                    token_ids = plan.token_ids.len(),
                    flat_indices = plan.flat_indices.len(),
                    abort_ids = plan.abort_ids.len(),
                    completed_count = plan.completed_count,
                    finished = plan.finished,
                    "TX I/O worker sent BEAM_PLAN"
                );
            }
            Err(err) => panic!(
                "failed to deliver BEAM_PLAN for request {} within {}ms: {}",
                plan.request_id, TX_SEND_TIMEOUT_MS, err
            ),
        }
    }

    fn send_beam_finished(&mut self, plan: &BeamWorkerPlan) {
        let Some(socket) = self.enginecore_tx_socket.as_ref() else {
            return;
        };
        let metadata = BeamFinishedPayloadV1 {
            version: 1,
            engine_index: plan.engine_index,
            client_index: plan.client_index,
            correlation_id: &plan.correlation_id,
            request_id: &plan.request_id,
            root_request_id: &plan.root_request_id,
            request_ids: &plan.scored_parent_ids,
        };
        let payload = match rmp_serde::to_vec_named(&metadata) {
            Ok(payload) => payload,
            Err(err) => {
                warn!(
                    error = %err,
                    root_request_id = %plan.root_request_id,
                    "TX I/O worker failed to encode BEAM_FINISHED"
                );
                return;
            }
        };
        let frames = [MSG_BEAM_FINISHED.as_bytes(), payload.as_slice()];
        if let Err(err) = socket.send_multipart(frames, 0) {
            panic!(
                "failed to deliver BEAM_FINISHED for request {} within {}ms: {}",
                plan.request_id, TX_SEND_TIMEOUT_MS, err
            );
        }
    }

    fn send_beam_plan_v4(&mut self, plan: BeamWorkerPlan) {
        let beam_token_stride = plan.beam_tokens.first().map(Vec::len).unwrap_or(0);
        if beam_token_stride == 0
            || plan
                .beam_tokens
                .iter()
                .any(|tokens| tokens.len() != beam_token_stride)
        {
            warn!(
                correlation_id = %plan.correlation_id,
                request_id = %plan.request_id,
                beam_token_stride,
                beam_histories = plan.beam_tokens.len(),
                "TX I/O worker falling back to BEAM_PLAN v3 for ragged histories"
            );
            self.send_beam_plan_v3(&plan);
            return;
        }

        if beamproc_enginecore_applies_live_plans()
            && plan.action == "observed_full"
            && !plan.finished
        {
            self.send_beam_plan_v5(plan, beam_token_stride);
            return;
        }

        let Some(socket) = self.enginecore_tx_socket.as_ref() else {
            return;
        };
        let beam_token_count = plan.beam_tokens.len() * beam_token_stride;
        let mut beam_tokens = Vec::with_capacity(beam_token_count);
        for tokens in &plan.beam_tokens {
            beam_tokens.extend_from_slice(tokens);
        }

        let plan_meta = BeamPlanPayloadV4 {
            version: 4,
            engine_index: plan.engine_index,
            client_index: plan.client_index,
            correlation_id: &plan.correlation_id,
            request_id: &plan.request_id,
            root_request_id: &plan.root_request_id,
            action: &plan.action,
            beam_width: plan.beam_width,
            scored_parent_ids: &plan.scored_parent_ids,
            parent_ids: &plan.parent_ids,
            child_ids: &plan.child_ids,
            abort_ids: &plan.abort_ids,
            completed_count: plan.completed_count,
            finished: plan.finished,
            token_ids_len: plan.token_ids.len(),
            scores_len: plan.scores.len(),
            flat_indices_len: plan.flat_indices.len(),
            beam_tokens_len: beam_tokens.len(),
            beam_token_stride,
        };

        let payload = match rmp_serde::to_vec_named(&plan_meta) {
            Ok(payload) => payload,
            Err(err) => {
                warn!(
                    error = %err,
                    correlation_id = %plan.correlation_id,
                    request_id = %plan.request_id,
                    "TX I/O worker failed to encode BEAM_PLAN v4 metadata"
                );
                return;
            }
        };
        let flat_index_bytes = usize_slice_as_u64_le_bytes(&plan.flat_indices);
        let token_id_bytes = typed_slice_as_bytes(&plan.token_ids);
        let score_bytes = typed_slice_as_bytes(&plan.scores);
        let beam_token_bytes = typed_slice_as_bytes(&beam_tokens);
        let frame_bytes = payload.len()
            + token_id_bytes.len()
            + score_bytes.len()
            + flat_index_bytes.len()
            + beam_token_bytes.len();
        let frames = [
            MSG_BEAM_PLAN.as_bytes(),
            payload.as_slice(),
            token_id_bytes,
            score_bytes,
            flat_index_bytes.as_slice(),
            beam_token_bytes,
        ];

        match socket.send_multipart(frames, 0) {
            Ok(()) => {
                debug!(
                    root_request_id = %plan.root_request_id,
                    beam_width = plan.beam_width,
                    beam_token_stride,
                    beam_token_values = beam_tokens.len(),
                    frame_bytes,
                    "TX I/O worker sent BEAM_PLAN v4"
                );
            }
            Err(err) => panic!(
                "failed to deliver BEAM_PLAN v4 for request {} within {}ms: {}",
                plan.request_id, TX_SEND_TIMEOUT_MS, err
            ),
        }
    }

    fn send_beam_plan_v5(&mut self, plan: BeamWorkerPlan, beam_token_stride: usize) {
        let Some(socket) = self.enginecore_tx_socket.as_ref() else {
            return;
        };
        let beam_token_count = plan.beam_tokens.len() * beam_token_stride;
        let mut beam_tokens = Vec::with_capacity(beam_token_count);
        for tokens in &plan.beam_tokens {
            beam_tokens.extend_from_slice(tokens);
        }

        let plan_meta = BeamPlanPayloadV5 {
            version: 5,
            engine_index: plan.engine_index,
            client_index: plan.client_index,
            correlation_id: &plan.correlation_id,
            request_id: &plan.request_id,
            root_request_id: &plan.root_request_id,
            action: &plan.action,
            beam_width: plan.beam_width,
            parent_ids: &plan.parent_ids,
            child_ids: &plan.child_ids,
            abort_ids: &plan.abort_ids,
            completed_count: plan.completed_count,
            finished: plan.finished,
            token_ids_len: plan.token_ids.len(),
            beam_tokens_len: beam_tokens.len(),
            beam_token_stride,
        };

        let payload = match rmp_serde::to_vec_named(&plan_meta) {
            Ok(payload) => payload,
            Err(err) => {
                warn!(
                    error = %err,
                    correlation_id = %plan.correlation_id,
                    request_id = %plan.request_id,
                    "TX I/O worker failed to encode BEAM_PLAN v5 metadata"
                );
                return;
            }
        };
        let token_id_bytes = typed_slice_as_bytes(&plan.token_ids);
        let beam_token_bytes = typed_slice_as_bytes(&beam_tokens);
        let frame_bytes = payload.len() + token_id_bytes.len() + beam_token_bytes.len();
        let frames = [
            MSG_BEAM_PLAN.as_bytes(),
            payload.as_slice(),
            token_id_bytes,
            beam_token_bytes,
        ];

        match socket.send_multipart(frames, 0) {
            Ok(()) => {
                debug!(
                    root_request_id = %plan.root_request_id,
                    beam_width = plan.beam_width,
                    beam_token_stride,
                    beam_token_values = beam_tokens.len(),
                    frame_bytes,
                    "TX I/O worker sent compact BEAM_PLAN v5"
                );
            }
            Err(err) => panic!(
                "failed to deliver BEAM_PLAN v5 for request {} within {}ms: {}",
                plan.request_id, TX_SEND_TIMEOUT_MS, err
            ),
        }
    }

    fn send_beam_plan_v3(&mut self, plan: &BeamWorkerPlan) {
        let Some(socket) = self.enginecore_tx_socket.as_ref() else {
            return;
        };
        let offset_capacity = if plan.finished {
            plan.beam_width + 1
        } else {
            plan.beam_tokens.len() + 1
        };
        let mut beam_token_offsets = Vec::with_capacity(offset_capacity);
        let beam_token_count = plan.beam_tokens.iter().map(Vec::len).sum::<usize>();
        let mut beam_tokens = Vec::with_capacity(beam_token_count);
        beam_token_offsets.push(0);
        for tokens in &plan.beam_tokens {
            beam_tokens.extend_from_slice(tokens);
            beam_token_offsets.push(beam_tokens.len());
        }
        if plan.finished {
            beam_token_offsets.resize(plan.beam_width + 1, beam_tokens.len());
        }

        let plan_meta = BeamPlanPayloadV3 {
            version: 3,
            engine_index: plan.engine_index,
            client_index: plan.client_index,
            correlation_id: &plan.correlation_id,
            request_id: &plan.request_id,
            root_request_id: &plan.root_request_id,
            action: &plan.action,
            beam_width: plan.beam_width,
            scored_parent_ids: &plan.scored_parent_ids,
            parent_ids: &plan.parent_ids,
            child_ids: &plan.child_ids,
            abort_ids: &plan.abort_ids,
            completed_count: plan.completed_count,
            finished: plan.finished,
            token_ids_len: plan.token_ids.len(),
            scores_len: plan.scores.len(),
            flat_indices_len: plan.flat_indices.len(),
            beam_tokens_len: beam_tokens.len(),
            beam_token_offsets_len: beam_token_offsets.len(),
        };

        let payload = match rmp_serde::to_vec_named(&plan_meta) {
            Ok(payload) => payload,
            Err(err) => {
                warn!(
                    error = %err,
                    correlation_id = %plan.correlation_id,
                    request_id = %plan.request_id,
                    "TX I/O worker failed to encode BEAM_PLAN v3 metadata"
                );
                return;
            }
        };
        let flat_index_bytes = usize_slice_as_u64_le_bytes(&plan.flat_indices);
        let beam_token_offset_bytes = usize_slice_as_u64_le_bytes(&beam_token_offsets);

        let token_id_bytes = typed_slice_as_bytes(&plan.token_ids);
        let score_bytes = typed_slice_as_bytes(&plan.scores);
        let beam_token_bytes = typed_slice_as_bytes(&beam_tokens);
        let frame_bytes = payload.len()
            + token_id_bytes.len()
            + score_bytes.len()
            + flat_index_bytes.len()
            + beam_token_bytes.len()
            + beam_token_offset_bytes.len();
        let frames = [
            MSG_BEAM_PLAN.as_bytes(),
            payload.as_slice(),
            token_id_bytes,
            score_bytes,
            flat_index_bytes.as_slice(),
            beam_token_bytes,
            beam_token_offset_bytes.as_slice(),
        ];

        match socket.send_multipart(frames, 0) {
            Ok(()) => {
                debug!(
                    engine_index = plan.engine_index,
                    client_index = plan.client_index,
                    correlation_id = %plan.correlation_id,
                    request_id = %plan.request_id,
                    root_request_id = %plan.root_request_id,
                    action = %plan.action,
                    beam_width = plan.beam_width,
                    scored_parent_ids = plan.scored_parent_ids.len(),
                    parent_ids = plan.parent_ids.len(),
                    child_ids = plan.child_ids.len(),
                    token_ids = plan.token_ids.len(),
                    beam_tokens = plan.beam_tokens.len(),
                    beam_token_values = beam_tokens.len(),
                    flat_indices = plan.flat_indices.len(),
                    abort_ids = plan.abort_ids.len(),
                    completed_count = plan.completed_count,
                    frame_bytes,
                    "TX I/O worker sent BEAM_PLAN v3"
                );
            }
            Err(err) => panic!(
                "failed to deliver BEAM_PLAN v3 for request {} within {}ms: {}",
                plan.request_id, TX_SEND_TIMEOUT_MS, err
            ),
        }
    }
}

struct EngineOutputRequestGroup {
    correlation_id: String,
    worker_index: usize,
    batch: BeamEngineOutputBatch,
    pool_return_tx: Option<Sender<BeamEngineOutputBatch>>,
}

struct FastDecodedEngineOutputBatch {
    engine_index: usize,
    client_index: usize,
    by_request: HashMap<String, EngineOutputRequestGroup>,
}

struct FastDecodedBeamContext<'a> {
    root_request_id: &'a str,
    beam_context: BeamContext,
}

#[derive(Debug)]
struct FastDecodeError;

struct MsgpackDecoder<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> MsgpackDecoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn engine_output_batch_version(&mut self) -> Result<Option<usize>, FastDecodeError> {
        let len = self.read_map_len()?;
        for _ in 0..len {
            let key = self.read_str()?;
            if key == "version" {
                return Ok(Some(self.read_usize()?));
            }
            self.skip_value()?;
        }
        Ok(None)
    }

    fn decode_v3_engine_output_batch(
        &mut self,
        worker: &mut IoRxState,
        token_id_bytes: &[u8],
        logprob_bytes: &[u8],
        rank_bytes: &[u8],
        new_token_id_bytes: &[u8],
    ) -> Result<FastDecodedEngineOutputBatch, FastDecodeError> {
        let len = self.read_map_len()?;
        let mut version = None;
        let mut engine_index = None;
        let mut client_index = None;
        let mut correlation_base = None;
        let mut contexts: Vec<Option<FastDecodedBeamContext>> = Vec::new();
        let mut by_request: HashMap<String, EngineOutputRequestGroup> = HashMap::new();

        for _ in 0..len {
            match self.read_str()? {
                "version" => version = Some(self.read_usize()?),
                "engine_index" => engine_index = Some(self.read_usize()?),
                "client_index" => client_index = Some(self.read_usize()?),
                "correlation_base" => correlation_base = self.read_optional_u64()?,
                "contexts" => contexts = self.read_contexts()?,
                "outputs" => {
                    if version != Some(6) {
                        return Err(FastDecodeError);
                    }
                    self.read_v3_outputs(
                        worker,
                        correlation_base,
                        &contexts,
                        token_id_bytes,
                        logprob_bytes,
                        rank_bytes,
                        new_token_id_bytes,
                        &mut by_request,
                    )?;
                }
                _ => self.skip_value()?,
            }
        }

        if version != Some(6) {
            return Err(FastDecodeError);
        }

        Ok(FastDecodedEngineOutputBatch {
            engine_index: engine_index.ok_or(FastDecodeError)?,
            client_index: client_index.ok_or(FastDecodeError)?,
            by_request,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn read_v3_outputs(
        &mut self,
        worker: &mut IoRxState,
        correlation_base: Option<u64>,
        contexts: &[Option<FastDecodedBeamContext<'a>>],
        token_id_bytes: &[u8],
        logprob_bytes: &[u8],
        rank_bytes: &[u8],
        new_token_id_bytes: &[u8],
        by_request: &mut HashMap<String, EngineOutputRequestGroup>,
    ) -> Result<(), FastDecodeError> {
        let len = self.read_array_len()?;
        for _ in 0..len {
            let output_len = self.read_array_len()?;
            if output_len != 12 {
                return Err(FastDecodeError);
            }

            let request_id = self.read_string_value()?;
            let new_token_offset = self.read_usize()?;
            let new_token_len = self.read_usize()?;
            let new_token_end = new_token_offset
                .checked_add(new_token_len)
                .ok_or(FastDecodeError)?;
            let new_token_ids =
                read_frame_range::<i32>(new_token_id_bytes, new_token_offset, new_token_end)?;
            let finish_reason = self.read_optional_string_value()?;
            let stop_reason = self.read_optional_string_value()?;
            let num_cached_tokens = self.read_optional_usize()?;
            let candidate_token_offset = self.read_usize()?;
            let candidate_rows = self.read_usize()?;
            let candidate_cols = self.read_usize()?;
            let sampled_rank_offset = self.read_usize()?;
            let sampled_rank_len = self.read_usize()?;
            let context_index = self.read_optional_usize()?;

            validate_output_binary_ranges(
                token_id_bytes,
                logprob_bytes,
                rank_bytes,
                candidate_token_offset,
                candidate_rows,
                candidate_cols,
                sampled_rank_offset,
                sampled_rank_len,
            )?;
            let decoded_context = context_index
                .and_then(|index| contexts.get(index))
                .and_then(Option::as_ref)
                .ok_or(FastDecodeError)?;
            let root_request_id = decoded_context.root_request_id;
            if !by_request.contains_key(root_request_id) {
                let correlation_id = format!("{}:{}", correlation_base.unwrap_or(0), request_id);
                let group = worker.claim_engine_output_group(root_request_id, correlation_id);
                by_request.insert(root_request_id.to_string(), group);
            }

            let Some(group) = by_request.get_mut(root_request_id) else {
                return Err(FastDecodeError);
            };
            let candidate_len = candidate_rows
                .checked_mul(candidate_cols)
                .ok_or(FastDecodeError)?;
            let candidate_end = candidate_token_offset
                .checked_add(candidate_len)
                .ok_or(FastDecodeError)?;
            let rank_end = sampled_rank_offset
                .checked_add(sampled_rank_len)
                .ok_or(FastDecodeError)?;
            let (candidate_token_start, candidate_token_end) = append_frame_range::<i32>(
                &mut group.batch.candidate_token_ids,
                token_id_bytes,
                candidate_token_offset,
                candidate_end,
            )?;
            let (candidate_logprob_start, candidate_logprob_end) = append_frame_range::<f32>(
                &mut group.batch.candidate_logprobs,
                logprob_bytes,
                candidate_token_offset,
                candidate_end,
            )?;
            let (sampled_rank_start, sampled_rank_end) = append_frame_range::<i32>(
                &mut group.batch.sampled_token_ranks,
                rank_bytes,
                sampled_rank_offset,
                rank_end,
            )?;
            group.batch.outputs.push(BeamEngineOutput {
                request_id,
                new_token_ids,
                finish_reason,
                stop_reason,
                num_cached_tokens,
                candidate_token_start,
                candidate_token_end,
                candidate_logprob_start,
                candidate_logprob_end,
                candidate_rows,
                candidate_cols,
                sampled_rank_start,
                sampled_rank_end,
                beam_context: Some(decoded_context.beam_context.clone()),
            });
        }
        Ok(())
    }

    fn read_contexts(
        &mut self,
    ) -> Result<Vec<Option<FastDecodedBeamContext<'a>>>, FastDecodeError> {
        let len = self.read_array_len()?;
        let mut contexts = Vec::with_capacity(len);
        for _ in 0..len {
            if self.peek_byte()? == 0xc0 {
                self.pos += 1;
                contexts.push(None);
            } else {
                contexts.push(Some(self.read_context()?));
            }
        }
        Ok(contexts)
    }

    fn read_context(&mut self) -> Result<FastDecodedBeamContext<'a>, FastDecodeError> {
        let len = self.read_map_len()?;
        let mut root_request_id = None;
        let mut context = BeamContext {
            beam_width: None,
            max_tokens: None,
            sampling_max_tokens: None,
            global_max_tokens: None,
            total_decode_steps: None,
            step_index: None,
            is_final_step: false,
            ignore_eos: false,
            include_stop_token: false,
            eos_token_id: None,
            end_token_id: None,
            catalog_strategy: "fill".to_string(),
            return_logprobs: true,
            parent_token_ids: Vec::new(),
            use_shadow_parent_tokens: false,
            prompt_token_count: None,
            client_index: 0,
            priority: 0,
        };

        for _ in 0..len {
            match self.read_str()? {
                "root_request_id" => root_request_id = Some(self.read_str()?),
                "beam_width" => context.beam_width = self.read_optional_usize()?,
                "max_tokens" => context.max_tokens = self.read_optional_usize()?,
                "sampling_max_tokens" => {
                    context.sampling_max_tokens = self.read_optional_usize()?
                }
                "global_max_tokens" => context.global_max_tokens = self.read_optional_usize()?,
                "total_decode_steps" => context.total_decode_steps = self.read_optional_usize()?,
                "step_index" => context.step_index = self.read_optional_usize()?,
                "is_final_step" => context.is_final_step = self.read_bool()?,
                "ignore_eos" => context.ignore_eos = self.read_bool()?,
                "include_stop_token" => context.include_stop_token = self.read_bool()?,
                "eos_token_id" => context.eos_token_id = self.read_optional_i32()?,
                "end_token_id" => context.end_token_id = self.read_optional_i32()?,
                "catalog_strategy" => context.catalog_strategy = self.read_str()?.to_string(),
                "return_logprobs" => context.return_logprobs = self.read_bool()?,
                "parent_token_ids" => context.parent_token_ids = self.read_i32_array()?,
                "use_shadow_parent_tokens" => {
                    context.use_shadow_parent_tokens = self.read_bool()?
                }
                "prompt_token_count" => context.prompt_token_count = self.read_optional_usize()?,
                "client_index" => context.client_index = self.read_usize()?,
                "priority" => context.priority = self.read_i64()?,
                _ => self.skip_value()?,
            }
        }
        Ok(FastDecodedBeamContext {
            root_request_id: root_request_id.ok_or(FastDecodeError)?,
            beam_context: context,
        })
    }

    fn read_i32_array(&mut self) -> Result<Vec<i32>, FastDecodeError> {
        let len = self.read_array_len()?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_i32_value()?);
        }
        Ok(values)
    }

    fn read_optional_i32(&mut self) -> Result<Option<i32>, FastDecodeError> {
        if self.peek_byte()? == 0xc0 {
            self.pos += 1;
            return Ok(None);
        }
        Ok(Some(self.read_i32_value()?))
    }

    fn read_i32_value(&mut self) -> Result<i32, FastDecodeError> {
        if self.peek_is_str()? {
            let value = self.read_str()?;
            return value.parse::<i32>().map_err(|_| FastDecodeError);
        }
        let value = self.read_i64()?;
        i32::try_from(value).map_err(|_| FastDecodeError)
    }

    fn read_optional_string_value(&mut self) -> Result<Option<String>, FastDecodeError> {
        if self.peek_byte()? == 0xc0 {
            self.pos += 1;
            return Ok(None);
        }
        Ok(Some(self.read_string_value()?))
    }

    fn read_string_value(&mut self) -> Result<String, FastDecodeError> {
        if self.peek_is_str()? {
            return Ok(self.read_str()?.to_string());
        }
        Ok(self.read_i64()?.to_string())
    }

    fn read_optional_usize(&mut self) -> Result<Option<usize>, FastDecodeError> {
        if self.peek_byte()? == 0xc0 {
            self.pos += 1;
            return Ok(None);
        }
        Ok(Some(self.read_usize()?))
    }

    fn read_optional_u64(&mut self) -> Result<Option<u64>, FastDecodeError> {
        if self.peek_byte()? == 0xc0 {
            self.pos += 1;
            return Ok(None);
        }
        Ok(Some(self.read_u64()?))
    }

    fn read_bool(&mut self) -> Result<bool, FastDecodeError> {
        match self.read_byte()? {
            0xc2 => Ok(false),
            0xc3 => Ok(true),
            _ => Err(FastDecodeError),
        }
    }

    fn read_usize(&mut self) -> Result<usize, FastDecodeError> {
        usize::try_from(self.read_u64()?).map_err(|_| FastDecodeError)
    }

    fn read_i64(&mut self) -> Result<i64, FastDecodeError> {
        let byte = self.read_byte()?;
        match byte {
            0x00..=0x7f => Ok(i64::from(byte)),
            0xcc => Ok(i64::from(self.read_u8()?)),
            0xcd => Ok(i64::from(self.read_u16()?)),
            0xce => Ok(i64::from(self.read_u32()?)),
            0xcf => i64::try_from(self.read_u64_raw()?).map_err(|_| FastDecodeError),
            0xd0 => Ok(i64::from(i8::from_be_bytes([self.read_u8()?]))),
            0xd1 => Ok(i64::from(i16::from_be_bytes(self.read_array()?))),
            0xd2 => Ok(i64::from(i32::from_be_bytes(self.read_array()?))),
            0xd3 => Ok(i64::from_be_bytes(self.read_array()?)),
            0xe0..=0xff => Ok(i64::from(i8::from_be_bytes([byte]))),
            _ => Err(FastDecodeError),
        }
    }

    fn read_u64(&mut self) -> Result<u64, FastDecodeError> {
        let byte = self.read_byte()?;
        match byte {
            0x00..=0x7f => Ok(u64::from(byte)),
            0xcc => Ok(u64::from(self.read_u8()?)),
            0xcd => Ok(u64::from(self.read_u16()?)),
            0xce => Ok(u64::from(self.read_u32()?)),
            0xcf => self.read_u64_raw(),
            _ => Err(FastDecodeError),
        }
    }

    fn read_map_len(&mut self) -> Result<usize, FastDecodeError> {
        let byte = self.read_byte()?;
        match byte {
            0x80..=0x8f => Ok(usize::from(byte & 0x0f)),
            0xde => Ok(usize::from(self.read_u16()?)),
            0xdf => usize::try_from(self.read_u32()?).map_err(|_| FastDecodeError),
            _ => Err(FastDecodeError),
        }
    }

    fn read_array_len(&mut self) -> Result<usize, FastDecodeError> {
        let byte = self.read_byte()?;
        match byte {
            0x90..=0x9f => Ok(usize::from(byte & 0x0f)),
            0xdc => Ok(usize::from(self.read_u16()?)),
            0xdd => usize::try_from(self.read_u32()?).map_err(|_| FastDecodeError),
            _ => Err(FastDecodeError),
        }
    }

    fn read_str(&mut self) -> Result<&'a str, FastDecodeError> {
        let byte = self.read_byte()?;
        let len = match byte {
            0xa0..=0xbf => usize::from(byte & 0x1f),
            0xd9 => usize::from(self.read_u8()?),
            0xda => usize::from(self.read_u16()?),
            0xdb => usize::try_from(self.read_u32()?).map_err(|_| FastDecodeError)?,
            _ => return Err(FastDecodeError),
        };
        let slice = self.read_slice(len)?;
        std::str::from_utf8(slice).map_err(|_| FastDecodeError)
    }

    fn skip_value(&mut self) -> Result<(), FastDecodeError> {
        let byte = self.read_byte()?;
        match byte {
            0x00..=0x7f | 0xc0 | 0xc2 | 0xc3 | 0xe0..=0xff => Ok(()),
            0xcc | 0xd0 => self.skip_bytes(1),
            0xcd | 0xd1 => self.skip_bytes(2),
            0xce | 0xd2 | 0xca => self.skip_bytes(4),
            0xcf | 0xd3 | 0xcb => self.skip_bytes(8),
            0xa0..=0xbf => self.skip_bytes(usize::from(byte & 0x1f)),
            0xc4 | 0xd9 => {
                let len = usize::from(self.read_u8()?);
                self.skip_bytes(len)
            }
            0xc5 | 0xda => {
                let len = usize::from(self.read_u16()?);
                self.skip_bytes(len)
            }
            0xc6 | 0xdb => {
                let len = usize::try_from(self.read_u32()?).map_err(|_| FastDecodeError)?;
                self.skip_bytes(len)
            }
            0x90..=0x9f => {
                for _ in 0..usize::from(byte & 0x0f) {
                    self.skip_value()?;
                }
                Ok(())
            }
            0xdc => {
                for _ in 0..usize::from(self.read_u16()?) {
                    self.skip_value()?;
                }
                Ok(())
            }
            0xdd => {
                let len = usize::try_from(self.read_u32()?).map_err(|_| FastDecodeError)?;
                for _ in 0..len {
                    self.skip_value()?;
                }
                Ok(())
            }
            0x80..=0x8f => {
                for _ in 0..usize::from(byte & 0x0f) {
                    self.skip_value()?;
                    self.skip_value()?;
                }
                Ok(())
            }
            0xde => {
                for _ in 0..usize::from(self.read_u16()?) {
                    self.skip_value()?;
                    self.skip_value()?;
                }
                Ok(())
            }
            0xdf => {
                let len = usize::try_from(self.read_u32()?).map_err(|_| FastDecodeError)?;
                for _ in 0..len {
                    self.skip_value()?;
                    self.skip_value()?;
                }
                Ok(())
            }
            _ => Err(FastDecodeError),
        }
    }

    fn peek_is_str(&self) -> Result<bool, FastDecodeError> {
        let byte = self.peek_byte()?;
        Ok(matches!(byte, 0xa0..=0xbf | 0xd9 | 0xda | 0xdb))
    }

    fn peek_byte(&self) -> Result<u8, FastDecodeError> {
        self.bytes.get(self.pos).copied().ok_or(FastDecodeError)
    }

    fn read_byte(&mut self) -> Result<u8, FastDecodeError> {
        let byte = self.peek_byte()?;
        self.pos += 1;
        Ok(byte)
    }

    fn read_u8(&mut self) -> Result<u8, FastDecodeError> {
        self.read_byte()
    }

    fn read_u16(&mut self) -> Result<u16, FastDecodeError> {
        Ok(u16::from_be_bytes(self.read_array()?))
    }

    fn read_u32(&mut self) -> Result<u32, FastDecodeError> {
        Ok(u32::from_be_bytes(self.read_array()?))
    }

    fn read_u64_raw(&mut self) -> Result<u64, FastDecodeError> {
        Ok(u64::from_be_bytes(self.read_array()?))
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], FastDecodeError> {
        let slice = self.read_slice(N)?;
        slice.try_into().map_err(|_| FastDecodeError)
    }

    fn read_slice(&mut self, len: usize) -> Result<&'a [u8], FastDecodeError> {
        let end = self.pos.checked_add(len).ok_or(FastDecodeError)?;
        let slice = self.bytes.get(self.pos..end).ok_or(FastDecodeError)?;
        self.pos = end;
        Ok(slice)
    }

    fn skip_bytes(&mut self, len: usize) -> Result<(), FastDecodeError> {
        self.read_slice(len).map(|_| ())
    }
}

#[derive(Serialize)]
struct BeamPlanPayloadV3<'a> {
    version: u32,
    engine_index: usize,
    client_index: usize,
    correlation_id: &'a str,
    request_id: &'a str,
    root_request_id: &'a str,
    action: &'a str,
    beam_width: usize,
    scored_parent_ids: &'a [String],
    parent_ids: &'a [String],
    child_ids: &'a [String],
    abort_ids: &'a [String],
    completed_count: usize,
    finished: bool,
    token_ids_len: usize,
    scores_len: usize,
    flat_indices_len: usize,
    beam_tokens_len: usize,
    beam_token_offsets_len: usize,
}

#[derive(Serialize)]
struct BeamFinishedPayloadV1<'a> {
    version: u32,
    engine_index: usize,
    client_index: usize,
    correlation_id: &'a str,
    request_id: &'a str,
    root_request_id: &'a str,
    request_ids: &'a [String],
}

#[derive(Serialize)]
struct BeamPlanPayloadV4<'a> {
    version: u32,
    engine_index: usize,
    client_index: usize,
    correlation_id: &'a str,
    request_id: &'a str,
    root_request_id: &'a str,
    action: &'a str,
    beam_width: usize,
    scored_parent_ids: &'a [String],
    parent_ids: &'a [String],
    child_ids: &'a [String],
    abort_ids: &'a [String],
    completed_count: usize,
    finished: bool,
    token_ids_len: usize,
    scores_len: usize,
    flat_indices_len: usize,
    beam_tokens_len: usize,
    beam_token_stride: usize,
}

#[derive(Serialize)]
struct BeamPlanPayloadV5<'a> {
    version: u32,
    engine_index: usize,
    client_index: usize,
    correlation_id: &'a str,
    request_id: &'a str,
    root_request_id: &'a str,
    action: &'a str,
    beam_width: usize,
    parent_ids: &'a [String],
    child_ids: &'a [String],
    abort_ids: &'a [String],
    completed_count: usize,
    finished: bool,
    token_ids_len: usize,
    beam_tokens_len: usize,
    beam_token_stride: usize,
}

#[derive(Serialize)]
struct BeamFinalPayloadV3<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    beam_timing: Option<super::beam_worker::BeamTiming>,
    num_processed_tokens: u64,
    version: u32,
    engine_index: usize,
    client_index: usize,
    request_id: &'a str,
    root_request_id: &'a str,
    correlation_id: &'a str,
    action: &'a str,
    beam_width: usize,
    completed_count: usize,
    finished: bool,
    final_token_ids_include_prompt: bool,
    final_finish_reasons: &'a [String],
    final_stop_token_ids: &'a [Option<i32>],
    final_beams: usize,
    final_token_ids_len: usize,
    final_token_offsets_len: usize,
    final_scores_len: usize,
    final_logprob_token_ids_len: usize,
    final_logprob_values_len: usize,
    final_logprob_step_offsets_len: usize,
    final_logprob_beam_offsets_len: usize,
}

#[derive(Debug, Deserialize)]
struct BeamStateUpdatePayload {
    version: u32,
    root_request_id: ScalarValue,
    beam_width: usize,
    active_parent_ids: Vec<ScalarValue>,
    active_parent_cum_logprobs: Vec<f32>,
    #[serde(default)]
    active_parent_valid_token_ids: Vec<Vec<ScalarValue>>,
}

#[derive(Debug, Deserialize)]
struct BeamAbortPayload {
    #[serde(default)]
    root_request_ids: Vec<ScalarValue>,
}

#[derive(Debug, Deserialize)]
struct EngineOutputBatchPayload {
    version: u32,
    engine_index: usize,
    client_index: usize,
    #[serde(default)]
    correlation_base: Option<u64>,
    #[serde(default)]
    contexts: Vec<Option<BeamContextPayload>>,
    outputs: Vec<EngineOutputPayloadWire>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EngineOutputPayloadWire {
    V1(EngineOutputPayload),
    V2(EngineOutputPayloadV2),
}

#[derive(Debug, Deserialize)]
struct EngineOutputPayload {
    request_id: ScalarValue,
    root_request_id: ScalarValue,
    correlation_id: ScalarValue,
    new_token_ids: Vec<i32>,
    finish_reason: Option<ScalarValue>,
    stop_reason: Option<ScalarValue>,
    num_cached_tokens: Option<usize>,
    #[serde(default)]
    candidate_token_offset: usize,
    #[serde(default)]
    candidate_rows: usize,
    #[serde(default)]
    candidate_cols: usize,
    #[serde(default)]
    sampled_rank_offset: usize,
    #[serde(default)]
    sampled_rank_len: usize,
    #[serde(default)]
    beam_context: Option<BeamContextPayload>,
}

#[derive(Debug, Deserialize)]
struct EngineOutputPayloadV2(
    ScalarValue,
    ScalarValue,
    ScalarValue,
    Vec<i32>,
    Option<ScalarValue>,
    Option<ScalarValue>,
    Option<usize>,
    usize,
    usize,
    usize,
    usize,
    usize,
    Option<BeamContextPayload>,
);

#[derive(Debug, Clone, Deserialize)]
struct BeamContextPayload {
    #[serde(default)]
    beam_width: Option<usize>,
    max_tokens: Option<usize>,
    sampling_max_tokens: Option<usize>,
    global_max_tokens: Option<usize>,
    total_decode_steps: Option<usize>,
    step_index: Option<usize>,
    #[serde(default)]
    is_final_step: bool,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    include_stop_token: bool,
    eos_token_id: Option<ScalarValue>,
    end_token_id: Option<ScalarValue>,
    #[serde(default = "default_catalog_strategy")]
    catalog_strategy: String,
    #[serde(default = "default_return_logprobs")]
    return_logprobs: bool,
    #[serde(default)]
    parent_token_ids: Vec<ScalarValue>,
    #[serde(default)]
    use_shadow_parent_tokens: bool,
    prompt_token_count: Option<usize>,
    #[serde(default)]
    client_index: usize,
    #[serde(default)]
    priority: i64,
}

fn default_catalog_strategy() -> String {
    "fill".to_string()
}

fn default_return_logprobs() -> bool {
    true
}

impl EngineOutputPayload {
    fn root_request_id_string(&self) -> String {
        self.root_request_id.to_string_value()
    }

    fn correlation_id_string(&self) -> String {
        self.correlation_id.to_string_value()
    }

    fn append_engine_output(
        self,
        token_id_bytes: &[u8],
        logprob_bytes: &[u8],
        rank_bytes: &[u8],
        batch: &mut BeamEngineOutputBatch,
    ) -> Result<(), FastDecodeError> {
        let candidate_len = self
            .candidate_rows
            .checked_mul(self.candidate_cols)
            .ok_or(FastDecodeError)?;
        let candidate_end = self
            .candidate_token_offset
            .checked_add(candidate_len)
            .ok_or(FastDecodeError)?;
        let rank_end = self
            .sampled_rank_offset
            .checked_add(self.sampled_rank_len)
            .ok_or(FastDecodeError)?;

        let (candidate_token_start, candidate_token_end) = append_frame_range::<i32>(
            &mut batch.candidate_token_ids,
            token_id_bytes,
            self.candidate_token_offset,
            candidate_end,
        )?;
        let (candidate_logprob_start, candidate_logprob_end) = append_frame_range::<f32>(
            &mut batch.candidate_logprobs,
            logprob_bytes,
            self.candidate_token_offset,
            candidate_end,
        )?;
        let (sampled_rank_start, sampled_rank_end) = append_frame_range::<i32>(
            &mut batch.sampled_token_ranks,
            rank_bytes,
            self.sampled_rank_offset,
            rank_end,
        )?;

        batch.outputs.push(BeamEngineOutput {
            request_id: self.request_id.into_string(),
            new_token_ids: self.new_token_ids,
            finish_reason: self.finish_reason.map(ScalarValue::into_string),
            stop_reason: self.stop_reason.map(ScalarValue::into_string),
            num_cached_tokens: self.num_cached_tokens,
            candidate_token_start,
            candidate_token_end,
            candidate_logprob_start,
            candidate_logprob_end,
            candidate_rows: self.candidate_rows,
            candidate_cols: self.candidate_cols,
            sampled_rank_start,
            sampled_rank_end,
            beam_context: self.beam_context.map(BeamContextPayload::into_beam_context),
        });
        Ok(())
    }
}

impl EngineOutputPayloadWire {
    fn validate_binary_ranges(
        &self,
        token_id_bytes: &[u8],
        logprob_bytes: &[u8],
        rank_bytes: &[u8],
    ) -> Result<(), FastDecodeError> {
        let (candidate_offset, candidate_rows, candidate_cols, rank_offset, rank_len) = match self {
            Self::V1(payload) => (
                payload.candidate_token_offset,
                payload.candidate_rows,
                payload.candidate_cols,
                payload.sampled_rank_offset,
                payload.sampled_rank_len,
            ),
            Self::V2(payload) => (payload.7, payload.8, payload.9, payload.10, payload.11),
        };
        validate_output_binary_ranges(
            token_id_bytes,
            logprob_bytes,
            rank_bytes,
            candidate_offset,
            candidate_rows,
            candidate_cols,
            rank_offset,
            rank_len,
        )
    }

    fn route_ids(
        &self,
        _correlation_base: Option<u64>,
        _contexts: &[Option<BeamContextPayload>],
    ) -> (String, String) {
        match self {
            Self::V1(payload) => (
                payload.root_request_id_string(),
                payload.correlation_id_string(),
            ),
            Self::V2(payload) => (payload.1.to_string_value(), payload.2.to_string_value()),
        }
    }

    fn append_routed_output(
        self,
        _correlation_base: Option<u64>,
        _contexts: &[Option<BeamContextPayload>],
        token_id_bytes: &[u8],
        logprob_bytes: &[u8],
        rank_bytes: &[u8],
        batch: &mut BeamEngineOutputBatch,
    ) -> Result<(), FastDecodeError> {
        match self {
            Self::V1(payload) => {
                payload.append_routed_output(token_id_bytes, logprob_bytes, rank_bytes, batch)
            }
            Self::V2(payload) => payload.into_payload().append_routed_output(
                token_id_bytes,
                logprob_bytes,
                rank_bytes,
                batch,
            ),
        }
    }
}

impl EngineOutputPayload {
    fn append_routed_output(
        self,
        token_id_bytes: &[u8],
        logprob_bytes: &[u8],
        rank_bytes: &[u8],
        batch: &mut BeamEngineOutputBatch,
    ) -> Result<(), FastDecodeError> {
        self.append_engine_output(token_id_bytes, logprob_bytes, rank_bytes, batch)
    }
}

impl EngineOutputPayloadV2 {
    fn into_payload(self) -> EngineOutputPayload {
        EngineOutputPayload {
            request_id: self.0,
            root_request_id: self.1,
            correlation_id: self.2,
            new_token_ids: self.3,
            finish_reason: self.4,
            stop_reason: self.5,
            num_cached_tokens: self.6,
            candidate_token_offset: self.7,
            candidate_rows: self.8,
            candidate_cols: self.9,
            sampled_rank_offset: self.10,
            sampled_rank_len: self.11,
            beam_context: self.12,
        }
    }
}

fn beamproc_enginecore_applies_live_plans() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(BEAMPROC_ENABLED_ENV)
            .map(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "y" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn beamproc_output_pool_slots() -> usize {
    env::var(BEAMPROC_OUTPUT_POOL_SLOTS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_OUTPUT_POOL_SLOTS)
}

fn beamproc_output_pool_capacity() -> usize {
    env::var(BEAMPROC_OUTPUT_POOL_CAPACITY_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_OUTPUT_POOL_CAPACITY)
}

impl BeamContextPayload {
    fn into_beam_context(self) -> BeamContext {
        BeamContext {
            beam_width: self.beam_width,
            max_tokens: self.max_tokens,
            sampling_max_tokens: self.sampling_max_tokens,
            global_max_tokens: self.global_max_tokens,
            total_decode_steps: self.total_decode_steps,
            step_index: self.step_index,
            is_final_step: self.is_final_step,
            ignore_eos: self.ignore_eos,
            include_stop_token: self.include_stop_token,
            eos_token_id: self.eos_token_id.and_then(ScalarValue::into_i32),
            end_token_id: self.end_token_id.and_then(ScalarValue::into_i32),
            catalog_strategy: self.catalog_strategy,
            return_logprobs: self.return_logprobs,
            parent_token_ids: self
                .parent_token_ids
                .into_iter()
                .filter_map(ScalarValue::into_i32)
                .collect(),
            use_shadow_parent_tokens: self.use_shadow_parent_tokens,
            prompt_token_count: self.prompt_token_count,
            client_index: self.client_index,
            priority: self.priority,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ScalarValue {
    String(String),
    Integer(i64),
    Unsigned(u64),
}

impl ScalarValue {
    fn to_string_value(&self) -> String {
        match self {
            Self::String(value) => value.clone(),
            Self::Integer(value) => value.to_string(),
            Self::Unsigned(value) => value.to_string(),
        }
    }

    fn into_string(self) -> String {
        match self {
            Self::String(value) => value,
            Self::Integer(value) => value.to_string(),
            Self::Unsigned(value) => value.to_string(),
        }
    }

    fn into_i32(self) -> Option<i32> {
        match self {
            Self::String(value) => value.parse::<i32>().ok(),
            Self::Integer(value) => i32::try_from(value).ok(),
            Self::Unsigned(value) => i32::try_from(value).ok(),
        }
    }
}

fn validate_binary_frame<T>(bytes: &[u8], field: &str) -> bool {
    if !bytes.len().is_multiple_of(mem::size_of::<T>()) {
        warn!(
            field,
            bytes = bytes.len(),
            "I/O worker received malformed binary frame"
        );
        return false;
    }
    true
}

fn append_frame_range<T: Copy>(
    target: &mut Vec<T>,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<(usize, usize), FastDecodeError> {
    validate_frame_range::<T>(bytes, start, end)?;
    let item_size = mem::size_of::<T>();
    let byte_start = start.checked_mul(item_size).ok_or(FastDecodeError)?;
    let byte_end = end.checked_mul(item_size).ok_or(FastDecodeError)?;
    let slice = bytes.get(byte_start..byte_end).ok_or(FastDecodeError)?;
    let out_start = target.len();
    let len = slice.len() / item_size;
    target.reserve(len);
    #[cfg(target_endian = "little")]
    unsafe {
        let dst = target.as_mut_ptr().add(out_start).cast::<u8>();
        ptr::copy_nonoverlapping(slice.as_ptr(), dst, slice.len());
        target.set_len(out_start + len);
    }
    #[cfg(not(target_endian = "little"))]
    {
        let _ = slice;
        panic!("unsupported non-little-endian BeamProc frame");
    }
    Ok((out_start, target.len()))
}

#[allow(clippy::too_many_arguments)]
fn validate_output_binary_ranges(
    token_id_bytes: &[u8],
    logprob_bytes: &[u8],
    rank_bytes: &[u8],
    candidate_offset: usize,
    candidate_rows: usize,
    candidate_cols: usize,
    rank_offset: usize,
    rank_len: usize,
) -> Result<(), FastDecodeError> {
    let candidate_len = candidate_rows
        .checked_mul(candidate_cols)
        .ok_or(FastDecodeError)?;
    let candidate_end = candidate_offset
        .checked_add(candidate_len)
        .ok_or(FastDecodeError)?;
    let rank_end = rank_offset.checked_add(rank_len).ok_or(FastDecodeError)?;
    validate_frame_range::<i32>(token_id_bytes, candidate_offset, candidate_end)?;
    validate_frame_range::<f32>(logprob_bytes, candidate_offset, candidate_end)?;
    validate_frame_range::<i32>(rank_bytes, rank_offset, rank_end)
}

fn validate_frame_range<T>(bytes: &[u8], start: usize, end: usize) -> Result<(), FastDecodeError> {
    let item_size = mem::size_of::<T>();
    if item_size == 0 || end < start {
        return Err(FastDecodeError);
    }
    let byte_start = start.checked_mul(item_size).ok_or(FastDecodeError)?;
    let byte_end = end.checked_mul(item_size).ok_or(FastDecodeError)?;
    bytes
        .get(byte_start..byte_end)
        .map(|_| ())
        .ok_or(FastDecodeError)
}

fn read_frame_range<T: Copy>(
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Vec<T>, FastDecodeError> {
    let item_size = mem::size_of::<T>();
    if item_size == 0 || end < start {
        return Err(FastDecodeError);
    }
    let byte_start = start.checked_mul(item_size).ok_or(FastDecodeError)?;
    let byte_end = end.checked_mul(item_size).ok_or(FastDecodeError)?;
    let slice = bytes.get(byte_start..byte_end).ok_or(FastDecodeError)?;
    let len = slice.len() / item_size;
    let mut values: Vec<T> = Vec::with_capacity(len);
    #[cfg(target_endian = "little")]
    unsafe {
        let dst = values.as_mut_ptr().cast::<u8>();
        ptr::copy_nonoverlapping(slice.as_ptr(), dst, slice.len());
        values.set_len(len);
    }
    #[cfg(not(target_endian = "little"))]
    {
        let _ = slice;
        panic!("unsupported non-little-endian BeamProc frame");
    }
    Ok(values)
}

fn typed_slice_as_bytes<T>(values: &[T]) -> &[u8] {
    if values.is_empty() {
        return &[];
    }
    #[cfg(target_endian = "little")]
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len().saturating_mul(mem::size_of::<T>()),
        )
    }
    #[cfg(not(target_endian = "little"))]
    {
        let _ = values;
        panic!("unsupported non-little-endian BeamProc frame");
    }
}

fn usize_slice_as_u64_le_bytes(values: &[usize]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(mem::size_of::<u64>()));
    for value in values {
        bytes.extend_from_slice(&(*value as u64).to_le_bytes());
    }
    bytes
}

fn hash_worker_index(request_id: &str, worker_count: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    request_id.hash(&mut hasher);
    (hasher.finish() as usize) % worker_count
}

fn start_zmq_monitor(
    zmq_context: &zmq::Context,
    socket: &zmq::Socket,
    peer: &'static str,
) -> Option<JoinHandle<()>> {
    let endpoint = format!("inproc://beam-proc-monitor-{}-{peer}", std::process::id());

    if let Err(err) = socket.monitor(&endpoint, zmq::SocketEvent::ALL as i32) {
        warn!(
            peer,
            monitor_endpoint = %endpoint,
            error = %err,
            "failed to enable ZMQ socket monitor"
        );
        return None;
    }

    let monitor_socket = match zmq_context.socket(zmq::PAIR) {
        Ok(socket) => socket,
        Err(err) => {
            warn!(
                peer,
                monitor_endpoint = %endpoint,
                error = %err,
                "failed to create ZMQ monitor socket"
            );
            return None;
        }
    };

    if let Err(err) = monitor_socket.connect(&endpoint) {
        warn!(
            peer,
            monitor_endpoint = %endpoint,
            error = %err,
            "failed to connect ZMQ monitor socket"
        );
        return None;
    }

    thread::Builder::new()
        .name(format!("zmq-monitor-{peer}"))
        .spawn(move || run_zmq_monitor(peer, monitor_socket))
        .map_err(|err| {
            warn!(
                peer,
                monitor_endpoint = %endpoint,
                error = %err,
                "failed to spawn ZMQ monitor thread"
            );
        })
        .ok()
}

fn run_zmq_monitor(peer: &'static str, monitor_socket: zmq::Socket) {
    loop {
        let event_msg = match monitor_socket.recv_msg(0) {
            Ok(msg) => msg,
            Err(err) => {
                debug!(peer, error = %err, "ZMQ monitor stopped");
                break;
            }
        };

        if event_msg.len() < 6 {
            warn!(
                peer,
                bytes = event_msg.len(),
                "ZMQ monitor received malformed event frame"
            );
            continue;
        }

        let event_raw = u16::from_ne_bytes([event_msg[0], event_msg[1]]);
        let event_value =
            u32::from_ne_bytes([event_msg[2], event_msg[3], event_msg[4], event_msg[5]]);
        let event = zmq::SocketEvent::from_raw(event_raw);

        let addr_msg = match monitor_socket.recv_msg(0) {
            Ok(msg) => msg,
            Err(err) => {
                warn!(
                    peer,
                    event = ?event,
                    event_value,
                    error = %err,
                    "ZMQ monitor failed to read address frame"
                );
                break;
            }
        };
        let address = addr_msg.as_str().unwrap_or("<non-utf8>");

        match event {
            zmq::SocketEvent::CONNECTED
            | zmq::SocketEvent::HANDSHAKE_SUCCEEDED
            | zmq::SocketEvent::DISCONNECTED
            | zmq::SocketEvent::MONITOR_STOPPED => info!(
                peer,
                event = ?event,
                event_value,
                address,
                "ZMQ monitor event"
            ),
            _ => debug!(
                peer,
                event = ?event,
                event_value,
                address,
                "ZMQ monitor event"
            ),
        }

        if event == zmq::SocketEvent::MONITOR_STOPPED {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct TestAbortPayload<'a> {
        request_ids: Vec<&'a str>,
        root_request_ids: Vec<&'a str>,
    }

    #[derive(Serialize)]
    struct TestEngineOutputBatch {
        version: u32,
        engine_index: usize,
        client_index: usize,
        correlation_base: u64,
        contexts: Vec<serde_json::Value>,
        outputs: Vec<serde_json::Value>,
    }

    #[test]
    fn binary_range_rejects_truncated_frame_without_mutating_target() {
        let bytes = typed_slice_as_bytes(&[10_i32, 20_i32]);
        let mut target = vec![1_i32];

        assert!(append_frame_range(&mut target, bytes, 0, 3).is_err());
        assert_eq!(target, vec![1]);
    }

    #[test]
    fn binary_ranges_reject_shape_overflow() {
        assert!(validate_output_binary_ranges(&[], &[], &[], 0, usize::MAX, 2, 0, 0).is_err());
    }

    #[test]
    fn engine_output_v6_routes_by_explicit_context_root() {
        let (worker_tx, worker_rx) = crossbeam_channel::bounded(1);
        let mut state = IoRxState::new(String::new(), String::new(), vec![worker_tx], 1, 1);
        let child_request_id = "customer-beam-value-beamproc-other-beamproc-1-0";
        let payload = rmp_serde::to_vec_named(&TestEngineOutputBatch {
            version: 6,
            engine_index: 0,
            client_index: 0,
            correlation_base: 7,
            contexts: vec![serde_json::json!({
                "root_request_id": "opaque-session-id",
            })],
            outputs: vec![serde_json::json!([
                child_request_id,
                0,
                0,
                null,
                null,
                null,
                0,
                0,
                0,
                0,
                0,
                0,
            ])],
        })
        .unwrap();

        assert!(
            state
                .try_handle_engine_output_batch_v3(&payload, &[], &[], &[], &[])
                .unwrap()
        );

        let (request_id, batch) = match worker_rx.try_recv().unwrap() {
            BeamWorkerMsg::EngineOutputBatch {
                request_id, batch, ..
            }
            | BeamWorkerMsg::EngineOutputBatchPooled {
                request_id, batch, ..
            } => (request_id, batch),
            message => panic!("expected engine output batch, got {message:?}"),
        };
        assert_eq!(request_id, "opaque-session-id");
        assert_eq!(batch.outputs[0].request_id, child_request_id);
    }

    #[test]
    fn abort_frame_is_routed_to_the_root_worker() {
        let (worker_tx, worker_rx) = crossbeam_channel::unbounded();
        let mut state = IoRxState::new(String::new(), String::new(), vec![worker_tx], 1, 1);
        let payload = rmp_serde::to_vec_named(&TestAbortPayload {
            request_ids: vec!["root-step-1-beam-0"],
            root_request_ids: vec!["root"],
        })
        .unwrap();

        let worker = thread::spawn(move || match worker_rx.recv().unwrap() {
            BeamWorkerMsg::Abort {
                root_request_id,
                ack,
            } => {
                assert_eq!(root_request_id, "root");
                ack.unwrap().send(()).unwrap();
            }
            message => panic!("expected abort message, got {message:?}"),
        });
        state.handle_enginecore_frames(vec![MSG_ABORT.as_bytes().to_vec(), payload]);
        worker.join().unwrap();
    }
}
