use crate::catalog::Catalog;
use crate::runtime::manager::{Manager, ManagerMessage};
use crossbeam_channel::Sender;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

#[derive(Debug, Clone)]
pub struct BeamEngineOutput {
    pub request_id: String,
    pub new_token_ids: Vec<i32>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<String>,
    pub num_cached_tokens: Option<usize>,
    pub candidate_token_start: usize,
    pub candidate_token_end: usize,
    pub candidate_logprob_start: usize,
    pub candidate_logprob_end: usize,
    pub candidate_rows: usize,
    pub candidate_cols: usize,
    pub sampled_rank_start: usize,
    pub sampled_rank_end: usize,
    pub beam_context: Option<BeamContext>,
}

#[derive(Debug)]
pub struct BeamEngineOutputBatch {
    pub outputs: Vec<BeamEngineOutput>,
    pub candidate_token_ids: Vec<i32>,
    pub candidate_logprobs: Vec<f32>,
    pub sampled_token_ranks: Vec<i32>,
}

impl BeamEngineOutputBatch {
    pub fn with_capacity(output_capacity: usize) -> Self {
        Self {
            outputs: Vec::with_capacity(output_capacity),
            candidate_token_ids: Vec::new(),
            candidate_logprobs: Vec::new(),
            sampled_token_ranks: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.outputs.clear();
        self.candidate_token_ids.clear();
        self.candidate_logprobs.clear();
        self.sampled_token_ranks.clear();
    }

    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    pub fn append_output(&mut self, mut output: BeamEngineOutput, source: &Self) {
        let candidate_token_start = self.candidate_token_ids.len();
        self.candidate_token_ids
            .extend_from_slice(source.candidate_tokens(&output));
        let candidate_token_end = self.candidate_token_ids.len();
        let candidate_logprob_start = self.candidate_logprobs.len();
        self.candidate_logprobs
            .extend_from_slice(source.candidate_logprobs(&output));
        let candidate_logprob_end = self.candidate_logprobs.len();
        let sampled_rank_start = self.sampled_token_ranks.len();
        self.sampled_token_ranks
            .extend_from_slice(source.sampled_token_ranks(&output));
        let sampled_rank_end = self.sampled_token_ranks.len();
        output.candidate_token_start = candidate_token_start;
        output.candidate_token_end = candidate_token_end;
        output.candidate_logprob_start = candidate_logprob_start;
        output.candidate_logprob_end = candidate_logprob_end;
        output.sampled_rank_start = sampled_rank_start;
        output.sampled_rank_end = sampled_rank_end;
        self.outputs.push(output);
    }

    pub fn candidate_row_count(&self, output: &BeamEngineOutput) -> usize {
        if output.candidate_cols == 0 {
            0
        } else {
            output.candidate_rows
        }
    }

    pub fn candidate_count(&self, output: &BeamEngineOutput) -> usize {
        output
            .candidate_token_end
            .saturating_sub(output.candidate_token_start)
    }

    pub fn candidate_tokens(&self, output: &BeamEngineOutput) -> &[i32] {
        self.candidate_token_ids
            .get(output.candidate_token_start..output.candidate_token_end)
            .unwrap_or(&[])
    }

    pub fn candidate_logprobs(&self, output: &BeamEngineOutput) -> &[f32] {
        self.candidate_logprobs
            .get(output.candidate_logprob_start..output.candidate_logprob_end)
            .unwrap_or(&[])
    }

    pub fn sampled_token_ranks(&self, output: &BeamEngineOutput) -> &[i32] {
        self.sampled_token_ranks
            .get(output.sampled_rank_start..output.sampled_rank_end)
            .unwrap_or(&[])
    }

    fn candidate_token_row(&self, output: &BeamEngineOutput, row: usize) -> &[i32] {
        candidate_row(self.candidate_tokens(output), row, output.candidate_cols)
    }

    fn candidate_logprob_row(&self, output: &BeamEngineOutput, row: usize) -> &[f32] {
        candidate_row(self.candidate_logprobs(output), row, output.candidate_cols)
    }

    fn candidate_rows<'a>(
        &'a self,
        output: &'a BeamEngineOutput,
    ) -> impl Iterator<Item = (&'a [i32], &'a [f32])> {
        (0..self.candidate_row_count(output)).map(|row| {
            (
                self.candidate_token_row(output, row),
                self.candidate_logprob_row(output, row),
            )
        })
    }
}

#[derive(Debug, Clone)]
pub struct BeamContext {
    pub beam_width: Option<usize>,
    pub max_tokens: Option<usize>,
    pub sampling_max_tokens: Option<usize>,
    pub global_max_tokens: Option<usize>,
    pub total_decode_steps: Option<usize>,
    pub step_index: Option<usize>,
    pub is_final_step: bool,
    pub ignore_eos: bool,
    pub include_stop_token: bool,
    pub eos_token_id: Option<i32>,
    pub end_token_id: Option<i32>,
    pub catalog_strategy: String,
    pub return_logprobs: bool,
    pub parent_token_ids: Vec<i32>,
    pub use_shadow_parent_tokens: bool,
    pub prompt_token_count: Option<usize>,
    pub client_index: usize,
    pub priority: i64,
}

// CLOCK_MONOTONIC shares the OS clock used by CPython perf_counter on Linux.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct BeamTiming {
    pub clock: &'static str,
    pub prefill_end: f64,
    pub prefill_only: bool,
}

fn beam_timing(prefill_only: bool) -> Option<BeamTiming> {
    #[cfg(target_os = "linux")]
    {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) } == 0 {
            return Some(BeamTiming {
                clock: "CLOCK_MONOTONIC",
                prefill_end: ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9,
                prefill_only,
            });
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct BeamWorkerPlan {
    pub timing: Option<BeamTiming>,
    pub num_processed_tokens: u64,
    pub engine_index: usize,
    pub client_index: usize,
    pub request_id: String,
    pub root_request_id: String,
    pub correlation_id: String,
    pub action: String,
    pub beam_width: usize,
    pub scored_parent_ids: Vec<String>,
    pub parent_ids: Vec<String>,
    pub child_ids: Vec<String>,
    pub token_ids: Vec<i32>,
    pub beam_tokens: Vec<Vec<i32>>,
    pub scores: Vec<f32>,
    pub flat_indices: Vec<usize>,
    pub abort_ids: Vec<String>,
    pub completed_count: usize,
    pub finished: bool,
    /// EngineCore already cleans up roots whose scheduled final step ran.
    /// A finished plan with this false needs an explicit early-finish notice.
    pub scheduled_final: bool,
    pub final_token_ids_include_prompt: bool,
    pub final_token_ids: Vec<Vec<i32>>,
    pub final_scores: Vec<f32>,
    pub final_finish_reasons: Vec<String>,
    pub final_stop_token_ids: Vec<Option<i32>>,
    pub final_logprob_token_ids: Vec<i32>,
    pub final_logprob_values: Vec<f32>,
    pub final_logprob_step_offsets: Vec<usize>,
    pub final_logprob_beam_offsets: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct BeamLogprobStep {
    pub token_ids: Vec<i32>,
    pub logprobs: Vec<f32>,
}

type SharedBeamLogprobSteps = Vec<Arc<BeamLogprobStep>>;

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct BeamCandidate {
    pub parent_beam_id: u64,
    pub token_id: i32,
    pub logprob: f32,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct CompletedBeam {
    pub beam_id: u64,
    pub tokens: Vec<i32>,
    pub cum_logprob: f32,
    pub finish_reason: BeamFinishReason,
    pub stop_token_id: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeamFinishReason {
    Stop,
    Length,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct ForkPlan {
    pub request_id: String,
    pub step: usize,
    pub parent_engine_request_ids: Vec<String>,
    pub child_engine_request_ids: Vec<String>,
    pub token_ids: Vec<i32>,
    pub abort_engine_request_ids: Vec<String>,
    pub completed: Vec<CompletedBeam>,
    pub finished: bool,
}

#[derive(Debug)]
pub enum BeamWorkerMsg {
    EngineOutputBatch {
        engine_index: usize,
        client_index: usize,
        request_id: String,
        correlation_id: String,
        batch: BeamEngineOutputBatch,
    },
    EngineOutputBatchPooled {
        engine_index: usize,
        client_index: usize,
        request_id: String,
        correlation_id: String,
        batch: BeamEngineOutputBatch,
        pool_return_tx: Sender<BeamEngineOutputBatch>,
    },
    BeamStateUpdate {
        root_request_id: String,
        beam_width: usize,
        active_parent_ids: Vec<String>,
        active_parent_cum_logprobs: Vec<f32>,
        active_parent_valid_token_ids: Vec<Vec<i32>>,
    },
    Abort {
        root_request_id: String,
        ack: Option<Sender<()>>,
    },
    Shutdown,
}

impl ManagerMessage for BeamWorkerMsg {
    fn shutdown() -> Self {
        Self::Shutdown
    }

    fn is_shutdown(&self) -> bool {
        matches!(self, Self::Shutdown)
    }
}

pub struct BeamWorker {
    worker_id: usize,
    manager: Manager<BeamWorkerMsg>,
}

impl BeamWorker {
    pub fn init(
        worker_id: usize,
        plan_tx: Sender<BeamWorkerPlan>,
        catalog: Option<Arc<Catalog>>,
        queue_capacity: usize,
    ) -> Self {
        let mut state = BeamWorkerState::new(worker_id, plan_tx, catalog);
        let manager = Manager::init_with_capacity(move |msg| state.handle(msg), queue_capacity);

        Self { worker_id, manager }
    }

    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    pub fn message_queue(&self) -> Sender<BeamWorkerMsg> {
        self.manager.message_queue()
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        self.manager.run()
    }

    pub fn is_finished(&self) -> bool {
        self.manager.is_finished()
    }

    pub fn shutdown(&self) -> anyhow::Result<()> {
        self.manager
            .shutdown()
            .map_err(|_| anyhow::anyhow!("beam worker shutdown channel disconnected"))
    }

    pub fn join(&mut self) -> thread::Result<()> {
        self.manager.join()
    }
}

#[cfg(test)]
#[derive(Debug)]
struct RequestState {
    request_id: String,
    beam_width: usize,
    max_tokens: usize,
    ignore_eos: bool,
    include_stop_token: bool,
    eos_token_id: Option<i32>,
    end_token_id: Option<i32>,
    current_step: usize,
    next_beam_id: u64,
    active: Vec<ActiveBeam>,
    completed: Vec<CompletedBeam>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct ActiveBeam {
    beam_id: u64,
    engine_request_id: String,
    tokens: Vec<i32>,
    cum_logprob: f32,
}

struct BeamWorkerState {
    worker_id: usize,
    shadow_families: HashMap<String, ShadowBeamFamily>,
    observed_completed: HashMap<String, Vec<ObservedFinalBeam>>,
    pending_engine_outputs: HashMap<String, PendingEngineOutputs>,
    cancelled_epochs: HashMap<String, Instant>,
    cancelled_epoch_order: VecDeque<(Instant, String)>,
    plan_tx: Sender<BeamWorkerPlan>,
    catalog: Option<Arc<Catalog>>,
    candidate_scratch: Vec<ObservedCandidate>,
}

const CANCELLED_EPOCH_TTL: Duration = Duration::from_secs(10 * 60);
const MAX_CANCELLED_EPOCHS: usize = 4096;

#[derive(Clone, Debug)]
struct ObservedCandidate {
    parent_index: usize,
    token_id: i32,
    continuation_score: f32,
    completed_score: f32,
    flat_index: usize,
    flags: u8,
}

const CANDIDATE_EXPLICIT_REJECTED: u8 = 1 << 0;
const CANDIDATE_CATALOG_REJECTED: u8 = 1 << 1;
const CANDIDATE_EOS_REJECTED: u8 = 1 << 2;
const CANDIDATE_INVALID: u8 = 1 << 3;
const CANDIDATE_INCLUDE_STOP_TOKEN: u8 = 1 << 4;

impl ObservedCandidate {
    #[inline]
    fn has_flag(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }

    #[inline]
    fn stop_token_id(&self) -> Option<i32> {
        // EOS_REJECTED is set only when matching_stop_token returned this
        // candidate's token and ignore_eos was false. No separate ID is needed.
        self.has_flag(CANDIDATE_EOS_REJECTED)
            .then_some(self.token_id)
    }
}

impl BeamWorkerState {
    fn new(
        worker_id: usize,
        plan_tx: Sender<BeamWorkerPlan>,
        catalog: Option<Arc<Catalog>>,
    ) -> Self {
        Self {
            worker_id,
            shadow_families: HashMap::new(),
            observed_completed: HashMap::new(),
            pending_engine_outputs: HashMap::new(),
            cancelled_epochs: HashMap::new(),
            cancelled_epoch_order: VecDeque::new(),
            plan_tx,
            catalog,
            candidate_scratch: Vec::new(),
        }
    }

    fn handle(&mut self, msg: BeamWorkerMsg) {
        match msg {
            BeamWorkerMsg::EngineOutputBatch {
                engine_index,
                client_index,
                request_id,
                correlation_id,
                batch,
            } => {
                let _ = self.handle_engine_output_batch(
                    engine_index,
                    client_index,
                    request_id,
                    correlation_id,
                    batch,
                );
            }
            BeamWorkerMsg::EngineOutputBatchPooled {
                engine_index,
                client_index,
                request_id,
                correlation_id,
                batch,
                pool_return_tx,
            } => {
                let batch = self.handle_engine_output_batch(
                    engine_index,
                    client_index,
                    request_id,
                    correlation_id,
                    batch,
                );
                if let Err(err) = pool_return_tx.send(batch) {
                    warn!(
                        worker_id = self.worker_id,
                        error = %err,
                        "beam worker failed to return pooled output buffer"
                    );
                }
            }
            BeamWorkerMsg::BeamStateUpdate {
                root_request_id,
                beam_width,
                active_parent_ids,
                active_parent_cum_logprobs,
                active_parent_valid_token_ids,
            } => {
                if self.is_cancelled_epoch(&root_request_id) {
                    debug!(
                        worker_id = self.worker_id,
                        root_request_id = %root_request_id,
                        "beam worker rejected state update for cancelled request epoch"
                    );
                    return;
                }
                let family = self
                    .shadow_families
                    .entry(root_request_id.clone())
                    .or_insert_with(|| ShadowBeamFamily::new(root_request_id.clone()));
                self.pending_engine_outputs.remove(&root_request_id);
                family.update(
                    beam_width,
                    active_parent_ids,
                    active_parent_cum_logprobs,
                    active_parent_valid_token_ids,
                );
                debug!(
                    worker_id = self.worker_id,
                    root_request_id = %root_request_id,
                    beam_width = family.beam_width,
                    active = family.parent_cum_logprobs.len(),
                    valid_masks = family.parent_valid_token_ids.len(),
                    "beam worker updated shadow beam state"
                );
            }
            BeamWorkerMsg::Abort {
                root_request_id,
                ack,
            } => {
                self.tombstone_cancelled_epoch(root_request_id.clone());
                let removed_family = self.shadow_families.remove(&root_request_id).is_some();
                let removed_completed = self.observed_completed.remove(&root_request_id).is_some();
                let removed_pending = self
                    .pending_engine_outputs
                    .remove(&root_request_id)
                    .is_some();
                debug!(
                    worker_id = self.worker_id,
                    root_request_id = %root_request_id,
                    removed_family,
                    removed_completed,
                    removed_pending,
                    "beam worker cleared aborted request state"
                );
                if let Some(ack) = ack {
                    let _ = ack.send(());
                }
            }
            BeamWorkerMsg::Shutdown => {}
        }
    }

    fn handle_engine_output_batch(
        &mut self,
        engine_index: usize,
        client_index: usize,
        request_id: String,
        correlation_id: String,
        mut batch: BeamEngineOutputBatch,
    ) -> BeamEngineOutputBatch {
        if self.is_cancelled_epoch(&request_id) {
            debug!(
                worker_id = self.worker_id,
                request_id = %request_id,
                correlation_id = %correlation_id,
                "beam worker rejected late output for cancelled request epoch"
            );
            batch.clear();
            return batch;
        }
        let token_count = batch
            .outputs
            .iter()
            .map(|output| output.new_token_ids.len())
            .sum::<usize>();
        let finished_count = batch
            .outputs
            .iter()
            .filter(|output| output.finish_reason.is_some())
            .count();
        let cached_tokens = batch
            .outputs
            .iter()
            .filter_map(|output| output.num_cached_tokens)
            .max();
        let stop_reason_count = batch
            .outputs
            .iter()
            .filter(|output| output.stop_reason.is_some())
            .count();
        let candidate_positions = batch
            .outputs
            .iter()
            .map(|output| batch.candidate_row_count(output))
            .sum::<usize>();
        let candidate_count = batch
            .outputs
            .iter()
            .map(|output| batch.candidate_count(output))
            .sum::<usize>();
        let candidate_logprob_count = batch
            .outputs
            .iter()
            .map(|output| batch.candidate_logprobs(output).len())
            .sum::<usize>();
        let rank_count = batch
            .outputs
            .iter()
            .map(|output| batch.sampled_token_ranks(output).len())
            .sum::<usize>();
        let context_count = batch
            .outputs
            .iter()
            .filter(|output| output.beam_context.is_some())
            .count();
        let sample_context = batch.outputs.iter().find_map(|output| {
            output.beam_context.as_ref().map(|context| {
                (
                    context.max_tokens,
                    context.sampling_max_tokens,
                    context.global_max_tokens,
                    context.total_decode_steps,
                    context.step_index,
                    context.is_final_step,
                    context.ignore_eos,
                    context.eos_token_id,
                    context.parent_token_ids.len(),
                )
            })
        });
        debug!(
            worker_id = self.worker_id,
            engine_index,
            client_index,
            request_id = %request_id,
            correlation_id = %correlation_id,
            outputs = batch.output_count(),
            token_count,
            finished_count,
            stop_reason_count,
            cached_tokens,
            candidate_positions,
            candidate_count,
            candidate_logprob_count,
            rank_count,
            context_count,
            sample_context = ?sample_context,
            "beam worker received engine output batch"
        );
        if let Some(plan) = self.buffer_or_build_full_observed_plan(
            engine_index,
            client_index,
            request_id.clone(),
            correlation_id.clone(),
            &mut batch,
        ) {
            let finished_root_request_id = plan.finished.then(|| plan.root_request_id.clone());
            self.update_shadow_state_from_plan(&plan);
            match self.plan_tx.send(plan) {
                Ok(()) => {}
                Err(err) => {
                    warn!(
                        worker_id = self.worker_id,
                        request_id = %request_id,
                        correlation_id = %correlation_id,
                        error = %err,
                        "beam worker failed to send plan"
                    );
                }
            }
            if let Some(root_request_id) = finished_root_request_id {
                self.shadow_families.remove(&root_request_id);
                self.observed_completed.remove(&root_request_id);
                self.pending_engine_outputs.remove(&root_request_id);
                debug!(
                    worker_id = self.worker_id,
                    root_request_id = %root_request_id,
                    "beam worker cleaned final request state"
                );
            }
        }
        batch.clear();
        batch
    }

    fn tombstone_cancelled_epoch(&mut self, root_request_id: String) {
        let now = Instant::now();
        self.cancelled_epochs.insert(root_request_id.clone(), now);
        self.cancelled_epoch_order.push_back((now, root_request_id));
        self.prune_cancelled_epochs(now);
    }

    fn is_cancelled_epoch(&mut self, root_request_id: &str) -> bool {
        self.prune_cancelled_epochs(Instant::now());
        self.cancelled_epochs.contains_key(root_request_id)
    }

    fn prune_cancelled_epochs(&mut self, now: Instant) {
        while let Some((created_at, root_request_id)) = self.cancelled_epoch_order.front() {
            if self.cancelled_epochs.len() <= MAX_CANCELLED_EPOCHS
                && now.duration_since(*created_at) < CANCELLED_EPOCH_TTL
            {
                break;
            }
            let created_at = *created_at;
            let root_request_id = root_request_id.clone();
            self.cancelled_epoch_order.pop_front();
            if self.cancelled_epochs.get(&root_request_id) == Some(&created_at) {
                self.cancelled_epochs.remove(&root_request_id);
            }
        }
    }

    fn update_shadow_state_from_plan(&mut self, plan: &BeamWorkerPlan) {
        if plan.finished {
            return;
        }
        if plan.child_ids.is_empty() {
            return;
        }

        self.pending_engine_outputs.remove(&plan.root_request_id);
        let family = self
            .shadow_families
            .entry(plan.root_request_id.clone())
            .or_insert_with(|| ShadowBeamFamily::new(plan.root_request_id.clone()));
        family.update(
            plan.beam_width,
            plan.child_ids.clone(),
            plan.scores.clone(),
            Vec::new(),
        );
    }

    fn buffer_or_build_full_observed_plan(
        &mut self,
        engine_index: usize,
        client_index: usize,
        request_id: String,
        correlation_id: String,
        batch: &mut BeamEngineOutputBatch,
    ) -> Option<BeamWorkerPlan> {
        let Some(family) = self.shadow_families.get(&request_id) else {
            return Some(self.build_full_observed_plan(
                engine_index,
                client_index,
                request_id,
                correlation_id,
                batch,
            ));
        };
        if family.active_parent_ids.is_empty() {
            return Some(self.build_full_observed_plan(
                engine_index,
                client_index,
                request_id,
                correlation_id,
                batch,
            ));
        }
        if let Some(output) = batch
            .outputs
            .iter()
            .find(|output| !output_has_matching_shadow_history(family, output))
        {
            warn!(
                worker_id = self.worker_id,
                root_request_id = %request_id,
                parent_id = %output.request_id,
                context_step_index = ?output
                    .beam_context
                    .as_ref()
                    .and_then(|context| context.step_index),
                expected_step_index = ?beamproc_child_step(&output.request_id),
                has_shadow_tokens = family.parent_tokens(&output.request_id).is_some(),
                "beam worker rejected grouped output without matching shadow history"
            );
            batch.clear();
            return None;
        }

        // Grouped EngineCore outputs normally arrive complete and in the
        // active-parent order. Keep their pooled candidate buffers in place
        // instead of copying through pending and ordered batches.
        if !self.pending_engine_outputs.contains_key(&request_id)
            && batch.outputs.len() == family.active_parent_ids.len()
            && batch
                .outputs
                .iter()
                .zip(&family.active_parent_ids)
                .all(|(output, parent_id)| output.request_id == *parent_id)
        {
            return Some(self.build_full_observed_plan(
                engine_index,
                client_index,
                request_id,
                correlation_id,
                batch,
            ));
        }

        let expected_parent_ids = family.active_parent_ids.clone();
        let expected_count = expected_parent_ids.len();
        let pending = self
            .pending_engine_outputs
            .entry(request_id.clone())
            .or_insert_with(|| {
                PendingEngineOutputs::new(
                    engine_index,
                    client_index,
                    request_id.clone(),
                    correlation_id.clone(),
                )
            });
        pending.engine_index = engine_index;
        pending.client_index = client_index;
        pending.correlation_id = correlation_id.clone();
        let drained = std::mem::take(&mut batch.outputs);
        for output in drained {
            pending.batch.append_output(output, batch);
            let inserted_index = pending.batch.outputs.len().saturating_sub(1);
            let parent_id = pending.batch.outputs[inserted_index].request_id.clone();
            pending.outputs_by_parent.insert(parent_id, inserted_index);
        }
        batch.clear();

        let received_count = expected_parent_ids
            .iter()
            .filter(|parent_id| pending.outputs_by_parent.contains_key(*parent_id))
            .count();
        if received_count < expected_count {
            info!(
                worker_id = self.worker_id,
                root_request_id = %request_id,
                expected_parents = expected_count,
                buffered_parents = received_count,
                latest_batch_correlation_id = %correlation_id,
                "beam worker buffered partial engine outputs"
            );
            return None;
        }

        let pending = self
            .pending_engine_outputs
            .remove(&request_id)
            .expect("pending output buffer exists after completion check");
        let mut ordered_batch = BeamEngineOutputBatch::with_capacity(expected_count);
        for parent_id in expected_parent_ids.iter() {
            if let Some(output_index) = pending.outputs_by_parent.get(parent_id)
                && let Some(output) = pending.batch.outputs.get(*output_index).cloned()
            {
                ordered_batch.append_output(output, &pending.batch);
            }
        }
        debug!(
            worker_id = self.worker_id,
            root_request_id = %request_id,
            expected_parents = expected_count,
            buffered_parents = ordered_batch.output_count(),
            "beam worker completed buffered engine outputs"
        );

        Some(self.build_full_observed_plan(
            pending.engine_index,
            pending.client_index,
            pending.request_id,
            pending.correlation_id,
            &ordered_batch,
        ))
    }

    fn record_output_parent_tokens(
        &mut self,
        root_request_id: &str,
        batch: &BeamEngineOutputBatch,
    ) {
        let family = self
            .shadow_families
            .entry(root_request_id.to_string())
            .or_insert_with(|| ShadowBeamFamily::new(root_request_id.to_string()));
        for output in &batch.outputs {
            let Some(context) = output.beam_context.as_ref() else {
                continue;
            };
            if !is_beamproc_child_request(&output.request_id) {
                family.record_root_context(context);
            }
            if context.parent_token_ids.is_empty() {
                continue;
            }
            family
                .record_parent_tokens(output.request_id.clone(), context.parent_token_ids.clone());
        }
    }

    fn build_full_observed_plan(
        &mut self,
        engine_index: usize,
        client_index: usize,
        request_id: String,
        correlation_id: String,
        batch: &BeamEngineOutputBatch,
    ) -> BeamWorkerPlan {
        let outputs = batch.outputs.as_slice();
        let root_request_id = request_id;
        self.record_output_parent_tokens(&root_request_id, batch);
        let family = self.shadow_families.get(&root_request_id);
        let beam_width = family
            .map(|family| family.beam_width)
            .filter(|beam_width| *beam_width > 0)
            .unwrap_or_else(|| infer_beam_width(batch));
        let scored_parent_ids = outputs
            .iter()
            .map(|output| output.request_id.clone())
            .collect::<Vec<_>>();
        let plan_prompt_token_count = outputs
            .first()
            .and_then(|output| {
                output
                    .beam_context
                    .as_ref()
                    .and_then(|context| context.prompt_token_count)
            })
            .or_else(|| family.and_then(|family| family.prompt_token_count));
        let return_logprobs = family
            .and_then(|family| family.return_logprobs)
            .or_else(|| {
                outputs.iter().find_map(|output| {
                    output
                        .beam_context
                        .as_ref()
                        .map(|context| context.return_logprobs)
                })
            })
            .unwrap_or(true);
        let mut completed_candidates = Vec::new();
        let mut current_logprob_steps = if return_logprobs {
            (0..outputs.len())
                .map(|_| {
                    Arc::new(BeamLogprobStep {
                        token_ids: Vec::with_capacity(beam_width),
                        logprobs: Vec::with_capacity(beam_width),
                    })
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let candidate_capacity = outputs.len().saturating_mul(beam_width);
        // Each worker handles one message at a time. Retain its allocation
        // across steps and requests; outgoing plans own their selected data.
        let candidates = &mut self.candidate_scratch;
        candidates.clear();
        candidates.reserve(candidate_capacity);
        for (parent_index, output) in outputs.iter().enumerate() {
            let parent_cum_logprob = family
                .and_then(|family| family.parent_cum_logprobs.get(&output.request_id))
                .copied()
                .unwrap_or(0.0);
            let explicit_valid_token_ids =
                family.and_then(|family| family.parent_valid_token_ids.get(&output.request_id));
            let beam_context = output.beam_context.as_ref();
            let parent_token_ids = beam_context
                .and_then(|context| {
                    (!context.parent_token_ids.is_empty())
                        .then_some(context.parent_token_ids.as_slice())
                })
                .or_else(|| family.and_then(|family| family.parent_tokens(&output.request_id)))
                .unwrap_or(&[]);
            let prompt_token_count = beam_context
                .and_then(|context| context.prompt_token_count)
                .or_else(|| family.and_then(|family| family.prompt_token_count));
            let use_family_static = is_beamproc_child_request(&output.request_id);
            let ignore_eos = if use_family_static {
                family
                    .and_then(|family| family.ignore_eos)
                    .unwrap_or_else(|| beam_context.is_some_and(|context| context.ignore_eos))
            } else {
                beam_context.is_some_and(|context| context.ignore_eos)
            };
            let include_stop_token = if use_family_static {
                family
                    .and_then(|family| family.include_stop_token)
                    .unwrap_or_else(|| {
                        beam_context.is_some_and(|context| context.include_stop_token)
                    })
            } else {
                beam_context.is_some_and(|context| context.include_stop_token)
            };
            let eos_token_id = if use_family_static {
                family
                    .and_then(|family| family.eos_token_id.flatten())
                    .or_else(|| beam_context.and_then(|context| context.eos_token_id))
            } else {
                beam_context.and_then(|context| context.eos_token_id)
            };
            let end_token_id = if use_family_static {
                family
                    .and_then(|family| family.end_token_id.flatten())
                    .or_else(|| beam_context.and_then(|context| context.end_token_id))
            } else {
                beam_context.and_then(|context| context.end_token_id)
            };
            let catalog_strategy = if use_family_static {
                family
                    .and_then(|family| family.catalog_strategy.as_deref())
                    .unwrap_or_else(|| {
                        beam_context
                            .map(|context| context.catalog_strategy.as_str())
                            .unwrap_or("fill")
                    })
            } else {
                beam_context
                    .map(|context| context.catalog_strategy.as_str())
                    .unwrap_or("fill")
            };
            let catalog_valid_token_ids = if explicit_valid_token_ids.is_none() {
                self.catalog.as_deref().map(|catalog| {
                    catalog
                        .prefix(parent_token_ids, prompt_token_count)
                        .valid_next_sorted()
                })
            } else {
                None
            };
            let valid_token_ids = explicit_valid_token_ids
                .map(Vec::as_slice)
                .or(catalog_valid_token_ids);
            let explicit_filter = explicit_valid_token_ids.is_some();
            let fill_catalog = catalog_strategy == "fill";
            let mut logprob_step = current_logprob_steps
                .get_mut(parent_index)
                .and_then(Arc::get_mut);
            for (tokens, logprobs) in batch.candidate_rows(output) {
                let mut fill_used = 0usize;
                for (candidate_index, (mut token_id, mut logprob)) in
                    sampled_candidates(tokens, logprobs, beam_width).enumerate()
                {
                    let (
                        filtered_token_id,
                        filtered_logprob,
                        rejected_by_catalog,
                        catalog_rejected,
                    ) = apply_catalog_strategy(
                        token_id,
                        logprob,
                        valid_token_ids,
                        fill_catalog,
                        &mut fill_used,
                    );
                    token_id = filtered_token_id;
                    logprob = filtered_logprob;
                    if let Some(step) = logprob_step.as_mut() {
                        step.token_ids.push(token_id);
                        step.logprobs.push(logprob);
                    }
                    let explicit_rejected = explicit_filter && rejected_by_catalog;
                    let stop_token_id = matching_stop_token(token_id, eos_token_id, end_token_id);
                    let eos_rejected = stop_token_id.is_some() && !ignore_eos;
                    let is_invalid = catalog_rejected || eos_rejected;
                    let completed_logprob = if catalog_rejected {
                        f32::NEG_INFINITY
                    } else {
                        logprob
                    };
                    let continuation_logprob = if is_invalid {
                        f32::NEG_INFINITY
                    } else {
                        logprob
                    };
                    let flags = (u8::from(explicit_rejected) * CANDIDATE_EXPLICIT_REJECTED)
                        | (u8::from(catalog_rejected) * CANDIDATE_CATALOG_REJECTED)
                        | (u8::from(eos_rejected) * CANDIDATE_EOS_REJECTED)
                        | (u8::from(is_invalid) * CANDIDATE_INVALID)
                        | (u8::from(include_stop_token) * CANDIDATE_INCLUDE_STOP_TOKEN);
                    candidates.push(ObservedCandidate {
                        parent_index,
                        token_id,
                        continuation_score: parent_cum_logprob + continuation_logprob,
                        completed_score: parent_cum_logprob + completed_logprob,
                        flat_index: parent_index * beam_width + candidate_index,
                        flags,
                    });
                }
            }
        }
        // Match Python's candidate count before EOS/finite-score filtering and
        // beam selection. Include masked candidates, not just selected children.
        let processed_candidates = candidates.len() as u64;
        let diagnostics_enabled =
            tracing::enabled!(tracing::Level::WARN) || tracing::enabled!(tracing::Level::DEBUG);
        let invalid_candidate_count = if diagnostics_enabled {
            candidates
                .iter()
                .filter(|candidate| candidate.has_flag(CANDIDATE_INVALID))
                .count()
        } else {
            0
        };
        for candidate in candidates.iter() {
            if candidate.has_flag(CANDIDATE_EOS_REJECTED)
                && !candidate.has_flag(CANDIDATE_CATALOG_REJECTED)
            {
                let mut completed_token_ids =
                    materialize_candidate_tokens(batch, family, candidate);
                if !candidate.has_flag(CANDIDATE_INCLUDE_STOP_TOKEN) {
                    completed_token_ids.pop();
                }
                completed_candidates.push(ObservedFinalBeam {
                    token_ids: completed_token_ids,
                    score: candidate.completed_score,
                    finish_reason: BeamFinishReason::Stop,
                    stop_token_id: candidate.stop_token_id(),
                    order: candidate.flat_index,
                    logprob_steps: if return_logprobs {
                        materialize_candidate_logprob_steps(
                            batch,
                            family,
                            candidate,
                            &current_logprob_steps,
                        )
                    } else {
                        Vec::new()
                    },
                });
            }
        }
        if !completed_candidates.is_empty() {
            self.observed_completed
                .entry(root_request_id.clone())
                .or_default()
                .extend(completed_candidates);
        }
        let candidate_count_before_retain = candidates.len();
        candidates.retain(|candidate| !candidate.has_flag(CANDIDATE_INVALID));
        let candidate_count_after_retain = candidates.len();
        let reached_final_step = outputs.iter().any(|output| {
            output
                .beam_context
                .as_ref()
                .is_some_and(|context| context.is_final_step)
        });
        let completed_count = outputs
            .iter()
            .filter(|output| output.finish_reason.is_some())
            .count();
        // A catalog leaf or a row containing only stop tokens has no active
        // continuation. Finish immediately even when the model-level sampled
        // token was a custom end token rather than the tokenizer EOS.
        let finished = reached_final_step || candidates.is_empty();
        rank_top_candidates(candidates, beam_width);
        let mut final_context_count = 0usize;
        let mut max_step_index: Option<usize> = None;
        let mut total_decode_steps: Option<usize> = None;
        for output in outputs {
            let Some(context) = output.beam_context.as_ref() else {
                continue;
            };
            if diagnostics_enabled && context.is_final_step {
                final_context_count += 1;
            }
            if let Some(step_index) = context.step_index {
                max_step_index = Some(max_step_index.map_or(step_index, |max| max.max(step_index)));
            }
            if diagnostics_enabled && total_decode_steps.is_none() {
                total_decode_steps = context.total_decode_steps;
            }
        }
        let child_step_index = max_step_index.map(|step_index| step_index + 1).unwrap_or(0);
        let (parent_ids, child_ids, token_ids, scores, flat_indices) = if finished {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
        } else {
            (
                candidates
                    .iter()
                    .map(|candidate| {
                        outputs
                            .get(candidate.parent_index)
                            .map(|output| output.request_id.clone())
                            .unwrap_or_default()
                    })
                    .collect::<Vec<_>>(),
                candidates
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        format!("{root_request_id}-beamproc-{child_step_index}-{index}")
                    })
                    .collect::<Vec<_>>(),
                candidates
                    .iter()
                    .map(|candidate| candidate.token_id)
                    .collect::<Vec<_>>(),
                candidates
                    .iter()
                    .map(|candidate| candidate.continuation_score)
                    .collect::<Vec<_>>(),
                candidates
                    .iter()
                    .map(|candidate| candidate.flat_index)
                    .collect::<Vec<_>>(),
            )
        };
        let selected_explicit_rejected = if diagnostics_enabled {
            candidates
                .iter()
                .filter(|candidate| candidate.has_flag(CANDIDATE_EXPLICIT_REJECTED))
                .count()
        } else {
            0
        };
        let selected_catalog_rejected = if diagnostics_enabled {
            candidates
                .iter()
                .filter(|candidate| candidate.has_flag(CANDIDATE_CATALOG_REJECTED))
                .count()
        } else {
            0
        };
        let selected_eos_rejected = if diagnostics_enabled {
            candidates
                .iter()
                .filter(|candidate| candidate.has_flag(CANDIDATE_EOS_REJECTED))
                .count()
        } else {
            0
        };
        let selected_invalid = if diagnostics_enabled {
            candidates
                .iter()
                .filter(|candidate| candidate.has_flag(CANDIDATE_INVALID))
                .count()
        } else {
            0
        };
        let mut final_beams = Vec::new();
        if finished {
            final_beams.extend(
                self.observed_completed
                    .get(&root_request_id)
                    .into_iter()
                    .flat_map(|completed| completed.iter().cloned()),
            );
            final_beams.extend(candidates.iter().map(|candidate| ObservedFinalBeam {
                token_ids: materialize_candidate_tokens(batch, family, candidate),
                score: candidate.continuation_score,
                finish_reason: BeamFinishReason::Length,
                stop_token_id: None,
                order: candidate.flat_index,
                logprob_steps: if return_logprobs {
                    materialize_candidate_logprob_steps(
                        batch,
                        family,
                        candidate,
                        &current_logprob_steps,
                    )
                } else {
                    Vec::new()
                },
            }));
            final_beams.sort_by(|left, right| {
                right
                    .score
                    .total_cmp(&left.score)
                    .then_with(|| left.order.cmp(&right.order))
            });
            final_beams.truncate(beam_width);
        }
        let can_strip_prompt = finished
            && plan_prompt_token_count.is_some_and(|prompt_token_count| {
                final_beams
                    .iter()
                    .all(|beam| beam.token_ids.len() >= prompt_token_count)
            });
        let final_token_ids_include_prompt = !can_strip_prompt;
        let final_token_ids = if finished {
            final_beams
                .iter()
                .map(|beam| {
                    if can_strip_prompt {
                        beam.token_ids[plan_prompt_token_count.unwrap_or_default()..].to_vec()
                    } else {
                        beam.token_ids.clone()
                    }
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let final_scores = if finished {
            final_beams
                .iter()
                .map(|beam| beam.score)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let final_finish_reasons = if finished {
            final_beams
                .iter()
                .map(|beam| beam.finish_reason.as_str().to_string())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let final_stop_token_ids = if finished {
            final_beams
                .iter()
                .map(|beam| beam.stop_token_id)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let (
            final_logprob_token_ids,
            final_logprob_values,
            final_logprob_step_offsets,
            final_logprob_beam_offsets,
        ) = if finished && return_logprobs {
            let step_count = final_beams
                .iter()
                .map(|beam| beam.logprob_steps.len())
                .sum::<usize>();
            let value_capacity = step_count.saturating_mul(beam_width);
            let mut token_ids = Vec::with_capacity(value_capacity);
            let mut values = Vec::with_capacity(value_capacity);
            let mut step_offsets = Vec::with_capacity(step_count + 1);
            let mut beam_offsets = Vec::with_capacity(final_beams.len() + 1);
            step_offsets.push(0);
            beam_offsets.push(0);
            for beam in &final_beams {
                for step in &beam.logprob_steps {
                    token_ids.extend_from_slice(&step.token_ids);
                    values.extend_from_slice(&step.logprobs);
                    step_offsets.push(token_ids.len());
                }
                beam_offsets.push(step_offsets.len() - 1);
            }
            (token_ids, values, step_offsets, beam_offsets)
        } else {
            (
                Vec::new(),
                Vec::new(),
                vec![0],
                vec![0; final_beams.len() + 1],
            )
        };
        let observed_completed_count = if diagnostics_enabled {
            self.observed_completed
                .get(&root_request_id)
                .map(Vec::len)
                .unwrap_or(0)
        } else {
            0
        };
        if finished {
            warn!(
                worker_id = self.worker_id,
                root_request_id = %root_request_id,
                correlation_id = %correlation_id,
                beam_width,
                outputs = outputs.len(),
                candidates_before_retain = candidate_count_before_retain,
                candidates_after_retain = candidate_count_after_retain,
                candidates_after_rank = candidates.len(),
                invalid_candidate_count,
                completed_count,
                context_count = outputs.iter().filter(|output| output.beam_context.is_some()).count(),
                final_context_count,
                max_step_index = ?max_step_index,
                total_decode_steps = ?total_decode_steps,
                selected_invalid,
                selected_explicit_rejected,
                selected_catalog_rejected,
                selected_eos_rejected,
                observed_completed = observed_completed_count,
                final_beams = final_token_ids.len(),
                reached_final_step,
                finished,
                "BEAMPROC_RUST_FINAL_DIAG"
            );
        }
        debug!(
            worker_id = self.worker_id,
            root_request_id = %root_request_id,
            correlation_id = %correlation_id,
            beam_width,
            outputs = outputs.len(),
            candidates = candidates.len(),
            invalid_candidate_count,
            completed_count,
            context_count = outputs.iter().filter(|output| output.beam_context.is_some()).count(),
            final_context_count,
            max_step_index = ?max_step_index,
            total_decode_steps = ?total_decode_steps,
            selected_invalid,
            selected_explicit_rejected,
            selected_catalog_rejected,
            selected_eos_rejected,
            observed_completed = observed_completed_count,
            final_beams = final_token_ids.len(),
            reached_final_step,
            finished,
            "beam worker built observed plan"
        );
        let beam_tokens = if !finished {
            let mut child_updates = child_ids
                .iter()
                .zip(candidates.iter())
                .map(|(child_id, candidate)| {
                    (
                        child_id.clone(),
                        materialize_candidate_tokens(batch, family, candidate),
                        if return_logprobs {
                            materialize_candidate_logprob_steps(
                                batch,
                                family,
                                candidate,
                                &current_logprob_steps,
                            )
                        } else {
                            Vec::new()
                        },
                    )
                })
                .collect::<Vec<_>>();
            if let Some(family) = self.shadow_families.get_mut(&root_request_id) {
                for (child_id, token_ids, logprob_steps) in &mut child_updates {
                    family.record_parent_tokens(child_id.clone(), token_ids.clone());
                    if return_logprobs {
                        family.record_parent_logprob_steps(
                            child_id.clone(),
                            std::mem::take(logprob_steps),
                        );
                    }
                }
            }
            child_updates
                .into_iter()
                .map(|(_, token_ids, _)| token_ids)
                .collect()
        } else {
            Vec::new()
        };
        let plan_beam_width = if finished {
            beam_width
        } else {
            child_ids.len()
        };
        candidates.clear();
        let mut plan = BeamWorkerPlan {
            timing: None,
            num_processed_tokens: processed_candidates,
            engine_index,
            client_index,
            request_id: root_request_id.clone(),
            root_request_id,
            correlation_id,
            action: "observed_full".to_string(),
            beam_width: plan_beam_width,
            scored_parent_ids,
            parent_ids,
            child_ids,
            token_ids,
            beam_tokens,
            scores,
            flat_indices,
            abort_ids: Vec::new(),
            completed_count,
            finished,
            scheduled_final: reached_final_step,
            final_token_ids_include_prompt,
            final_token_ids,
            final_scores,
            final_finish_reasons,
            final_stop_token_ids,
            final_logprob_token_ids,
            final_logprob_values,
            final_logprob_step_offsets,
            final_logprob_beam_offsets,
        };
        if let Some(family) = self.shadow_families.get_mut(&plan.root_request_id) {
            family.num_processed_tokens += processed_candidates;
            plan.num_processed_tokens = family.num_processed_tokens;
            if !family.timing_recorded {
                family.timing_recorded = true;
                family.timing = beam_timing(plan.finished);
            }
            plan.timing = family.timing;
        }
        plan
    }
}

#[derive(Debug, Clone)]
struct ObservedFinalBeam {
    token_ids: Vec<i32>,
    score: f32,
    finish_reason: BeamFinishReason,
    stop_token_id: Option<i32>,
    order: usize,
    logprob_steps: SharedBeamLogprobSteps,
}

impl BeamFinishReason {
    fn as_str(self) -> &'static str {
        match self {
            BeamFinishReason::Stop => "stop",
            BeamFinishReason::Length => "length",
        }
    }
}

struct ShadowBeamFamily {
    timing_recorded: bool,
    timing: Option<BeamTiming>,
    num_processed_tokens: u64,
    beam_width: usize,
    active_parent_ids: Vec<String>,
    parent_cum_logprobs: HashMap<String, f32>,
    parent_valid_token_ids: HashMap<String, Vec<i32>>,
    parent_token_ids: HashMap<String, Vec<i32>>,
    parent_logprob_steps: HashMap<String, SharedBeamLogprobSteps>,
    prompt_token_count: Option<usize>,
    ignore_eos: Option<bool>,
    include_stop_token: Option<bool>,
    eos_token_id: Option<Option<i32>>,
    end_token_id: Option<Option<i32>>,
    catalog_strategy: Option<String>,
    return_logprobs: Option<bool>,
}

struct PendingEngineOutputs {
    engine_index: usize,
    client_index: usize,
    request_id: String,
    correlation_id: String,
    batch: BeamEngineOutputBatch,
    outputs_by_parent: HashMap<String, usize>,
}

impl PendingEngineOutputs {
    fn new(
        engine_index: usize,
        client_index: usize,
        request_id: String,
        correlation_id: String,
    ) -> Self {
        Self {
            engine_index,
            client_index,
            request_id,
            correlation_id,
            batch: BeamEngineOutputBatch::with_capacity(0),
            outputs_by_parent: HashMap::new(),
        }
    }
}

impl ShadowBeamFamily {
    fn new(_root_request_id: String) -> Self {
        Self {
            timing_recorded: false,
            timing: None,
            num_processed_tokens: 0,
            beam_width: 0,
            active_parent_ids: Vec::new(),
            parent_cum_logprobs: HashMap::new(),
            parent_valid_token_ids: HashMap::new(),
            parent_token_ids: HashMap::new(),
            parent_logprob_steps: HashMap::new(),
            prompt_token_count: None,
            ignore_eos: None,
            include_stop_token: None,
            eos_token_id: None,
            end_token_id: None,
            catalog_strategy: None,
            return_logprobs: None,
        }
    }

    fn record_root_context(&mut self, context: &BeamContext) {
        if context.prompt_token_count.is_some() {
            self.prompt_token_count = context.prompt_token_count;
        }
        self.ignore_eos = Some(context.ignore_eos);
        self.include_stop_token = Some(context.include_stop_token);
        self.eos_token_id = Some(context.eos_token_id);
        self.end_token_id = Some(context.end_token_id);
        self.catalog_strategy = Some(context.catalog_strategy.clone());
        self.return_logprobs = Some(context.return_logprobs);
    }

    fn update(
        &mut self,
        beam_width: usize,
        active_parent_ids: Vec<String>,
        active_parent_cum_logprobs: Vec<f32>,
        active_parent_valid_token_ids: Vec<Vec<i32>>,
    ) {
        self.beam_width = beam_width;
        self.active_parent_ids = active_parent_ids.clone();
        self.parent_cum_logprobs.clear();
        self.parent_valid_token_ids.clear();
        let active_parent_id_set = active_parent_ids
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        self.parent_token_ids
            .retain(|parent_id, _| active_parent_id_set.contains(parent_id.as_str()));
        self.parent_logprob_steps
            .retain(|parent_id, _| active_parent_id_set.contains(parent_id.as_str()));
        let mut valid_token_ids = active_parent_valid_token_ids.into_iter();
        for (index, parent_id) in active_parent_ids.into_iter().enumerate() {
            if let Some(cum_logprob) = active_parent_cum_logprobs.get(index) {
                self.parent_cum_logprobs
                    .insert(parent_id.clone(), *cum_logprob);
            }
            if let Some(mut parent_valid_token_ids) = valid_token_ids.next() {
                parent_valid_token_ids.sort_unstable();
                parent_valid_token_ids.dedup();
                self.parent_valid_token_ids
                    .insert(parent_id, parent_valid_token_ids);
            }
        }
    }

    fn record_parent_tokens(&mut self, parent_id: String, token_ids: Vec<i32>) {
        if !token_ids.is_empty() {
            self.parent_token_ids.insert(parent_id, token_ids);
        }
    }

    fn parent_tokens(&self, parent_id: &str) -> Option<&[i32]> {
        self.parent_token_ids.get(parent_id).map(Vec::as_slice)
    }

    fn record_parent_logprob_steps(
        &mut self,
        parent_id: String,
        logprob_steps: SharedBeamLogprobSteps,
    ) {
        self.parent_logprob_steps.insert(parent_id, logprob_steps);
    }

    fn parent_logprob_steps(&self, parent_id: &str) -> Option<&[Arc<BeamLogprobStep>]> {
        self.parent_logprob_steps.get(parent_id).map(Vec::as_slice)
    }
}

fn candidate_parent_tokens<'a>(
    batch: &'a BeamEngineOutputBatch,
    family: Option<&'a ShadowBeamFamily>,
    candidate: &ObservedCandidate,
) -> &'a [i32] {
    let Some(output) = batch.outputs.get(candidate.parent_index) else {
        return &[];
    };
    output
        .beam_context
        .as_ref()
        .and_then(|context| {
            (!context.parent_token_ids.is_empty()).then_some(context.parent_token_ids.as_slice())
        })
        .or_else(|| family.and_then(|family| family.parent_tokens(&output.request_id)))
        .unwrap_or(&[])
}

fn candidate_row<T>(flat: &[T], row: usize, cols: usize) -> &[T] {
    if cols == 0 {
        return &[];
    }
    let start = row.saturating_mul(cols);
    let end = start.saturating_add(cols);
    flat.get(start..end).unwrap_or(&[])
}

fn is_beamproc_child_request(request_id: &str) -> bool {
    request_id.contains("-beamproc-")
}

fn beamproc_child_step(request_id: &str) -> Option<usize> {
    request_id
        .rsplit_once("-beamproc-")
        .and_then(|(_, suffix)| suffix.split('-').next())
        .and_then(|step| step.parse().ok())
}

fn output_has_matching_shadow_history(
    family: &ShadowBeamFamily,
    output: &BeamEngineOutput,
) -> bool {
    let Some(context) = output.beam_context.as_ref() else {
        return true;
    };
    !context.use_shadow_parent_tokens
        || (family.parent_tokens(&output.request_id).is_some()
            && context.step_index == beamproc_child_step(&output.request_id))
}

fn materialize_candidate_tokens(
    batch: &BeamEngineOutputBatch,
    family: Option<&ShadowBeamFamily>,
    candidate: &ObservedCandidate,
) -> Vec<i32> {
    let parent_tokens = candidate_parent_tokens(batch, family, candidate);
    let mut token_ids = Vec::with_capacity(parent_tokens.len() + 1);
    token_ids.extend_from_slice(parent_tokens);
    token_ids.push(candidate.token_id);
    token_ids
}

fn materialize_candidate_logprob_steps(
    batch: &BeamEngineOutputBatch,
    family: Option<&ShadowBeamFamily>,
    candidate: &ObservedCandidate,
    current_steps: &[Arc<BeamLogprobStep>],
) -> SharedBeamLogprobSteps {
    let parent_id = batch
        .outputs
        .get(candidate.parent_index)
        .map(|output| output.request_id.as_str());
    let parent_steps = parent_id
        .and_then(|parent_id| family.and_then(|family| family.parent_logprob_steps(parent_id)))
        .unwrap_or(&[]);
    let mut steps = Vec::with_capacity(parent_steps.len() + 1);
    steps.extend_from_slice(parent_steps);
    if let Some(step) = current_steps.get(candidate.parent_index) {
        steps.push(step.clone());
    }
    steps
}

fn infer_beam_width(batch: &BeamEngineOutputBatch) -> usize {
    batch
        .outputs
        .first()
        .map(|output| {
            output
                .beam_context
                .as_ref()
                .and_then(|context| context.beam_width)
                .filter(|beam_width| *beam_width > 0)
                .unwrap_or_else(|| output.candidate_cols.saturating_sub(1).max(1))
        })
        .unwrap_or_else(|| batch.output_count().max(1))
}

fn sampled_candidates<'a>(
    tokens: &'a [i32],
    logprobs: &'a [f32],
    beam_width: usize,
) -> impl Iterator<Item = (i32, f32)> + 'a {
    let len = tokens.len().min(logprobs.len());
    let (start, end) = if len == 0 || beam_width == 0 {
        (0, 0)
    } else if len > beam_width {
        let sampled_token = tokens[0];
        let has_sampled_duplicate = (1..=beam_width).any(|index| tokens[index] == sampled_token);
        if has_sampled_duplicate {
            (1, beam_width + 1)
        } else {
            (0, beam_width)
        }
    } else {
        (0, len)
    };

    tokens[start..end]
        .iter()
        .copied()
        .zip(logprobs[start..end].iter().copied())
}

fn matching_stop_token(
    token_id: i32,
    eos_token_id: Option<i32>,
    end_token_id: Option<i32>,
) -> Option<i32> {
    if eos_token_id == Some(token_id) {
        eos_token_id
    } else if end_token_id == Some(token_id) {
        end_token_id
    } else {
        None
    }
}

fn apply_catalog_strategy(
    token_id: i32,
    logprob: f32,
    valid_token_ids: Option<&[i32]>,
    fill_catalog: bool,
    fill_used: &mut usize,
) -> (i32, f32, bool, bool) {
    let Some(valid_token_ids) = valid_token_ids else {
        return (token_id, logprob, false, false);
    };
    if valid_token_ids.binary_search(&token_id).is_ok() {
        return (token_id, logprob, false, false);
    }
    if fill_catalog && !valid_token_ids.is_empty() {
        let fill_index = *fill_used % valid_token_ids.len();
        *fill_used += 1;
        return (
            valid_token_ids[fill_index],
            -10_000.0 - fill_index as f32 * 0.1,
            true,
            false,
        );
    }
    (token_id, logprob, true, true)
}

fn compare_observed_candidates(
    left: &ObservedCandidate,
    right: &ObservedCandidate,
) -> std::cmp::Ordering {
    right
        .continuation_score
        .total_cmp(&left.continuation_score)
        .then_with(|| left.flat_index.cmp(&right.flat_index))
}

fn rank_top_candidates(candidates: &mut Vec<ObservedCandidate>, beam_width: usize) {
    if beam_width == 0 {
        candidates.clear();
        return;
    }
    if candidates.len() > beam_width {
        candidates.select_nth_unstable_by(beam_width - 1, compare_observed_candidates);
        candidates.truncate(beam_width);
    }
    candidates.sort_by(compare_observed_candidates);
}

#[cfg(test)]
impl RequestState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        request_id: String,
        beam_width: usize,
        max_tokens: usize,
        ignore_eos: bool,
        include_stop_token: bool,
        eos_token_id: Option<i32>,
        end_token_id: Option<i32>,
        initial_engine_request_id: String,
    ) -> Self {
        Self {
            request_id,
            beam_width,
            max_tokens,
            ignore_eos,
            include_stop_token,
            eos_token_id,
            end_token_id,
            current_step: 0,
            next_beam_id: 1,
            active: vec![ActiveBeam {
                beam_id: 0,
                engine_request_id: initial_engine_request_id,
                tokens: Vec::new(),
                cum_logprob: 0.0,
            }],
            completed: Vec::new(),
        }
    }

    fn process_engine_step(&mut self, step: usize, candidates: Vec<BeamCandidate>) -> ForkPlan {
        self.current_step = step;
        let mut continuations = Vec::new();
        let mut newly_completed = Vec::new();

        for (candidate_order, candidate) in candidates.into_iter().enumerate() {
            let Some(parent) = self
                .active
                .iter()
                .find(|beam| beam.beam_id == candidate.parent_beam_id)
            else {
                warn!(
                    request_id = %self.request_id,
                    parent_beam_id = candidate.parent_beam_id,
                    "beam worker ignored candidate for unknown parent beam"
                );
                continue;
            };

            let cum_logprob = parent.cum_logprob + candidate.logprob;
            let is_eos = self.eos_token_id == Some(candidate.token_id);

            if is_eos && !self.ignore_eos {
                let mut tokens = parent.tokens.clone();
                if self.include_stop_token {
                    tokens.push(candidate.token_id);
                }
                newly_completed.push(CompletedBeam {
                    beam_id: self.next_beam_id,
                    tokens,
                    cum_logprob,
                    finish_reason: BeamFinishReason::Stop,
                    stop_token_id: self.eos_token_id,
                });
                self.next_beam_id += 1;
                continue;
            }

            let mut tokens = parent.tokens.clone();
            tokens.push(candidate.token_id);
            continuations.push(ScoredContinuation {
                candidate_order,
                parent_engine_request_id: parent.engine_request_id.clone(),
                token_id: candidate.token_id,
                tokens,
                cum_logprob,
            });
        }

        self.completed.extend(newly_completed);
        continuations.sort_by(|left, right| {
            right
                .cum_logprob
                .total_cmp(&left.cum_logprob)
                .then_with(|| left.candidate_order.cmp(&right.candidate_order))
        });
        continuations.truncate(self.beam_width);
        continuations.sort_by_key(|continuation| continuation.candidate_order);

        let previous_engine_ids = self
            .active
            .iter()
            .map(|beam| beam.engine_request_id.clone())
            .collect::<Vec<_>>();

        let mut next_active = Vec::with_capacity(continuations.len());
        let mut parent_engine_request_ids = Vec::with_capacity(continuations.len());
        let mut child_engine_request_ids = Vec::with_capacity(continuations.len());
        let mut token_ids = Vec::with_capacity(continuations.len());

        for (rank, continuation) in continuations.into_iter().enumerate() {
            let child_beam_id = self.next_beam_id;
            self.next_beam_id += 1;
            let child_engine_request_id = format!("{}-beam-{}-{}", self.request_id, step, rank);

            parent_engine_request_ids.push(continuation.parent_engine_request_id);
            child_engine_request_ids.push(child_engine_request_id.clone());
            token_ids.push(continuation.token_id);

            next_active.push(ActiveBeam {
                beam_id: child_beam_id,
                engine_request_id: child_engine_request_id,
                tokens: continuation.tokens,
                cum_logprob: continuation.cum_logprob,
            });
        }

        let kept_parent_ids = parent_engine_request_ids
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let abort_engine_request_ids = previous_engine_ids
            .into_iter()
            .filter(|engine_id| !kept_parent_ids.contains(engine_id))
            .collect::<Vec<_>>();

        self.active = next_active;
        let finished = self.current_step >= self.max_tokens || self.active.is_empty();
        if finished {
            self.finish_active_beams();
        }

        let mut completed = self.completed.clone();
        completed.sort_by(|left, right| right.cum_logprob.total_cmp(&left.cum_logprob));
        completed.truncate(self.beam_width);

        ForkPlan {
            request_id: self.request_id.clone(),
            step,
            parent_engine_request_ids,
            child_engine_request_ids,
            token_ids,
            abort_engine_request_ids,
            completed,
            finished,
        }
    }

    fn finish_active_beams(&mut self) {
        for beam in self.active.drain(..) {
            let mut tokens = beam.tokens;
            if let Some(end_token_id) = self.end_token_id {
                tokens.push(end_token_id);
            }
            self.completed.push(CompletedBeam {
                beam_id: beam.beam_id,
                tokens,
                cum_logprob: beam.cum_logprob,
                finish_reason: BeamFinishReason::Length,
                stop_token_id: None,
            });
        }
    }
}

#[cfg(test)]
#[derive(Debug)]
struct ScoredContinuation {
    candidate_order: usize,
    parent_engine_request_id: String,
    token_id: i32,
    tokens: Vec<i32>,
    cum_logprob: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> RequestState {
        RequestState::new(
            "req".to_string(),
            2,
            3,
            false,
            false,
            Some(2),
            Some(99),
            "engine-0".to_string(),
        )
    }

    #[test]
    fn selects_top_k_continuations_and_builds_fork_plan() {
        let mut state = state();

        let plan = state.process_engine_step(
            1,
            vec![
                BeamCandidate {
                    parent_beam_id: 0,
                    token_id: 10,
                    logprob: -0.3,
                },
                BeamCandidate {
                    parent_beam_id: 0,
                    token_id: 11,
                    logprob: -0.1,
                },
                BeamCandidate {
                    parent_beam_id: 0,
                    token_id: 12,
                    logprob: -0.2,
                },
            ],
        );

        assert!(!plan.finished);
        assert_eq!(plan.token_ids, vec![11, 12]);
        assert_eq!(
            plan.parent_engine_request_ids,
            vec!["engine-0".to_string(), "engine-0".to_string()]
        );
        assert_eq!(
            plan.child_engine_request_ids,
            vec!["req-beam-1-0".to_string(), "req-beam-1-1".to_string()]
        );
        assert!(plan.abort_engine_request_ids.is_empty());
    }

    #[test]
    fn schedules_selected_continuations_in_candidate_order() {
        let mut state = state();

        let plan = state.process_engine_step(
            1,
            vec![
                BeamCandidate {
                    parent_beam_id: 0,
                    token_id: 10,
                    logprob: -0.2,
                },
                BeamCandidate {
                    parent_beam_id: 0,
                    token_id: 11,
                    logprob: -0.1,
                },
                BeamCandidate {
                    parent_beam_id: 0,
                    token_id: 12,
                    logprob: -0.3,
                },
            ],
        );

        assert!(!plan.finished);
        assert_eq!(plan.token_ids, vec![10, 11]);
    }

    #[test]
    fn moves_eos_candidates_to_completed_and_excludes_them_from_fork() {
        let mut state = state();

        let plan = state.process_engine_step(
            1,
            vec![
                BeamCandidate {
                    parent_beam_id: 0,
                    token_id: 2,
                    logprob: -0.05,
                },
                BeamCandidate {
                    parent_beam_id: 0,
                    token_id: 10,
                    logprob: -0.1,
                },
            ],
        );

        assert_eq!(plan.token_ids, vec![10]);
        assert_eq!(plan.completed.len(), 1);
        assert_eq!(plan.completed[0].tokens, Vec::<i32>::new());
        assert_eq!(plan.completed[0].finish_reason, BeamFinishReason::Stop);
        assert_eq!(plan.completed[0].stop_token_id, Some(2));
    }

    #[test]
    fn finishes_active_beams_with_end_token_at_max_tokens() {
        let mut state = RequestState::new(
            "req".to_string(),
            1,
            1,
            false,
            false,
            Some(2),
            Some(99),
            "engine-0".to_string(),
        );

        let plan = state.process_engine_step(
            1,
            vec![BeamCandidate {
                parent_beam_id: 0,
                token_id: 10,
                logprob: -0.1,
            }],
        );

        assert!(plan.finished);
        assert_eq!(plan.completed.len(), 1);
        assert_eq!(plan.completed[0].tokens, vec![10, 99]);
        assert_eq!(plan.completed[0].finish_reason, BeamFinishReason::Length);
    }

    #[test]
    fn dedups_sampled_token_when_it_appears_in_topk() {
        let candidates =
            sampled_candidates(&[10, 20, 10, 30, 40], &[0.0, -0.1, -0.2, -0.3, -0.4], 3)
                .collect::<Vec<_>>();

        assert_eq!(candidates, vec![(20, -0.1), (10, -0.2), (30, -0.3)]);
    }

    #[test]
    fn keeps_sampled_token_when_it_is_not_in_topk() {
        let candidates =
            sampled_candidates(&[10, 20, 30, 40, 50], &[0.0, -0.1, -0.2, -0.3, -0.4], 3)
                .collect::<Vec<_>>();

        assert_eq!(candidates, vec![(10, 0.0), (20, -0.1), (30, -0.2)]);
    }

    #[test]
    fn supports_candidate_windows_above_128() {
        let beam_width = 256;
        let mut tokens = vec![42];
        tokens.extend(0..=beam_width as i32);
        let logprobs = (0..tokens.len())
            .map(|index| -(index as f32))
            .collect::<Vec<_>>();

        let candidates = sampled_candidates(&tokens, &logprobs, beam_width).collect::<Vec<_>>();

        assert_eq!(candidates.len(), beam_width);
        assert_eq!(candidates.first().map(|candidate| candidate.0), Some(0));
        assert_eq!(candidates.last().map(|candidate| candidate.0), Some(255));
    }

    #[test]
    fn recognizes_distinct_custom_end_token() {
        assert_eq!(matching_stop_token(2, Some(2), Some(99)), Some(2));
        assert_eq!(matching_stop_token(99, Some(2), Some(99)), Some(99));
        assert_eq!(matching_stop_token(7, Some(2), Some(99)), None);
    }

    fn grouped_fixture(width: usize) -> (BeamWorkerState, BeamEngineOutputBatch) {
        let (tx, _) = crossbeam_channel::unbounded();
        let mut state = BeamWorkerState::new(0, tx, None);
        let ids: Vec<String> = (0..width).map(|i| format!("parent-{i}")).collect();
        let mut family = ShadowBeamFamily::new("root".into());
        family.update(width, ids.clone(), vec![0.0; width], Vec::new());
        family.return_logprobs = Some(false);
        state.shadow_families.insert("root".into(), family);
        let mut batch = BeamEngineOutputBatch::with_capacity(width);
        for (i, id) in ids.into_iter().enumerate() {
            batch
                .candidate_token_ids
                .extend((0..width).map(|j| j as i32 + 10));
            batch
                .candidate_logprobs
                .extend((0..width).map(|j| -(j as f32)));
            batch.outputs.push(BeamEngineOutput {
                request_id: id,
                new_token_ids: vec![],
                finish_reason: None,
                stop_reason: None,
                num_cached_tokens: None,
                candidate_token_start: i * width,
                candidate_token_end: (i + 1) * width,
                candidate_logprob_start: i * width,
                candidate_logprob_end: (i + 1) * width,
                candidate_rows: 1,
                candidate_cols: width,
                sampled_rank_start: 0,
                sampled_rank_end: 0,
                beam_context: None,
            });
        }
        (state, batch)
    }

    #[test]
    fn complete_grouped_batch_matches_reordered_and_partial_delivery() {
        let (mut direct, mut batch) = grouped_fixture(4);
        let plan = direct
            .buffer_or_build_full_observed_plan(0, 0, "root".into(), "c".into(), &mut batch)
            .unwrap();
        assert_eq!(batch.outputs.len(), 4); // The complete path retains pooled storage.
        assert_eq!(plan.num_processed_tokens, 16);
        for partial in [false, true] {
            let (mut state, mut other) = grouped_fixture(4);
            other.outputs.reverse();
            if partial {
                let remaining = other.outputs.split_off(2);
                let mut first = BeamEngineOutputBatch::with_capacity(2);
                for output in remaining {
                    first.append_output(output, &other);
                }
                assert!(
                    state
                        .buffer_or_build_full_observed_plan(
                            0,
                            0,
                            "root".into(),
                            "a".into(),
                            &mut first
                        )
                        .is_none()
                );
            }
            let result = state
                .buffer_or_build_full_observed_plan(0, 0, "root".into(), "c".into(), &mut other)
                .unwrap();
            assert_eq!(result.parent_ids, plan.parent_ids);
            assert_eq!(result.num_processed_tokens, 16);
            assert_eq!(result.child_ids, plan.child_ids);
            assert_eq!(result.token_ids, plan.token_ids);
            assert_eq!(result.scores, plan.scores);
            assert_eq!(result.flat_indices, plan.flat_indices);
            assert_eq!(result.beam_tokens, plan.beam_tokens);
            assert!(state.pending_engine_outputs.is_empty());
        }
    }

    #[test]
    #[ignore = "CPU microbenchmark; run explicitly with --ignored --nocapture"]
    fn benchmark_complete_grouped_batch() {
        for buffered in [true, false, false, true] {
            let (mut state, batch) = grouped_fixture(128);
            let mut elapsed = std::time::Duration::ZERO;
            for i in 0..1100 {
                let mut input = BeamEngineOutputBatch::with_capacity(128);
                for output in &batch.outputs {
                    input.append_output(output.clone(), &batch);
                }
                if buffered {
                    state.pending_engine_outputs.insert(
                        "root".into(),
                        PendingEngineOutputs::new(0, 0, "root".into(), "c".into()),
                    );
                }
                let start = std::time::Instant::now();
                let result = state
                    .buffer_or_build_full_observed_plan(0, 0, "root".into(), "c".into(), &mut input)
                    .unwrap();
                std::hint::black_box(result);
                if i >= 100 {
                    elapsed += start.elapsed();
                }
            }
            println!(
                "BW128 complete batch buffered={buffered}: {:.2} us/step (100 warmups, 1000 measured)",
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }

    #[test]
    fn catalog_fill_replaces_invalid_candidates_like_python() {
        let mut fill_used = 0;

        assert_eq!(
            apply_catalog_strategy(7, -0.5, Some(&[10, 20]), true, &mut fill_used),
            (10, -10_000.0, true, false),
        );
        assert_eq!(
            apply_catalog_strategy(8, -0.6, Some(&[10, 20]), true, &mut fill_used),
            (20, -10_000.1, true, false),
        );
    }

    #[test]
    fn catalog_null_rejects_invalid_candidates() {
        let mut fill_used = 0;

        assert_eq!(
            apply_catalog_strategy(7, -0.5, Some(&[10, 20]), false, &mut fill_used),
            (7, -0.5, true, true),
        );
        assert_eq!(fill_used, 0);
    }

    #[test]
    fn ignored_eos_remains_active_without_recording_completed_stop() {
        let (plan_tx, _plan_rx) = crossbeam_channel::unbounded();
        let mut state = BeamWorkerState::new(0, plan_tx, None);
        let root_request_id = "root".to_string();
        let mut family = ShadowBeamFamily::new(root_request_id.clone());
        family.update(1, vec!["parent".to_string()], vec![0.0], vec![vec![2]]);
        state
            .shadow_families
            .insert(root_request_id.clone(), family);

        let mut batch = BeamEngineOutputBatch {
            candidate_token_ids: vec![2],
            candidate_logprobs: vec![-0.1],
            sampled_token_ranks: vec![0],
            outputs: vec![BeamEngineOutput {
                request_id: "parent".to_string(),
                new_token_ids: vec![2],
                finish_reason: None,
                stop_reason: None,
                num_cached_tokens: None,
                candidate_token_start: 0,
                candidate_token_end: 1,
                candidate_logprob_start: 0,
                candidate_logprob_end: 1,
                candidate_rows: 1,
                candidate_cols: 1,
                sampled_rank_start: 0,
                sampled_rank_end: 1,
                beam_context: Some(BeamContext {
                    beam_width: Some(1),
                    max_tokens: Some(1),
                    sampling_max_tokens: Some(1),
                    global_max_tokens: Some(2),
                    total_decode_steps: Some(2),
                    step_index: Some(0),
                    is_final_step: false,
                    ignore_eos: true,
                    include_stop_token: false,
                    eos_token_id: Some(2),
                    end_token_id: None,
                    catalog_strategy: "fill".to_string(),
                    return_logprobs: true,
                    parent_token_ids: vec![100],
                    use_shadow_parent_tokens: false,
                    prompt_token_count: Some(1),
                    client_index: 0,
                    priority: 0,
                }),
            }],
        };

        let plan = state.build_full_observed_plan(
            0,
            0,
            root_request_id.clone(),
            "corr:0".to_string(),
            &batch,
        );

        assert!(!plan.finished);
        assert_eq!(plan.num_processed_tokens, 1);
        let initial_timing = plan.timing.unwrap();
        assert_eq!(initial_timing.clock, "CLOCK_MONOTONIC");
        assert!(!initial_timing.prefill_only);
        assert!(initial_timing.prefill_end > 0.0);
        assert!(initial_timing.prefill_end <= beam_timing(false).unwrap().prefill_end);
        assert_eq!(plan.token_ids, vec![2]);
        assert!(!state.observed_completed.contains_key(&root_request_id));

        // Reuse the worker across requests with growing and shrinking widths.
        // Old plans must own their data, and smaller steps must not reallocate.
        for width in [1, 32, 2, 32] {
            let previous_capacity = state.candidate_scratch.capacity();
            let previous_ptr = state.candidate_scratch.as_ptr();
            batch.candidate_token_ids = (1000..1000 + width as i32).collect();
            batch.candidate_logprobs = (0..width).map(|i| -(i as f32)).collect();
            let output = &mut batch.outputs[0];
            output.candidate_token_end = width;
            output.candidate_logprob_end = width;
            output.candidate_cols = width;
            output.beam_context.as_mut().unwrap().beam_width = Some(width);
            let next_plan = state.build_full_observed_plan(
                0,
                0,
                format!("other-root-{width}"),
                "reuse".to_string(),
                &batch,
            );
            assert!(next_plan.timing.unwrap().prefill_end >= initial_timing.prefill_end);
            assert_eq!(
                state.shadow_families["root"].timing.unwrap().prefill_end,
                initial_timing.prefill_end
            );
            assert_eq!(next_plan.token_ids, batch.candidate_token_ids);
            assert!(state.candidate_scratch.is_empty());
            assert!(state.candidate_scratch.capacity() >= width);
            if width <= previous_capacity {
                assert_eq!(state.candidate_scratch.capacity(), previous_capacity);
                assert_eq!(state.candidate_scratch.as_ptr(), previous_ptr);
            }
            assert_eq!(plan.token_ids, vec![2]);
        }
        // A later final plan retains the first boundary, even after other roots run.
        batch.outputs[0]
            .beam_context
            .as_mut()
            .unwrap()
            .is_final_step = true;
        let final_plan =
            state.build_full_observed_plan(0, 0, "root".into(), "final".into(), &batch);
        assert!(final_plan.finished);
        // Other roots and scratch-buffer reuse must not contaminate this total.
        assert_eq!(final_plan.num_processed_tokens, 2);
        assert!(final_plan.scheduled_final);
        assert_eq!(
            final_plan.timing.unwrap().prefill_end,
            initial_timing.prefill_end
        );
        assert!(!final_plan.timing.unwrap().prefill_only);
        let immediate =
            state.build_full_observed_plan(0, 0, "early".into(), "final".into(), &batch);
        assert!(immediate.timing.unwrap().prefill_only);
        assert_eq!(immediate.num_processed_tokens, 32);
        // The normal completion handler owns removal of the entire timing-bearing family.
        state.handle_engine_output_batch(0, 0, "root".into(), "done".into(), batch);
        assert!(!state.shadow_families.contains_key("root"));
    }

    #[test]
    fn observed_plan_merges_completed_eos_with_final_active_beams() {
        let (plan_tx, _plan_rx) = crossbeam_channel::unbounded();
        let mut state = BeamWorkerState::new(0, plan_tx, None);
        let root_request_id = "root".to_string();
        let mut family = ShadowBeamFamily::new(root_request_id.clone());
        family.update(2, vec!["parent".to_string()], vec![0.0], vec![vec![2, 10]]);
        state
            .shadow_families
            .insert(root_request_id.clone(), family);

        let first_batch = BeamEngineOutputBatch {
            candidate_token_ids: vec![2, 10],
            candidate_logprobs: vec![-0.1, -0.2],
            sampled_token_ranks: vec![0],
            outputs: vec![BeamEngineOutput {
                request_id: "parent".to_string(),
                new_token_ids: vec![10],
                finish_reason: None,
                stop_reason: None,
                num_cached_tokens: None,
                candidate_token_start: 0,
                candidate_token_end: 2,
                candidate_logprob_start: 0,
                candidate_logprob_end: 2,
                candidate_rows: 1,
                candidate_cols: 2,
                sampled_rank_start: 0,
                sampled_rank_end: 1,
                beam_context: Some(BeamContext {
                    beam_width: Some(2),
                    max_tokens: Some(1),
                    sampling_max_tokens: Some(1),
                    global_max_tokens: Some(3),
                    total_decode_steps: Some(2),
                    step_index: Some(0),
                    is_final_step: false,
                    ignore_eos: false,
                    include_stop_token: false,
                    eos_token_id: Some(2),
                    end_token_id: None,
                    catalog_strategy: "fill".to_string(),
                    return_logprobs: true,
                    parent_token_ids: vec![100],
                    use_shadow_parent_tokens: false,
                    prompt_token_count: Some(1),
                    client_index: 0,
                    priority: 0,
                }),
            }],
        };

        let first_plan = state.build_full_observed_plan(
            0,
            0,
            root_request_id.clone(),
            "corr:0".to_string(),
            &first_batch,
        );

        assert!(!first_plan.finished);
        assert_eq!(first_plan.num_processed_tokens, 2); // Includes the completed EOS candidate.
        assert_eq!(first_plan.beam_width, 1);
        assert_eq!(first_plan.token_ids, vec![10]);
        assert_eq!(first_plan.beam_tokens, vec![vec![100, 10]]);
        assert_eq!(state.observed_completed[&root_request_id].len(), 1);

        let child_id = first_plan.child_ids[0].clone();
        state
            .shadow_families
            .get_mut(&root_request_id)
            .unwrap()
            .update(2, vec![child_id.clone()], vec![-0.2], vec![vec![11]]);
        let final_batch = BeamEngineOutputBatch {
            candidate_token_ids: vec![11],
            candidate_logprobs: vec![-0.2],
            sampled_token_ranks: vec![0],
            outputs: vec![BeamEngineOutput {
                request_id: child_id,
                new_token_ids: vec![11],
                finish_reason: Some("length".to_string()),
                stop_reason: None,
                num_cached_tokens: None,
                candidate_token_start: 0,
                candidate_token_end: 1,
                candidate_logprob_start: 0,
                candidate_logprob_end: 1,
                candidate_rows: 1,
                candidate_cols: 1,
                sampled_rank_start: 0,
                sampled_rank_end: 1,
                beam_context: Some(BeamContext {
                    beam_width: Some(2),
                    max_tokens: Some(1),
                    sampling_max_tokens: Some(1),
                    global_max_tokens: Some(3),
                    total_decode_steps: Some(2),
                    step_index: Some(1),
                    is_final_step: true,
                    ignore_eos: false,
                    include_stop_token: false,
                    eos_token_id: Some(2),
                    end_token_id: None,
                    catalog_strategy: "fill".to_string(),
                    return_logprobs: true,
                    parent_token_ids: Vec::new(),
                    use_shadow_parent_tokens: true,
                    prompt_token_count: Some(1),
                    client_index: 0,
                    priority: 0,
                }),
            }],
        };

        let final_plan = state.build_full_observed_plan(
            0,
            0,
            root_request_id,
            "corr:1".to_string(),
            &final_batch,
        );

        assert!(final_plan.finished);
        assert!(final_plan.scheduled_final);
        assert!(!final_plan.final_token_ids_include_prompt);
        assert_eq!(final_plan.num_processed_tokens, 3);
        assert_eq!(final_plan.final_token_ids, vec![vec![], vec![10, 11]]);
        assert!(final_plan.parent_ids.is_empty());
        assert!(final_plan.child_ids.is_empty());
        assert!(final_plan.token_ids.is_empty());
        assert!(final_plan.scores.is_empty());
        assert!(final_plan.flat_indices.is_empty());
        assert_eq!(final_plan.final_scores, vec![-0.1, -0.4]);
        assert_eq!(final_plan.final_finish_reasons, vec!["stop", "length"]);
        assert_eq!(final_plan.final_stop_token_ids, vec![Some(2), None]);
        assert_eq!(final_plan.final_logprob_token_ids, vec![2, 10, 2, 10, 11]);
        assert_eq!(final_plan.final_logprob_values.len(), 5);
        assert_eq!(final_plan.final_logprob_step_offsets, vec![0, 2, 4, 5]);
        assert_eq!(final_plan.final_logprob_beam_offsets, vec![0, 1, 3]);
    }

    #[test]
    fn shadow_state_keeps_valid_masks_aligned_with_parent_ids() {
        let mut family = ShadowBeamFamily::new("root".to_string());
        family.update(
            2,
            vec!["parent-a".to_string(), "parent-b".to_string()],
            vec![-1.0, -2.0],
            vec![vec![20, 10, 20], vec![40, 30]],
        );

        assert_eq!(family.parent_cum_logprobs["parent-a"], -1.0);
        assert_eq!(family.parent_cum_logprobs["parent-b"], -2.0);
        assert_eq!(family.parent_valid_token_ids["parent-a"], vec![10, 20]);
        assert_eq!(family.parent_valid_token_ids["parent-b"], vec![30, 40]);
    }

    #[test]
    fn shadow_history_requires_matching_child_generation() {
        let parent_id = "root-beamproc-2-0".to_string();
        let mut family = ShadowBeamFamily::new("root".to_string());
        family.update(2, vec![parent_id.clone()], vec![-1.0], Vec::new());
        family.record_parent_tokens(parent_id.clone(), vec![100, 10]);
        let mut output = BeamEngineOutput {
            request_id: parent_id,
            new_token_ids: Vec::new(),
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: None,
            candidate_token_start: 0,
            candidate_token_end: 0,
            candidate_logprob_start: 0,
            candidate_logprob_end: 0,
            candidate_rows: 0,
            candidate_cols: 0,
            sampled_rank_start: 0,
            sampled_rank_end: 0,
            beam_context: Some(BeamContext {
                beam_width: Some(2),
                max_tokens: None,
                sampling_max_tokens: None,
                global_max_tokens: None,
                total_decode_steps: Some(3),
                step_index: Some(2),
                is_final_step: false,
                ignore_eos: false,
                include_stop_token: false,
                eos_token_id: None,
                end_token_id: None,
                catalog_strategy: "fill".to_string(),
                return_logprobs: true,
                parent_token_ids: Vec::new(),
                use_shadow_parent_tokens: true,
                prompt_token_count: Some(0),
                client_index: 0,
                priority: 0,
            }),
        };

        assert!(output_has_matching_shadow_history(&family, &output));
        output.beam_context.as_mut().unwrap().step_index = Some(3);
        assert!(!output_has_matching_shadow_history(&family, &output));
        output.beam_context.as_mut().unwrap().step_index = Some(2);
        output.request_id = "root-beamproc-2-stale".to_string();
        assert!(!output_has_matching_shadow_history(&family, &output));
    }

    #[test]
    fn explicit_context_beam_width_wins_over_candidate_columns() {
        let batch = BeamEngineOutputBatch {
            candidate_token_ids: Vec::new(),
            candidate_logprobs: Vec::new(),
            sampled_token_ranks: Vec::new(),
            outputs: vec![BeamEngineOutput {
                request_id: "root".to_string(),
                new_token_ids: Vec::new(),
                finish_reason: None,
                stop_reason: None,
                num_cached_tokens: None,
                candidate_token_start: 0,
                candidate_token_end: 0,
                candidate_logprob_start: 0,
                candidate_logprob_end: 0,
                candidate_rows: 1,
                candidate_cols: 129,
                sampled_rank_start: 0,
                sampled_rank_end: 0,
                beam_context: Some(BeamContext {
                    beam_width: Some(32),
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
                }),
            }],
        };

        assert_eq!(infer_beam_width(&batch), 32);
    }

    #[test]
    fn abort_clears_all_root_scoped_worker_state() {
        let (plan_tx, plan_rx) = crossbeam_channel::unbounded();
        let mut state = BeamWorkerState::new(0, plan_tx, None);
        let root_request_id = "aborted-root".to_string();
        state.shadow_families.insert(
            root_request_id.clone(),
            ShadowBeamFamily::new(root_request_id.clone()),
        );
        state
            .observed_completed
            .insert(root_request_id.clone(), Vec::new());
        state.pending_engine_outputs.insert(
            root_request_id.clone(),
            PendingEngineOutputs::new(0, 0, root_request_id.clone(), "corr".to_string()),
        );

        let family = state.shadow_families.get_mut(&root_request_id).unwrap();
        family.timing_recorded = true;
        family.timing = beam_timing(false);
        assert!(family.timing.is_some());

        let (ack_tx, ack_rx) = crossbeam_channel::bounded(1);
        state.handle(BeamWorkerMsg::Abort {
            root_request_id: root_request_id.clone(),
            ack: Some(ack_tx),
        });

        ack_rx.try_recv().unwrap();
        assert!(!state.shadow_families.contains_key(&root_request_id));
        assert!(!state.observed_completed.contains_key(&root_request_id));
        assert!(!state.pending_engine_outputs.contains_key(&root_request_id));
        assert!(state.cancelled_epochs.contains_key(&root_request_id));

        state.handle(BeamWorkerMsg::BeamStateUpdate {
            root_request_id: root_request_id.clone(),
            beam_width: 2,
            active_parent_ids: vec!["late-parent".to_string()],
            active_parent_cum_logprobs: vec![0.0],
            active_parent_valid_token_ids: Vec::new(),
        });
        let returned = state.handle_engine_output_batch(
            0,
            0,
            root_request_id.clone(),
            "late-correlation".to_string(),
            BeamEngineOutputBatch::with_capacity(0),
        );

        assert_eq!(returned.output_count(), 0);
        assert!(!state.shadow_families.contains_key(&root_request_id));
        assert!(!state.pending_engine_outputs.contains_key(&root_request_id));
        assert!(plan_rx.is_empty());
    }

    #[test]
    fn cancelled_epoch_tombstones_are_bounded_and_expire() {
        let (plan_tx, _plan_rx) = crossbeam_channel::bounded(1);
        let mut state = BeamWorkerState::new(0, plan_tx, None);
        for index in 0..=MAX_CANCELLED_EPOCHS {
            state.tombstone_cancelled_epoch(format!("root-{index}"));
        }
        assert_eq!(state.cancelled_epochs.len(), MAX_CANCELLED_EPOCHS);
        assert!(!state.cancelled_epochs.contains_key("root-0"));

        let expired_root = "expired-root".to_string();
        let expired_at = Instant::now() - CANCELLED_EPOCH_TTL - Duration::from_secs(1);
        state
            .cancelled_epochs
            .insert(expired_root.clone(), expired_at);
        state
            .cancelled_epoch_order
            .push_front((expired_at, expired_root.clone()));
        assert!(!state.is_cancelled_epoch(&expired_root));
    }
}
