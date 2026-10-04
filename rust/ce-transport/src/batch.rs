//! Prepare the existing v6 protocol from owned CE snapshots, on the TX worker.
use crate::{Frames, MAX_BYTES};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub(crate) const MAX_OUTPUTS: usize = 65536;

#[derive(Deserialize)]
struct Snapshot {
    engine_index: i64,
    client_index: i64,
    correlation_base: u64,
    vocab_offset: Option<i32>,
    vocab_size: Option<i32>,
    outputs: Vec<Output>,
}

#[derive(Deserialize)]
struct Output {
    request_id: String,
    child_ids: Option<Vec<String>>,
    context: Option<Value>,
    finish_reason: Value,
    stop_reason: Value,
    num_cached_tokens: Option<u64>,
    rows: usize,
    cols: usize,
}

// Same twelve positions as Python's v6 output rows.
#[derive(Serialize)]
struct Row<'a>(
    &'a str,
    usize,
    usize,
    &'a Value,
    &'a Value,
    Option<u64>,
    usize,
    usize,
    usize,
    usize,
    usize,
    Option<usize>,
);

#[derive(Serialize)]
struct Header<'a> {
    version: u8,
    engine_index: i64,
    client_index: i64,
    correlation_base: u64,
    contexts: Vec<&'a Value>,
    outputs: Vec<Row<'a>>,
}

pub(crate) fn encode(metadata: &[u8], buffers: Frames) -> Result<Frames, String> {
    let snapshot: Snapshot = rmp_serde::from_slice(metadata).map_err(|e| e.to_string())?;
    if snapshot.outputs.is_empty()
        || snapshot.outputs.len() > MAX_OUTPUTS
        || buffers.len() != snapshot.outputs.len() * 4
    {
        return Err("batch output/buffer count mismatch".into());
    }
    let mut header = Header {
        version: 6,
        engine_index: snapshot.engine_index,
        client_index: snapshot.client_index,
        correlation_base: snapshot.correlation_base,
        contexts: Vec::new(),
        outputs: Vec::new(),
    };
    let mut context_indices = HashMap::new();
    let mut tokens = Vec::new();
    let mut logprobs = Vec::new();
    let mut ranks = Vec::new();
    let mut new_tokens = Vec::new();
    let mut buffers = buffers.into_iter();
    for output in &snapshot.outputs {
        let token_block = buffers.next().unwrap();
        let logprob_block = buffers.next().unwrap();
        let rank_block = buffers.next().unwrap();
        let new_block = buffers.next().unwrap();
        let expected = output
            .rows
            .checked_mul(output.cols)
            .and_then(|n| n.checked_mul(4))
            .ok_or("batch dimensions overflow")?;
        if token_block.len() != expected
            || logprob_block.len() != expected
            || !rank_block.len().is_multiple_of(4)
            || !new_block.len().is_multiple_of(4)
        {
            return Err("batch numeric dimensions mismatch".into());
        }
        let context_index = if let Some(context) = &output.context {
            // Grouped roots deliberately receive a distinct shared context, as in Python.
            let key = rmp_serde::to_vec_named(context).map_err(|e| e.to_string())?;
            if output.child_ids.is_none() && context_indices.contains_key(&key) {
                Some(context_indices[&key])
            } else {
                let index = header.contexts.len();
                header.contexts.push(context);
                if output.child_ids.is_none() {
                    context_indices.insert(key, index);
                }
                Some(index)
            }
        } else {
            None
        };
        let token_offset = tokens.len() / 4;
        let rank_offset = ranks.len() / 4;
        let new_offset = new_tokens.len() / 4;
        if let Some(children) = &output.child_ids {
            if children.is_empty()
                || children.len() > MAX_OUTPUTS
                || output.rows != children.len()
                || !rank_block.is_empty()
                || !new_block.is_empty()
            {
                return Err("invalid grouped batch dimensions".into());
            }
            for (index, child) in children.iter().enumerate() {
                header.outputs.push(Row(
                    child,
                    new_offset,
                    0,
                    &output.finish_reason,
                    &output.stop_reason,
                    output.num_cached_tokens,
                    token_offset + index * output.cols,
                    1,
                    output.cols,
                    rank_offset + index,
                    1,
                    context_index,
                ));
            }
            ranks.resize(ranks.len() + children.len() * 4, 0);
        } else {
            let has_candidates = output.rows != 0;
            header.outputs.push(Row(
                &output.request_id,
                new_offset,
                new_block.len() / 4,
                &output.finish_reason,
                &output.stop_reason,
                output.num_cached_tokens,
                if has_candidates { token_offset } else { 0 },
                output.rows,
                output.cols,
                if has_candidates { rank_offset } else { 0 },
                rank_block.len() / 4,
                context_index,
            ));
            ranks.extend(rank_block);
            new_tokens.extend(new_block);
        }
        if header.outputs.len() > MAX_OUTPUTS {
            return Err("too many batch rows".into());
        }
        tokens.extend(token_block);
        logprobs.extend(logprob_block);
    }
    if let (Some(offset), Some(size)) = (snapshot.vocab_offset, snapshot.vocab_size)
        && offset != 0
        && size > 0
    {
        for bytes in tokens.chunks_exact_mut(4) {
            let token = i32::from_le_bytes(bytes.try_into().unwrap());
            if token >= 0 && token < size {
                bytes.copy_from_slice(&token.wrapping_add(offset).to_le_bytes());
            }
        }
    }
    let metadata = rmp_serde::to_vec_named(&header).map_err(|e| e.to_string())?;
    let frames = vec![
        b"ENGINE_OUTPUT_BATCH".to_vec(),
        metadata,
        tokens,
        logprobs,
        ranks,
        new_tokens,
    ];
    if frames.iter().map(Vec::len).sum::<usize>() > MAX_BYTES {
        return Err("prepared batch exceeds CE transport limits".into());
    }
    Ok(frames)
}
