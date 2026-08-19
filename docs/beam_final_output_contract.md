# Beam final output contract

The final Beam output is independent from the candidate width required to run
Beam search. `n` continues to select `beam_width`; changing that mapping is out
of scope for this contract.

## Audited callers

- OpenOneRec online requests do not set OpenAI `logprobs` or `top_logprobs`.
  They consume decoded choices and their ordering; cumulative Beam scores remain
  available internally on `CompletionOutput`.
- Serving and offline Beam regression tests historically expect the number of
  results to equal `beam_width`. Therefore `result_width` defaults to
  `beam_width`.
- Explicit OpenAI `logprobs=true` callers require a compatibility result. With
  no `top_logprobs` they receive the selected token logprob; with
  `top_logprobs>0` they receive the selected token plus the requested top-k.
- Repository debugging/evaluation code that reads complete candidate histories
  uses explicit logprob-bearing outputs or the legacy mixed-version contract.

No prompt, token, request ID, resource digest, or exact width is used as a
Prometheus label.

## Contract

```python
BeamOutputSpec(
    result_width: int,
    logprobs_mode: Literal["none", "selected", "top_k"],
    top_logprobs: int = 0,
)
```

- `none` returns final token IDs, cumulative score and finish/stop reasons in a
  versioned `beam_final_lite` payload. It never reconstructs candidate history.
- `selected` reconstructs and returns only each selected token's logprob.
- `top_k` reconstructs the compatibility history and projects it to the
  requested candidate count.
- Unsupported modes, widths and mode/top-k combinations fail during request
  validation.

The Lite wire format is `{type: "beam_final_lite", version: 1, ...}`. Unknown
types or versions are rejected explicitly.
