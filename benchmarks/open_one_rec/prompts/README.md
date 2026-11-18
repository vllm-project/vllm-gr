# OpenOneRec judge prompts

The `.txt` files in this directory are evaluation material, not project code.
They are the LLM-as-Judge prompts of the OpenOneRec benchmark, taken
unchanged from
[Kuaishou-OneRec/OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec)
at revision `ae668a6a30bd71bde7a3f459fe0ac958182ce1d2` (Apache-2.0). Each
file has a `.license` file next to it with the upstream copyright, and the
root `NOTICE` names the upstream constant each file came from.

The prompts are in Chinese, like the OpenOneRec evaluation data they score.
Translating them would change what the judge model sees and therefore the
benchmark scores, so they are kept as written.

| File | Used by | Purpose |
| --- | --- | --- |
| `item_understand_wip_extraction.txt` | `tasks/item_understand/utils.py` | Extract weighted information points (WIPs) from a video summary |
| `item_understand_wip_matching.txt` | `tasks/item_understand/utils.py` | Match model WIPs against ground-truth WIPs |
| `rec_reason_evaluation.txt` | `tasks/rec_reason/utils.py` | Score a generated recommendation reason from 1 to 5 |

The texts are `str.format` templates: `{}` is filled by the evaluator and
`{{` / `}}` are literal braces. Load them with
`benchmarks.open_one_rec.prompts.load_prompt(name)`. Every upstream prompt
ends with one space after its final colon; the files store a trailing newline
there instead, so whitespace hooks do not change them, and `load_prompt`
restores the space.
