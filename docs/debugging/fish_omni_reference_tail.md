# Drama 3 reference-tail investigation

Status: unresolved. The changes on this branch correct three implementation
contracts; they are not a complete fix for reference speech being generated
after the requested text. Do not treat passing unit tests as speech-quality
acceptance.

## Confirmed implementation problems

- The reference DAC codec was constructed under the surrounding AR model's
  BF16/FP16 default dtype. Loading an FP32 checkpoint into those parameters
  rounded the weights before a later FP32 conversion. Constructing the codec
  in FP32 preserves the checkpoint until the requested runtime conversion.
- The semantic candidate set excluded code 0 even though it belongs to the
  complete semantic codebook. The candidate set now includes it.
- Semantic nucleus filtering excluded the candidate that first crosses the
  cumulative top-p threshold. That candidate can be EOS. Both semantic draws
  now retain the crossing candidate; acoustic-head filtering stays unchanged.

Recorded failing trajectories did not show an already sampled EOS being
ignored. Reference content was present in accepted AR codes before EOS;
decoding those codes independently reproduced the extra speech. The observed
problem is therefore not explained by an EOS index offset or output-side
concatenation alone. Correcting EOS eligibility does not by itself eliminate
reference-tail generation.

## Historical controlled measurements

Each cell below used the same 64 request bodies, including repeated seeds,
unseen seeds, different target texts, and C4/C8 concurrency checks. Routed
experts remained INT4. BF16 attention means replacing the quantized attention
projections with their original BF16 weights in a diagnostic runtime.

| Attention projections | Mamba recurrent state | Reference phrases detected | 400-frame limit reached | CER above 0.35 |
| --- | --- | --- | --- | --- |
| INT4 | BF16 | 26/64 | 0/64 | 36/64 |
| INT4 | FP32 | 34/64 | 1/64 | 44/64 |
| BF16 | BF16 | 0/64 | 1/64 | 9/64 |
| BF16 | FP32 | 0/64 | 3/64 | 16/64 |

These observations narrow the reference-tail problem to the INT4 attention
path under the tested conditions. They do not fully separate weight precision
from kernel implementation differences. Changing only the Mamba state dtype
was insufficient. Restoring BF16 attention still produced humming, loops, or
incorrect target speech, so no tested profile passed complete quality
acceptance.

The diagnostic runs used an RTX PRO 6000 GPU (SM120), vLLM 0.32.0, and PyTorch
2.13 with CUDA 13. ASR screening used Whisper-small in FP32, beam size 5, without
supplying the expected or reference transcript. ASR phrase detection and CER
are screening signals, not manual listening or a universal correctness
claim. These are historical measurements from an isolated diagnostic export,
not a CI result or an accuracy claim for every later branch head.

## Validation and remaining work

The targeted regression suite previously passed 21 tests: 9 CPU logic/loader
tests and 12 CUDA sampling tests. Before the changes, three codec precision
regressions and two selected EOS-boundary CUDA regressions failed.

The remaining AR content errors have not been explained by a matched native
high-precision, teacher-forced, layer-by-layer comparison. Their cause remains
unknown. The diagnostic BF16-attention overlay is not enabled by this patch.
There is no audio trimming, forced-duration EOS, or ASR-based output rewriting.

Run from the repository root with matching vLLM, PyTorch, Triton, and pytest
installed. The sampling module requires a supported CUDA GPU; these unit tests
do not need model weights.

```bash
files=(tests/model_executor/models/fish_omni/test_fish_omni_{candidates,nucleus,sampling}.py
       tests/model_executor/models/fish_speech/test_dac_encoder_precision.py)
pytest -q "${files[@]}"
pytest -q -m core_model --run-level core_model "${files[@]}"
```

AI assistance: OpenAI Codex assisted with investigation, implementation,
regression tests, and this note. This branch has not been submitted as an
upstream PR or certified as a complete speech-quality fix.
