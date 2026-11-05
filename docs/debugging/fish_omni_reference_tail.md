# Drama 3 reference-tail investigation

Status: attention precision loss is isolated by controlled comparisons. An
opt-in mixed-precision candidate passes eager reference-tail screening, but an
independent static-graph request still generates reference content. It is not
a complete fix. Earlier fixes
to three implementation contracts were insufficient on their own. Do not treat
passing unit tests as speech-quality acceptance.

## Latest input-only reference control

A later control selected the longest complete reference sentence using audited
speech spans and retained 0.25 seconds of original audio after its aligned end.
For the two-sentence reference, this selected the first sentence and produced
a 3.81-second exact PCM slice. The matching transcript was independently
checked by blind ASR before generation.

On the frozen `33e15506` runtime with leading-reference-ellipsis normalization,
the five historical requests per model had zero fixed reference-phrase
detections with this reference. Complete-body screening passed three of five
FP8 outputs and one of five INT4 outputs; neither Japanese newline case passed.
The same-round 4.4-second first-sentence control passed three of five bodies
on each model. These are small eager-mode screening sets, not a general
reference-tail or speech-quality guarantee. These measurements used offline
reference preparation. The speech adapter now also implements automatic
first-sentence preparation through CPU alignment and a bounded cache; those
earlier measurements do not certify the new serving integration. User text and
generated audio remain outside reference preprocessing.

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

The remaining target-body errors have not been explained by a matched native
high-precision, teacher-forced, layer-by-layer comparison. Their cause remains
unknown. The diagnostic BF16-attention overlay is not enabled by default; the later
opt-in INT8 candidate is described below.
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

## Attention precision and input boundaries: later controlled evidence

The later 70-request comparison separates the original INT4 weights from the
Marlin implementation. With identical request bodies, prompt frames, and
INT4 experts, the INT4 Marlin path produced 38 reference-phrase detections;
independent BF16 multiplication of the same dequantized INT4 attention weights
produced 41. Original parent BF16 attention produced zero. Operator layout
checks passed, and weight error from the RTN export was much larger than the
measured kernel difference. This supports attention weight precision loss as
the cause of the observed reference continuation; it does not establish a
Marlin implementation defect.

Restoring only GDN V was insufficient: a full 70-request INT8-V run still
produced five reference-phrase detections. Restoring all 80 quantized attention
linear operators to freshly quantized W8A16, while retaining INT4 experts,
produced zero reference-phrase detections in the same 70 requests and in a
separate 32-request Japanese/English set with two references and rolling
concurrency eight. Eleven historical outputs and two new outputs still had
CER above 0.35. The same two new failures also occurred with original BF16
attention. These eager diagnostic results do not establish complete speech
accuracy or static-graph deployment acceptance.

The opt-in candidate accepts a matching local parent checkpoint through
`stages[0].engine_extras.hf_overrides.int4_attention_source` in the deployment
YAML. It requires a CUDA TP=1 `drama3-int4` checkpoint and the original
unquantized attention weights. It cannot recover discarded precision from an
INT4 checkpoint alone. This is mixed precision (INT4 experts, W8A16 attention),
not a repaired pure INT4 export. The default path is unchanged when the option
is absent. The static-graph 70-request replay had zero reference hits, but one of the
32 independent requests (a Japanese newline case) generated the reference
transcript. This rejects the candidate as a complete reference-tail fix.
The same 32 requests with exact original BF16 attention also produced one
reference continuation, on the other seed of the newline case. Therefore
attention weight precision alone does not explain this independent failure.
The native FP8 service also failed its five controls with repetitive humming
and token-cap exhaustion, including an ordinary historical target, so it is
not a usable correct-speech baseline. An FP32 recurrent-state control reduced fixed phrase hits in the new set to
zero but still had target-body failures; serial controls also retained an
additional reference-overlap suspect and token-cap exhaustion. Existing CUDA
GDN custom-versus-stock comparisons passed all eight cases, which does not
prove correctness of the real failing trajectories. The same-framework original
FP8 parent also produced reference content in one of the 32 requests. An
independent native backbone/tower with the parent's FP8 experts dequantized to
BF16 still produced reference content on the failing Japanese case, with an
identical prefill embedding and identical first 65 frames of sampling noise.
Dequantization does not recover precision already lost in the FP8 checkpoint.
These results reject an INT4-attention-only explanation for that case; they do
not establish that the parent checkpoint or every shared inference assumption
is correct. Independent FP64 recurrence on the real first-layer GDN prefill
gave relative L2 errors of about 0.22% for its output and state. This does not
validate later layers or pure decode. Root-cause investigation remains open.

Full captured sampler replays on original INT4 weights matched independent
FP64 probability calculations for 703 semantic frames and 6,327 acoustic draws,
including all three natural EOS boundaries. However, an exact request/seed
repeat produced different logits on its first forward, with identical recorded
CPU prompts, reference codes and sampling noise. The parent FP8 control's
166-step repeat was bitwise identical throughout. These profiles also differ
in state dtype and backend settings; the comparison does not isolate weight
precision. Subsequent first-divergence logging verified identical actual GPU frames,
positions, embeddings, first-layer GDN results and expert inputs/routing.
The first differing consumed operator output was the INT4 routed-expert
result: five elements out of 262 by 2048, maximum difference 0.0001221.
Unused slot frames and recurrent state before fresh-state clearing differed,
but consumed GDN results matched. This does not support first-layer stale
state as the explanation.

Standalone replay of that actual expert input reproduced the variation:
20 ordinary repetitions all differed. Eight captures had distinct physical
expert-token orders and GEMM outputs. Holding one order fixed made 20 repeats
bitwise identical; independent canonical ordering also made 20 repeats
identical. This isolates per-layer numerical variation, not reference-tail
causality. Small differences can disappear at later rounding boundaries.
An initial FP32 reference comparison reported about 0.38% relative L2,
but it used different intermediate rounding policies. That number is not a
kernel-error estimate. Matching the actual BF16 SiLU/multiply, down-output,
router and weighted-partial rounding reduced the full-layer reference gap to
about 0.012%. Forty selected down partials matched independent CPU FP64
arithmetic exactly under that policy. Small remaining differences do not by
themselves establish a kernel defect or a whole-model accuracy certificate. A full-AR canonical-order control
then produced bitwise-identical arrays for all 223 semantic/acoustic steps
across the same request repeated after a different request, including every
logit, sampling-noise value and accepted code. Both outputs still spoke the
reference tail. Thus eliminating this numerical nondeterminism is insufficient
to fix the reported failure.

Matching the native BF16-expert policy for FP32 weighted-product/reduction
improved one repeated historical request, but failed three of four predefined
body-quality counterexamples. It is also not an accepted speech fix. A later
single-projection feasibility check re-quantized matching parent BF16 weights
to the same group-128 W4 format using one captured input Gram. Actual Marlin
output error fell from 8.73% to 1.56%; another target sharing the same reference
fell from 8.72% to 2.50%. The original projection output matched its captured
hash. This is neither full-model calibration nor independent speech-quality
acceptance, and it requires parent weights rather than recovering information
from the INT4 checkpoint alone.

A subsequent isolated experiment calibrated all 80 attention projections to
same-format W4 using six new Japanese/English examples and 1,783 actual input
rows per projection. Median Gram-weighted relative output error fell from
10.27% to 0.99%, and all 1,008 untargeted parameter/buffer hashes stayed
unchanged. Teacher weights and Grams were released before actual Marlin
inference; packed candidate bytes were independently verified locally. Two
historical seeds then matched their target bodies, but both predefined H07
counterexamples still failed body verification. This candidate is not an
accepted complete fix. Changing separators or adding speaker tags did not
repair those counterexamples; short ordinary two-sentence controls succeeded.
Calibration statistics alone do not establish end-to-end speech correctness. The full historical
70-request diagnostic found no fixed reference phrase, but seven results had
ASR character error above 0.25 and one hummed until its original token cap.
All full audio and request hashes were verified locally. These are screening
results, not an accepted complete fix or proof of absent reference content.

A subsequent copy-only audit checked actual convolution and recurrent decode
calls in all 30 GDN layers at 11 selected steps of two complete failed
requests: 1,320 independent CPU FP64 comparisons, with full coverage and no
observer errors. This does not cover every generated step. The largest
BF16-rounded recurrent-state relative gap was about 0.0081%; the convolution
gap was about 0.57%. Compiled local IR confirmed BF16 multiplication before
promotion to the FP32 accumulator at all four convolution taps. This matches
an existing [upstream precision report](https://github.com/vllm-project/vllm/issues/52887);
the [official CUDA implementation](https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_update.cu)
promotes operands before multiplication.

An isolated copy with two operand-promotion changes passed prefill/decode
analytic cancellation fixtures with and without SiLU: the old kernel returned
zero where the corrected kernel returned the expected nonzero result. Across
16 real captured activation updates, the maximum BF16-rounded reference gap
fell from 0.357% to 0.000229%, with identical convolution-history updates.
This proves an operator precision correction, not reference-tail causality or
complete speech-quality acceptance. Original installed packages and checkpoint
files remain unchanged during these private diagnostics. In the full same-W4
speech counterfactual, one original historical request still generated the
reference sentence and H07 still hummed. Actual H07 decode audit confirmed
the corrected convolution ran in all 30 GDN layers at the selected steps:
the maximum rounded convolution gap fell from 0.57% to 0.0038%, with identical
history updates. Thus the operator correction alone is insufficient; this
candidate was rejected as a complete reference-tail fix.

Recollecting the six disjoint calibration texts under the corrected
convolution and applying all 80 same-format W4 candidates still failed three
of five original body-quality counterexamples. Two historical seeds had exact
target-body ASR, but R022 and both H07 seeds did not. A separate zero-statistics
counterfactual then used the exact matching-parent BF16 attention matrices,
corrected convolution, and the same W4 experts. All 80 parent projections were
actually called 930 times, while all calibration row/call counts stayed zero.
H07 still generated the reference sentence. Thus the remaining failure cannot
be attributed solely to attention-weight quantization, even after correcting
this convolution defect. A read-only checkpoint comparison found all 675 identically named preserved
tensors and all three tokenizer files byte-identical; the configuration
differed only in its quantization section. This does not verify the values or
origin of quantized expert weights. Expert/runtime behavior and the parent model remain
unresolved; this is not an unquantized whole-model baseline or an accepted fix.

A further five-request control restored the parent FP8 experts, with the same
corrected convolution, BF16 recurrent state and actual prompt/reference hashes.
All 80 loaded BF16 attention matrices matched the preceding teacher hashes.
Three historical target bodies improved, but both H07 bodies still failed;
absence of the fixed reference phrases was not accepted as speech correctness.
Executing the unchanged native and SGLang embedding methods on the actual
265-row prompt and checkpoint weights reproduced the saved embeddings bitwise.
This isolates those embedding formulas, not the complete SGLang runtime.

A whole-backbone control independently decoded the original W4 values for
all 130 attention and 30,720 expert projections, then executed native BF16
GEMMs with native FP32 state. All 30,850 loaded matrix hashes matched the
decoded values; the 675 retained tensors matched an independent checkpoint
audit. One recovered complete output again contained target plus reference
text in blind ASR. Marlin is therefore unnecessary for this symptom; this
does not establish that every Marlin path is numerically correct. The planned
five-request diagnostic stopped at a missing old noise-log fixture after
three complete AR outputs. Those three were fully decoded and preserved;
both H07 requests remain incomplete in this control, and no fix was accepted.

A fixed-prefix factorial control used the same native BF16 implementation
and FP32 caches for four attention/expert weight combinations. Repeated W4
baseline logits were bitwise equal, and all 675 non-target parameters stayed
unchanged. At one recorded reference-tail prefix, replacing only attention
weights raised the conditional EOS probability after top-k/top-p/RAS from
0.001679 to 0.823401; replacing only experts yielded 0.003132. At a separate
correct W4 prefix, parent attention instead lowered the stopping probability.
These selected-prefix results identify attention-weight sensitivity, not a
universal repair or free-generation quality improvement. Parent expert values
were released FP8 weights dequantized to BF16, not a pre-FP8 gold standard.

The calibration observer initially collected only the first 65 forwards per
request. A later audit recorded one extra forward after the first natural
EOS for each of six calibration requests. Excluding these extra computations
allowed complete calibration through all six EOS predictions, with 842 valid
forwards and identical teacher PCM. Recalibrating all 80 W4 targets reduced
the weighted error, but three of five original body-quality examples still
failed ASR screening. All complete audio was retained; neither the observer
correction nor zero detected reference phrases established a speech fix.

A CPU comparison covered all 130 attention and 30,720 expert matrices. Each
effective W4 matrix matched the preceding independent-load hash, and each
parent matrix matched the preceding counterfactual hash. Overall weight
relative L2 was 10.37%, with per-matrix values from 8.32% to 17.52%; no zero
parent matrix had nonzero W4 values. Local checks independently verified all
paired hashes and norm identities. This did not reveal an obvious whole-matrix
mismatch, but weight norms do not establish speech quality or identify the
cause of a missed EOS. The released parent experts were still FP8 values.

Related upstream reports describe released-FP8 continuation failures in
[native inference](https://github.com/fishaudio/fish-speech/issues/1357) and
[voice cloning](https://github.com/fishaudio/fish-speech/issues/1356). They do not
establish that these local failures share a cause or provide an accepted fix.

Input auditing found no historical reference/target separator leakage: 92
saved prompt matrices matched native and upstream builders elementwise. A
further 124 tokenizer variants matched native acceptance and prompt frames.
An independent missing-validation issue allowed literal structural markers
in user/reference text; the candidate rejects those markers and malformed
speaker tags, using the native Unicode whitespace rules. This prevents prompt
structure injection but does not explain the historical reference tails.

Additional unit regressions:

```bash
pytest -q tests/model_executor/models/fish_omni/test_fish_omni_prompt_boundaries.py \
  tests/model_executor/models/fish_omni/test_fish_omni_attention_recovery.py
pytest -q -m core_model --run-level core_model \
  tests/model_executor/models/fish_omni/test_fish_omni_attention_recovery_cuda.py
```

The CUDA tests require a supported Marlin GPU and matching vLLM/PyTorch
installation, including the compiled custom operators; no speech-model weights
are needed. They compare actual INT8 Marlin outputs with independent FP64
multiplication and replay a CUDA graph with changed input. Unit results do not
replace end-to-end listening and quality checks with the matching parent.
