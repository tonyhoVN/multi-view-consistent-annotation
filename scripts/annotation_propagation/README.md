# Annotation propagation

This tool transfers object masks from captured view 0 through a scan in the
exact order stored in `<run>/manifest.json`. It does not use ROS and does not
infer correspondence by sorting filenames.

For simulation scans, the default `auto` seed mode reads every object and mask
from view 0's Isaac segmentation manifest. For real scans, it asks Qwen3-VL for
the initial object boxes and converts them to masks with SAM.

For every class and subsequent manifest-ordered frame, propagation follows the
geometric algorithm in `TASK.md`:

1. Back-project the previous accepted mask and depth into a base-frame cloud.
2. Project that cloud into the current camera and require at least `tau_proj`
   pixels.
3. Prompt SAM using the median projected pixel.
4. Validate the mask-area ratio against the largest accepted mask.
5. If area validation fails, use the class name with Grounding DINO and segment
   its strongest box with SAM.
6. Back-project the candidate and reject it when its 3-D center moves more than
   `tau_dist` from the previous accepted cloud.
7. Save accepted masks and use them as the geometric prior for the next frame.

## Run propagate annotation pipeline

Validate paths, route ordering, calibration, and simulation seeds without
loading neural-network models:

```bash
scripts/annotation_propagation/run_transfer.sh \
  scan_output/run_3/manifest.json --validate-only
```

Run propagation for all simulation seed objects:

```bash
scripts/annotation_propagation/run_transfer.sh \
  scan_output/run_3/manifest.json
```

Select objects or override thresholds:

```bash
scripts/annotation_propagation/run_transfer.sh \
  scan_output/run_3/manifest.json \
  --objects scissors mustard_bottle \
  --minimum-projected-points 10 \
  --maximum-center-distance 0.05 \
  --minimum-area-ratio 0.2 \
  --maximum-area-ratio 5.0
```

The default output is `scan_output/<run>/transfer_segment`. Each frame uses
the Isaac-compatible layout:

```text
segment_<path_index>/<camera_frame>/capture_000000/
  manifest.json
  <object-instance>.png
```

The output also contains `transfer_manifest.json` and `obj_scenes.json`.
Existing output at the selected destination is replaced at the start of a run.
Use `--output-dir` to choose another destination.

The spatial drift filter statistically removes isolated depth points and trims
the table plane along a user-defined table-local +Z axis. For a horizontal
table at base-frame Z = -0.10 m:

```bash
scripts/annotation_propagation/run_transfer.sh \
  scan_output/run_3/manifest.json \
  --table-origin 0 0 -0.20 \
  --table-z-axis 0 0 1 \
  --table-clearance 0.003
```

Add `--save-visualizations` to write annotated image copies under
`transfer_segment/visualizations`. Each copy shows the geometric point prompt,
mask overlay, boundary box, object label, and segmentation source. Rendering
happens after the reported annotation runtime is stopped.

`run_transfer.sh` runs both trajectory post-processing and annotation inside
the `vla` Conda environment, then applies the table, outlier, and visualization
settings shown above. Explicit options appended to the command override its
table and outlier defaults.

### Batch transfer across runs

`run_all_transfer.sh` runs only the mask transfer algorithm (post-processing
plus propagation via `run_transfer.sh`) across a range of runs — it does not
run the naive VLM baselines or evaluation:

```bash
scripts/annotation_propagation/run_all_transfer.sh [start] [end] [--stop-on-error] [-- transfer options ...]
```

`start`/`end` default to `1`/`24`. For each run `<n>` it expects
`scan_output/run_<n>/manifest.json` and produces `run_<n>/transfer_segment` as
described above. A missing manifest or a failing run is logged and skipped so
the batch keeps going; pass `--stop-on-error` to halt on the first failure
instead. Anything after a literal `--` is forwarded to `run_transfer.sh` for
every run, e.g. to select objects or override thresholds for the whole batch.

## Run naive Grounding DINO + SAM baseline

Annotate every frame independently, without TF, depth, or neighboring masks:

```bash
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/annotate_naive_vlm.py \
  scan_output/run_3/manifest.json
```

Simulation runs obtain only the class/instance list from the initial Isaac
segmentation manifest; those masks are not used for inference. For real runs,
provide the scene classes with `--objects`. The default output directory is
`scan_output/<run>/baseline_segment/naive_vlm_<detection_mode>`.
Add `--save-visualizations` to save mask overlays and boundary boxes under that
method's `visualizations/` directory. The shared transfer renderer is reused,
but no point is drawn because this baseline has no geometric point prompt.

Candidate filtering is enabled by default: only the highest-confidence box is
segmented, the largest connected component is retained, and very small masks
are rejected. To accept every detector box for each class and union all SAM
masks into one class segmentation, use:

```bash
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/annotate_naive_vlm.py \
  scan_output/run_3/manifest.json \
  --detection-mode zeroshot \
  --no-filter-candidates \
  --save-visualizations
```

No-filter outputs default to a separate directory such as
`baseline_segment/naive_vlm_zeroshot_no_filter`, preserving filtered results.

## Run SAM 2 video baseline

Track all objects jointly from the saved first-view masks through the selected
scan route:

```bash
scripts/annotation_propagation/run_sam2_video.sh \
  scan_output/run_3/manifest.json \
  --overwrite \
  --save-visualizations
```

The default model is `facebook/sam2.1-hiera-small`, and output is written to
`baseline_segment/sam2_video`. This baseline uses RGB frame order and first-view
mask prompts only; it does not use camera TF, depth, Grounding DINO, or the
transfer method's spatial/area filters. Use `--objects` to track a subset of
first-view classes. The output has the standard per-frame segmentation manifest
layout and can be evaluated with `evaluate_segmentation_map.py`.

## Grounding DINO alias experiment

Run the propagation method and both naive baselines for one existing scan using
the alternative Grounding DINO prompts in
`alias_experiment/object_aliases.yaml`, then evaluate every result immediately:

```bash
scripts/annotation_propagation/alias_experiment/run_alias_evaluation.sh 3
```

The wrapper enables visualization and uses table origin `(0, 0, -0.20)`, table
+Z `(0, 0, 1)`, `0.003` m clearance, 20 outlier neighbors, outlier standard
ratio `2.0`, maximum center distance `0.1` m, and accepted mask-area ratio
`[0.2, 2.5]`. Append the corresponding option to override any default.

The integer is the run index, so this example reads
`scan_output/run_3/manifest.json`. Pass a second integer to evaluate an
inclusive range of runs instead of a single one:

```bash
scripts/annotation_propagation/alias_experiment/run_alias_evaluation.sh 3 6
```

which evaluates runs 3 through 6. A run whose manifest is missing is logged
and skipped; the rest of the range still runs. It does not post-process the
scan and does not rerun or overwrite canonical annotations. Results are
isolated per run under:

```text
scan_output/run_3/alias_segment/
  transfer_segment/
  naive_vlm_zeroshot/
  naive_vlm_multi_shot/
```

Each method directory contains its annotations, visualization copies, runtime
manifest, and `map_report.json`. Saved object labels remain canonical for a
direct comparison with Isaac ground truth; only the text sent to Grounding
DINO is replaced by the alias. The exact mapping is copied into each runtime
manifest for reproducibility.

Additional runner options are forwarded to all three experiments. For example,
to union all Grounding DINO + SAM proposals in both naive baselines:

```bash
scripts/annotation_propagation/alias_experiment/run_alias_evaluation.sh 3 6 \
  --no-filter-candidates
```

The two no-filter baselines use directories ending in `_no_filter`, so this
command does not overwrite the filtered alias results. The transfer method is
unaffected by this baseline-only option.

### Evaluation-only batch run for the alias experiment

`run_all_evaluations_alias.sh` only re-evaluates alias predictions that
already exist — it does not run transfer propagation or naive VLM
annotation. Use it to re-score a range of runs after alias generation has
already completed:

```bash
scripts/annotation_propagation/alias_experiment/run_all_evaluations_alias.sh [start] [end] [--stop-on-error]
```

`start`/`end` default to `1`/`24`. For each run `<n>` it evaluates whichever
of these prediction directories exist under `scan_output/run_<n>/alias_segment/`:

```text
transfer_segment
naive_vlm_zeroshot
naive_vlm_multi_shot
naive_vlm_zeroshot_no_filter
naive_vlm_multi_shot_no_filter
```

writing each `map_report.json` in place. A missing manifest is logged and
skipped; a missing prediction directory for one method is skipped without
failing the run (not every run has every method generated). Pass
`--stop-on-error` to halt on the first evaluation failure instead of
continuing through the range.

Use `--aliases path/to/aliases.yaml` to test a different mapping, or run only
one method directly (still inside the required Conda environment):

```bash
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/alias_experiment/annotate_evaluate_aliases.py \
  scan_output/run_3/manifest.json \
  --method naive_vlm_zeroshot
```

Choose the detector execution mode:

```bash
# One combined multi-class DINO pass and one batched SAM pass per image.
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/annotate_naive_vlm.py \
  scan_output/run_1/manifest.json \
  --detection-mode zeroshot \
  --output-dir scan_output/run_1/baseline_segment/naive_vlm_zeroshot

# One independent DINO-to-SAM query per object per image.
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/annotate_naive_vlm.py \
  scan_output/run_1/manifest.json \
  --detection-mode multi-shot \
  --output-dir scan_output/run_1/baseline_segment/naive_vlm_multi_shot
```

`zeroshot` is the default and is normally faster. Both annotation methods save
total runtime, annotation runtime, seconds per frame, and seconds per
object-frame in their summary manifests. Timing includes lazy model loading and
mask writing.

Evaluate this baseline using the same evaluator:

```bash
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/evaluate_segmentation_map.py \
  scan_output/run_1/manifest.json \
  --predictions scan_output/run_1/baseline_segment/naive_vlm_zeroshot \
  --output scan_output/run_1/baseline_segment/naive_vlm_zeroshot/map_report.json
```

## Evaluate segmentation mAP

Compare one transferred run against the Isaac masks referenced by its scan
manifest:

```bash
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/evaluate_segmentation_map.py \
  scan_output/run_1/manifest.json \
  --predictions scan_output/run_1/transfer_segment \
  --output scan_output/run_1/transfer_segment/map_report.json
```

The evaluator reports standard mask mAP at IoU 0.50 and the mean over IoU
thresholds 0.50:0.05:0.95. By default, it evaluates only object classes visible
in the first route capture (normally Ready view/sample 0), matching the class
set given to the naive baseline. AP is calculated independently for those
classes over the full route and macro-averaged. Use `--objects` for an explicit
subset. Missing predictions count as false negatives. Prediction confidence is
read from `confidence` or `score`; transferred masks without either field use
1.0.

## Evaluation-only batch run

`run_evaluation.sh` only evaluates predictions that already exist — it does
not run transfer propagation or naive VLM annotation. Use it to re-score a
range of runs after generation has already completed (e.g. to compare against
updated ground truth, or after tuning nothing but re-scoring):

```bash
scripts/annotation_propagation/run_evaluation.sh [start] [end] [--stop-on-error]
```

`start`/`end` default to `1`/`24`. For each run `<n>` it evaluates whichever
of these prediction directories exist under `scan_output/run_<n>/`:

```text
transfer_segment
baseline_segment/naive_vlm_zeroshot
baseline_segment/naive_vlm_multi_shot
```

writing each `map_report.json` in place. A missing manifest or missing
prediction directory is logged and skipped so the batch keeps going; pass
`--stop-on-error` to halt on the first failure instead.

## Automated annotation + evaluation

`run_all_evaluations.sh` runs the full pipeline — transfer, both naive VLM
baselines, and evaluation of all three — across a range of runs:

```bash
scripts/annotation_propagation/run_all_evaluations.sh [start] [end] [--stop-on-error]
```

`start`/`end` default to `1`/`24`. For each run `<n>` it expects
`scan_output/run_<n>/manifest.json` and produces:

```text
scan_output/run_<n>/transfer_segment/map_report.json
scan_output/run_<n>/baseline_segment/naive_vlm_zeroshot/map_report.json
scan_output/run_<n>/baseline_segment/naive_vlm_multi_shot/map_report.json
```

A run whose manifest is missing or whose steps fail is logged and skipped so
the rest of the batch keeps going; pass `--stop-on-error` to halt on the first
failure instead. After every conda-run step it prints `nvidia-smi` memory
usage and pauses briefly so GPU memory is confirmed released before the next
step starts — each step is its own process, so the OS reclaims its CUDA
context on exit regardless.

## Summarizing results

`summarize_results.py` averages mAP and runtime across a run range for all
three methods, reading each `map_report.json` and its matching
`transfer_manifest.json` / `naive_vlm_manifest.json`:

```bash
python3 scripts/annotation_propagation/summarize_results.py [start] [end] \
  [--scan-dir scan_output] [--output report.json]
```

It prints, per method, the average `mAP50`, average `mAP50_95`, average
`runtime.total_seconds`, how many runs contributed data, and which runs in
the range were missing outputs. Pass `--output` to also write the same
aggregation as JSON.

To summarize the Grounding DINO alias experiment instead, use
`alias_experiment/summarize_alias_results.py`. It auto-discovers whatever
method subdirectories actually exist under each run's `alias_segment/`
(`transfer_segment`, `naive_vlm_zeroshot`, `naive_vlm_multi_shot`, and their
`_no_filter` variants) rather than assuming a fixed set, so it works whether
a run has the full experiment or only some methods generated:

```bash
python3 scripts/annotation_propagation/alias_experiment/summarize_alias_results.py [start] [end] \
  [--scan-dir scan_output] [--output report.json]
```
