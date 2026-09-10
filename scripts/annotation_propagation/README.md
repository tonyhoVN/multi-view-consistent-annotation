# Annotation propagation

This tool transfers object masks from captured view 0 through a scan in the
exact order stored in `manifest_<run>.json`. It does not use ROS and does not
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

## Run

Validate paths, route ordering, calibration, and simulation seeds without
loading neural-network models:

```bash
scripts/annotation_propagation/run_transfer.sh \
  scan_output/manifest_run_3.json --validate-only
```

Run propagation for all simulation seed objects:

```bash
scripts/annotation_propagation/run_transfer.sh \
  scan_output/manifest_run_3.json
```

Select objects or override thresholds:

```bash
scripts/annotation_propagation/run_transfer.sh \
  scan_output/manifest_run_3.json \
  --objects scissors mustard_bottle \
  --minimum-projected-points 10 \
  --maximum-center-distance 0.05 \
  --minimum-area-ratio 0.2 \
  --maximum-area-ratio 5.0
```

The default output is `scan_output/transfer_segment_<suffix>`. Each frame uses
the Isaac-compatible layout:

```text
segment_<path_index>/<camera_frame>/capture_000000/
  manifest.json
  <object-instance>.png
```

The output also contains `transfer_manifest.json` and `obj_scenes.json`.
Existing output at the selected destination is replaced at the start of a run.
Use `--output-dir` to choose another destination.

## Naive Grounding DINO + SAM baseline

Annotate every frame independently, without TF, depth, or neighboring masks:

```bash
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/annotate_naive_vlm.py \
  scan_output/manifest_run_3.json
```

Simulation runs obtain only the class/instance list from the initial Isaac
segmentation manifest; those masks are not used for inference. For real runs,
provide the scene classes with `--objects`. The default output directory is
`scan_output/naive_vlm_segment_<suffix>`.

Choose the detector execution mode:

```bash
# One combined multi-class DINO pass and one batched SAM pass per image.
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/annotate_naive_vlm.py \
  scan_output/manifest_run_1.json \
  --detection-mode zeroshot \
  --output-dir scan_output/naive_vlm_zeroshot_run_1

# One independent DINO-to-SAM query per object per image.
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/annotate_naive_vlm.py \
  scan_output/manifest_run_1.json \
  --detection-mode multi-shot \
  --output-dir scan_output/naive_vlm_multi_shot_run_1
```

`zeroshot` is the default and is normally faster. Both annotation methods save
total runtime, annotation runtime, seconds per frame, and seconds per
object-frame in their summary manifests. Timing includes lazy model loading and
mask writing.

Evaluate this baseline using the same evaluator:

```bash
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/evaluate_segmentation_map.py \
  scan_output/manifest_run_1.json \
  --predictions scan_output/naive_vlm_segment_run_1 \
  --output scan_output/naive_vlm_segment_run_1/map_report.json
```

## Evaluate segmentation mAP

Compare one transferred run against the Isaac masks referenced by its scan
manifest:

```bash
conda run --no-capture-output -n vla \
  python scripts/annotation_propagation/evaluate_segmentation_map.py \
  scan_output/manifest_run_1.json \
  --predictions scan_output/transfer_segment_run_1 \
  --output scan_output/transfer_segment_run_1/map_report.json
```

The evaluator reports standard mask mAP at IoU 0.50 and the mean over IoU
thresholds 0.50:0.05:0.95. AP is calculated independently for every object
class and macro-averaged across all classes present in ground truth. Missing
predictions count as false negatives. Prediction confidence is read from
`confidence` or `score`; transferred masks without either field use 1.0.
