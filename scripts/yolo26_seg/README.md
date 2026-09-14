# YOLO26-seg pseudo-label experiments

This folder trains one `yolo26n-seg.pt` model per annotation method and validates
every model against the saved Isaac ground truth. Training and validation runs
must be disjoint.

## Annotation sources

`--source transfer` reads `run_N/transfer_segment`. Other names are read from
`run_N/baseline_segment/<source>`, for example:

- `naive_vlm_zeroshot`
- `naive_vlm_multi_shot`
- `naive_vlm_zeroshot_no_filter`
- `naive_vlm_multi_shot_no_filter`

## Train all standard experiments

```bash
scripts/yolo26_seg/run_training.sh 1 35 36 44 \
  transfer naive_vlm_zeroshot naive_vlm_multi_shot
```

The first range is used for training and the second for validation. The command
runs inside the `vla` Conda environment. On first use, Ultralytics downloads
`yolo26n-seg.pt` to the current Ultralytics cache.

Training does not invoke validation. After training, validate the selected models:

```bash
scripts/yolo26_seg/run_validation.sh \
  transfer naive_vlm_zeroshot naive_vlm_multi_shot
```

## Prepare and train one method manually

```bash
conda run --no-capture-output -n vla python \
  scripts/yolo26_seg/prepare_dataset.py \
  --train-runs 1-35 --val-runs 36-44 \
  --source transfer \
  --output scan_output/yolo26_seg_datasets/transfer

conda run --no-capture-output -n vla python \
  scripts/yolo26_seg/train.py \
  scan_output/yolo26_seg_datasets/transfer/dataset.yaml \
  --name transfer --device 0
```

Training does not run validation. Validate the saved checkpoint separately:

```bash
conda run --no-capture-output -n vla python \
  scripts/yolo26_seg/validate.py \
  scan_output/yolo26_seg_datasets/transfer/dataset.yaml \
  scan_output/yolo26_seg/transfer/weights/best.pt \
  --project scan_output/yolo26_seg/transfer \
  --output scan_output/yolo26_seg/transfer/ground_truth_validation.json \
  --device 0 --exist-ok
```

Generated datasets contain relative symbolic links to the original color images,
YOLO polygon labels, `dataset.yaml`, and provenance in `metadata.json`. Validation
labels always come from `save_segment`, regardless of the training source.

Results are written under `scan_output/yolo26_seg/<source>/`. The compact report
`ground_truth_validation.json` contains segmentation and box mAP50/mAP50-95.
After `run_validation.sh` finishes, all method reports are also combined into
`scan_output/yolo26_seg_validation_summary.json`. Regenerate it manually with:

```bash
python3 scripts/yolo26_seg/summarize_validation.py
```
