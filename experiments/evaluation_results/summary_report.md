# WinCLIP zero-shot evaluation summary

Evaluation used pretrained WinCLIP weights in zero-shot mode (`k_shot=0`). No model training was performed. Results below summarize the existing per-category files.

## Results

| Category | Image AUROC | Pixel AUROC | Images | Normal | Defective | Mean inference (s) |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 1.0000 | 0.9347 | 20 | 10 | 10 | 10.8466 |
| cable | 0.7000 | 0.5180 | 20 | 10 | 10 | 9.9821 |
| capsule | 0.6400 | 0.9321 | 20 | 10 | 10 | 10.6402 |
| carpet | 0.9700 | 0.9694 | 20 | 10 | 10 | 10.4546 |
| grid | 1.0000 | 0.8924 | 20 | 10 | 10 | 9.5603 |
| hazelnut | 0.8800 | 0.9602 | 20 | 10 | 10 | 9.8298 |
| leather | 1.0000 | 0.9738 | 20 | 10 | 10 | 28.6000 |
| metal_nut | 0.9100 | 0.4680 | 20 | 10 | 10 | 21.5856 |
| pill | 0.8700 | 0.7939 | 20 | 10 | 10 | 11.1983 |
| screw | 0.6800 | 0.9014 | 20 | 10 | 10 | 11.1195 |
| tile | 1.0000 | 0.9455 | 20 | 10 | 10 | 18.4168 |
| toothbrush | 1.0000 | 0.8941 | 20 | 10 | 10 | 31.7377 |
| transistor | 0.7900 | 0.8828 | 20 | 10 | 10 | 4030.4256 |
| wood | 1.0000 | 0.9371 | 20 | 10 | 10 | 6.8529 |
| zipper | 0.9700 | 0.9377 | 20 | 10 | 10 | 7.1936 |

## Overall observations

- Valid AUROC results are available for 15 of 15 categories. Mean category-level image AUROC is 0.8940; mean category-level pixel AUROC is 0.8628. These are unweighted averages across categories.
- Image AUROC ranges from 0.6400 (capsule) to 1.0000 (bottle); pixel AUROC ranges from 0.4680 (metal_nut) to 0.9738 (leather).
- Inference times are reported per category from the recorded metrics JSON. The transistor value is unusually high (4030.4256 s per image) and is preserved as recorded; verify the source measurement before interpreting timing comparisons.

## Limitations

- This is a 20-image pilot per category, with 10 normal and 10 defective images in each supplied result. The sample is small and may not represent the full test distribution or the range of defect types and severity.
- AUROC estimates from such small samples can be unstable; results should be treated as exploratory, not as definitive benchmark performance. The pilot does not establish generalization, statistical significance, or production readiness.
- The report reflects existing files only. Pixel AUROC and timing depend on the recorded evaluation setup, and timing can be affected by hardware, startup, and system load.

## File validation

- Found metrics JSON and prediction CSV for all 15 expected categories. Each CSV has 20 prediction rows; JSON totals and normal/defective counts match the CSV labels.
- No missing or malformed results detected in the expected category files.

Charts: [Image AUROC](image_auroc_comparison.png) · [Pixel AUROC](pixel_auroc_comparison.png).
