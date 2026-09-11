# Aquaculture AI Course — 15 Practical Works

Reproducible supplementary materials for a pilot course in AI and computer vision for aquaculture undergraduates. The repository contains 15 Russian-language Jupyter notebooks, synthetic teaching datasets, assessments, and an article-alignment matrix.

## Scope and evidence statement

These materials document the intervention. They do not by themselves prove educational effectiveness, transferability, or scalability. All datasets are synthetic. Report student outcomes only from retained individual-level records and report model metrics only from saved, reproducible runs.

## Course structure

| Module | Works | Focus |
|---|---|---|
| Data foundations | 1–3 | profiling, cleaning, leakage-safe splitting |
| Regression | 4–6 | biomass prediction, nonlinearity, regularization |
| Classification | 7–9 | risk classification, thresholds, model comparison |
| Time series | 10–12 | baseline, ARIMA, LSTM preparation |
| Computer vision | 13–15 | OpenCV, detection metrics, YOLOv8 |

## Use

Install `requirements.txt`, open the repository root, and run notebooks in numeric order. Paths assume execution from each notebook's module folder. In Colab, upload the repository or mount Drive without changing its structure. YOLOv8 training requires internet access to obtain pretrained weights and benefits from a GPU.

## Reproducibility

The tabular and image datasets are included. The final YOLO notebook deliberately does not report a predetermined mAP. Save `results.csv`, model weights, environment details, and the exact test metrics after a real run before citing them.

## License

Code and original teaching text: MIT. Synthetic datasets: CC0.
