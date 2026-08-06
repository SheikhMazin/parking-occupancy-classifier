# Smart Parking Occupancy Detection

A computer vision system that classifies individual parking spaces as **empty** or **occupied** using transfer learning on the PKLot dataset. Built as an exploration of applying ML to a real infrastructure problem — parking lot inefficiency.

## Problem

Drivers waste significant time circling lots looking for open spots, and most parking infrastructure has no live occupancy data. This project explores a low-cost approach: instead of installing per-spot sensors (the industry-standard but hardware-heavy approach), use existing or low-cost cameras and a lightweight vision model to classify space occupancy directly from images.

## Approach

Rather than training a full object detector to locate *and* classify every space in a raw camera frame, this project splits the problem into two parts:

1. **Space localization** — defined once per fixed camera (manually, or via provided dataset annotations)
2. **Space classification** — a binary image classifier that looks at a single cropped space and predicts empty or occupied

This crop-then-classify design keeps the model simple to train while matching exactly how it would be used in a real deployment: one camera, many spaces, each cropped and classified independently (batched for efficiency).

## Dataset

**PKLot** — parking lots across 3 locations (PUC, UFPR04, UFPR05) under sunny, cloudy, and rainy conditions. Sourced via a [Kaggle mirror](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset) of the original PKLot dataset, exported in COCO object-detection format — 12,416 full parking-lot images, each annotated with bounding boxes for every individual parking space visible in the frame, rather than pre-cropped classification images.

A preprocessing step (`src/data/prepare_split.py`) parses these COCO annotations and extracts each individual annotated space as its own cropped image, sorted by label:

```
data/processed/{train,valid,test}/{Empty,Occupied}/
```

This produces several hundred thousand individual space-level crops across the three splits — this cropped data, not the original 12,416 full-lot images, is what's actually used for training and evaluation.

| Split | Empty | Occupied | Total |
|-------|-------|----------|-------|
| Train (subsampled) | 15,000 | 15,000 | 30,000 |
| Valid | 73,629 | 69,687 | 143,316 |
| Test | 36,584 | 34,100 | 70,684 |

Training was run on a balanced 30,000-image subsample of the full ~500,000-image train split, to keep iteration fast during development. Validation and test sets were used in full, untouched.

## Model

- **Architecture:** ResNet18, pretrained on ImageNet
- **Transfer learning:** all pretrained backbone layers frozen; the final fully-connected layer replaced with a fresh binary classification head (512 → 2)
- **Trainable parameters:** 1,026 out of 11,177,538 total
- **Input:** 128×128 RGB crops, normalized with standard ImageNet statistics
- **Loss:** Cross Entropy Loss
- **Optimizer:** Adam, lr=0.001
- **Training:** 5 epochs, batch size 32, best checkpoint selected by validation loss

## Results

**Test set accuracy: 94.88%** (70,684 held-out images, never seen during training)

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Empty | 0.96 | 0.94 | 0.95 |
| Occupied | 0.94 | 0.96 | 0.95 |

### Training vs. Validation Loss
![Loss Curve](assets/loss_curve.png)

Both curves drop sharply in the first epoch and stay close together throughout training, without validation loss diverging upward — a sign the model is learning genuine patterns rather than memorizing the training subsample. The best-performing epoch (lowest validation loss) is the one saved as the final checkpoint.

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

Rows are the true label, columns are the predicted label. The diagonal (top-left to bottom-right) represents correct predictions; the off-diagonal cells are errors. Notably, false negatives (predicting Empty when a space is actually Occupied) are lower than false positives — the more forgiving direction to err in for this use case, since sending a driver to a spot that's actually taken is a worse outcome than missing a genuinely open one.

### Per-Class Metrics
![Per-Class Metrics](assets/per_class_metrics.png)

Precision, recall, and F1-score for each class, side by side. Both classes score similarly (~0.94-0.96 across all three metrics), indicating the model isn't biased toward favoring one class over the other.

### Class Distribution Across Splits
![Class Distribution](assets/class_distribution.png)

Confirms class balance across train, validation, and test sets. Train was deliberately subsampled to an even 15,000/15,000 split; validation and test were left at their natural (still roughly balanced) sizes.

### Sample Predictions
![Prediction Samples](assets/prediction_samples.png)

25 randomly sampled test images with the model's prediction and the true label. Green titles are correct predictions, red are incorrect — a quick visual sense of what the model gets right (and how confidently) on typical examples.

### Misclassified Examples
![Misclassified Samples](assets/misclassified_samples.png)

25 randomly sampled *incorrect* predictions only. Reviewing these is useful for understanding failure modes — e.g., whether errors cluster around specific lighting conditions, partial occlusion, or ambiguous crops — and for identifying what a future fine-tuning pass should target.

## Project Structure

```
smart-parking/
├── data/
│   ├── raw/PKLot/              # original dataset + COCO annotations
│   └── processed/              # extracted, labeled space crops
├── src/
│   ├── data/
│   │   ├── prepare_split.py    # COCO annotations -> cropped, labeled images
│   │   └── dataset.py          # PyTorch Dataset/DataLoader construction
│   ├── models/
│   │   └── classifier.py       # ResNet18 transfer learning setup
│   ├── training/
│   │   └── train.py            # training loop, checkpointing
│   └── evaluation/
│       ├── evaluate.py         # test set metrics
│       ├── view_predictions.py # inspect saved predictions
│       └── make_visuals.py     # generates the charts above
├── checkpoints/                # saved model weights (gitignored)
└── assets/                     # generated charts for this README
```

## Setup

**1. Clone the repo and create a virtual environment**
```bash
git clone https://github.com/SheikhMazin/parking-occupancy-classifier.git
cd parking-occupancy-classifier
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
If you're on a different CUDA version (or CPU-only), the PyTorch line in `requirements.txt` may need to be swapped for the right build from the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) before installing the rest.

**3. Download the dataset**
Get the PKLot dataset from its [Kaggle page](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset) (requires a free Kaggle account). Either download the `.zip` manually from the page, or use `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("ammarnassanalhajali/pklot-dataset")
print(path)
```

**4. Place the dataset in the project**
Copy (or move) the downloaded dataset so its contents land at:
```
data/raw/PKLot/versions/1/{train,valid,test}/
```
Each of `train`, `valid`, and `test` should contain the image files plus an `_annotations.coco.json` file. This exact path is what `src/data/prepare_split.py` expects — if your download extracts to a different folder name/depth, adjust the paths at the top of `prepare_split.py` to match.

**5. Create the checkpoints folder** (or let it get created automatically — `train.py` does this on its own via `os.makedirs`)
```bash
mkdir -p checkpoints
```

## Running It

Run these in order, from the project root:

```bash
# 1. Extract labeled crops from raw COCO annotations
#    (reads data/raw/PKLot/..., writes to data/processed/{train,valid,test}/{Empty,Occupied}/)
python src/data/prepare_split.py

# 2. Train the model
#    (reads data/processed/, writes checkpoints/best_model.pth + checkpoints/training_history.pth)
python -m src.training.train

# 3. Evaluate on the held-out test set
#    (loads checkpoints/best_model.pth, writes checkpoints/test_results.pth, prints metrics)
python -m src.evaluation.evaluate

# 4. Generate charts
#    (reads checkpoints/training_history.pth + checkpoints/test_results.pth, writes assets/*.png)
python -m src.evaluation.make_visuals
```

**Notes:**
- Step 1 only needs to be run once — after that, `data/processed/` has everything training needs.
- Steps 2-4 must run in that order, since each depends on files the previous step produced.
- Re-running step 2 will overwrite `checkpoints/best_model.pth` with a fresh training run — back it up first if you want to keep a previous result.
- All commands assume you're running from the project root with the virtual environment activated, and that `src/`, `src/data/`, `src/models/`, and `src/training/` each contain an (empty) `__init__.py` file.

## Known Limitations & Next Steps

- **Cross-domain generalization is untested.** The model has only been trained and evaluated on PKLot's own camera angles and lots. Performance on a genuinely new camera setup (different angle, height, lighting) is unverified — the natural next step is fine-tuning on a small hand-labeled sample from a different lot.
- **Space localization is manual.** This project assumes bounding boxes for each space are known ahead of time (provided by the dataset, or defined once per fixed camera in a real deployment). It does not include a space-detection model.
- **Per-space inference doesn't scale to city-level deployments without batching.** At the scale of a single university or parking structure, batched inference on a lightweight classifier like this is fast enough (low seconds for thousands of spaces). At true city scale, a single-pass object detector would likely be a better architectural choice.
- **Data source for real deployment is not yet resolved.** Using this on a real lot would require either camera access permission from the lot's owner (e.g., a university's parking services) or a pivot toward crowdsourced occupancy reporting.

## Dataset Citation

```
Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., Koerich, A.,
PKLot – A robust dataset for parking lot classification,
Expert Systems with Applications, 42(11):4937-4949, 2015.
```
