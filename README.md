# 🐾 Paw Pass

**No paws, no pass.**

[![English](https://img.shields.io/badge/lang-English-0052cc?style=for-the-badge)](README.md)
[![Українська](https://img.shields.io/badge/%D0%BC%D0%BE%D0%B2%D0%B0-%D0%A3%D0%BA%D1%80%D0%B0%D1%97%D0%BD%D1%81%D1%8C%D0%BA%D0%B0-lightgrey?style=for-the-badge)](README.uk.md)

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68)](https://docs.ultralytics.com/)

---

A webcam watches the doorway. When it sees the cat, the door opens, and once
the cat has gone, it closes again. The lock is simulated behind a
`DoorController` interface, ready to be swapped for a real relay or servo.

## Highlights

- **Threshold chosen on validation, test scored once.** The numbers favoured
  0.70; 0.85 was kept on purpose, and the reasoning is [written down](#threshold-chosen-on-validation).
- **Error analysis found the root cause.** Missed Sphynx cats and a scratching
  post taken for a cat both trace back to [one reliance on fur texture](#what-the-errors-show).
- **Door logic that survives a real camera.** Hysteresis, frame streaks and a
  minimum open time, after a [live flicker bug](#real-time) that a test now replays.
- **Two interchangeable detectors.** The classifier trained here and a
  pretrained YOLO sit behind one interface, switched with a flag.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python main.py train                     # downloads the dataset, trains the model
python main.py camera --model mobilenet  # or --model yolo
```

`python main.py` lists every command. Quit the camera with `Q` / `Esc` or
`Ctrl+C`. On macOS the terminal needs camera access in System Settings →
Privacy & Security → Camera.

## How it works

```
TRAINING (once)                          RUNTIME (every frame)

Oxford-IIIT Pet                          Camera → OpenCV frame
   ↓ resize, normalize, augment             ↓ resize, normalize
   ↓ split 70 / 15 / 15                     ↓
MobileNetV2, new head trained  ────────► model → P(cat)
   ↓                                        ↓ hysteresis + frame streaks
threshold chosen on validation              ↓
score on test, once                      DoorController → OPEN / CLOSED
```

The task is binary classification: an RGB image `[3, 224, 224]` goes in, two
logits come out, softmax turns them into `P(cat)`.

## Dataset

[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/): 3,680 images
of 37 breeds. The 12 cat breeds are labelled `1`, the 25 dog breeds `0`.

| | Images | Share |
|---|---|---|
| not cat | 2,492 | 67.7% |
| cat | 1,188 | 32.3% |

Always answering "not cat" already scores 0.677 accuracy, so that is the
baseline, and precision / recall matter more than accuracy.

Split 70 / 15 / 15, stratified by label, fixed seed: 2,576 / 552 / 552.

![Dataset samples](assets/eda_samples.png)

## Model

MobileNetV2 pretrained on ImageNet. The convolutional layers are frozen and
the last layer is replaced with `Linear(1280 → 2)`, so only **2,562 of 2.2M**
parameters are trained. On 3,680 images this converges in five epochs.

| | |
|---|---|
| Input | Resize 224×224, ImageNet mean / std normalization |
| Augmentation (train only) | Horizontal flip, rotation ±10°, brightness jitter |
| Training | CrossEntropy, Adam lr 1e-3, batch 32, 5 epochs |
| Accuracy after 5 epochs | train 0.972, val 0.982 — no overfitting |

## Results

### Threshold, chosen on validation

A false positive opens the door for a stranger, so precision comes first.

```
   thr  Precision   Recall    FP    FN
  0.50      0.994    0.938     1    11
  0.70      1.000    0.888     0    20
  0.85      1.000    0.854     0    26   <- chosen
  0.95      1.000    0.792     0    37
```

On paper 0.70 wins: the same zero false positives, six more cats let in.
0.85 was kept on purpose. Every "not cat" in the data is a dog, so zero false
positives here says nothing about rugs or scratching posts, and the live
camera did mistake those for a cat. The extra margin covers that gap until
the data does.

### Final score on test (552 images, threshold 0.85, run once)

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| **0.966** | **1.000** | **0.893** | **0.944** |

![Confusion matrix](assets/confusion_matrix.png)

No non-cat would have opened the door; 19 of 178 cats would have waited.

### What the errors show

![Missed cats](assets/errors_false_negative.png)

The most confident misses include hairless Sphynx cats and cats with dark or
shadowed faces. Together with the scratching post the camera took for a cat,
this points to one cause: the model relies heavily on fur texture.

## Real-time

![Door opening for the cat, then closing once it has gone](assets/demo.jpg)

*Two consecutive events from `logs/events.csv`: the frame the camera logged, with the recorded decision drawn on top.*

Three layers stand between a frame's score and the lock:

| Layer | Setting | Prevents |
|---|---|---|
| Hysteresis | open ≥ 0.50, stay open ≥ 0.30 (YOLO) | Flicker when confidence hovers near one threshold |
| Frame streaks | 5 frames in a row to open or close | One blurry frame moving the door |
| Minimum open time | 3 s | Shutting the instant the cat steps out of view |

Hysteresis was added after a live run where the door toggled ten times in a
few seconds; `tests/test_debounce.py` replays that sequence.

Two interchangeable detectors:

| | `mobilenet` | `yolo` |
|---|---|---|
| Approach | Classification of the whole frame | Object detection, COCO class 15 |
| Trained here | Yes | No, pretrained |
| Thresholds (open / stay) | 0.85 / 0.60 | 0.50 / 0.30 |

YOLO localises the animal instead of scoring the whole frame, which makes it
the stronger choice on a real door.

Each door state change is written to `logs/events.csv` along with the frame
that caused it. Frames from false openings are ready-made hard negatives.

## Project layout

```
main.py              entry point: python main.py <command>
src/
├── config.py        paths and every tunable number
├── device.py        MPS / CUDA / CPU
├── ml/              used by both training and runtime
│   ├── preprocessing.py
│   ├── model.py
│   └── inference.py
├── training/        runs once
│   ├── data.py  eda.py  train.py  evaluate.py  error_analysis.py
└── runtime/         runs on every frame
    ├── detectors.py  debounce.py  door_controller.py
    ├── event_log.py  camera.py
tests/               19 tests of the door logic, no camera needed
```

Imports point one way: `training/` and `runtime/` use `ml/`, never each
other. A Raspberry Pi needs only `ml/`, `runtime/` and the trained weights.

| Command | Does |
|---|---|
| `python main.py eda` | Explore the dataset |
| `python main.py train` | Train → `models/cat_detector.pth` |
| `python main.py evaluate` | Threshold sweep on val, final score on test |
| `python main.py errors` | Save FP / FN examples from validation |
| `python main.py camera` | Run the door (`--model`, `--headless`, `--no-log`) |
| `pytest tests/ -q` | Run the tests |

## Limitations and next steps

| Limitation | Next step |
|---|---|
| "Not cat" is only dogs; rugs and scratching posts fool the model | Add hard negatives from `logs/frames/` |
| Model leans on fur texture; misses Sphynx cats | More hairless cats, stronger brightness augmentation |
| Close-up pet photos, not a doorway camera | Collect footage from the real camera position |
| Classification can't tell where the cat is | Use the YOLO backend for deployment |
| Any cat gets in, not only mine | Re-identification: compare an embedding to reference photos of one cat |
| No night footage | Infrared camera and night-time training data |
