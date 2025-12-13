# Face-Recognition Project — Implementation README (for Codex/AI agent)

This README summarizes the **project requirements + design decisions** from our class slides so an AI coding agent can improve our code while keeping us aligned with the guidelines.

Team: Aubin Mugisha, Joel Habiyakare, Logan Simba, Owen Leitzell  fileciteturn0file0 fileciteturn0file1

---

## 1) Project Objective

Train and evaluate a **CNN-based face recognition / verification system** using the **Labeled Faces in the Wild (LFW)** dataset:
- Input: a face image (after detection + alignment)
- Output: a **fixed-length embedding** (target: **128-D feature vector**) so we can compare faces by distance/similarity fileciteturn0file0

Use case is **verification** (same person vs different person), and optionally identification (who is this person) fileciteturn0file1

---

## 2) Dataset (LFW)

- LFW: **13,000+ labeled face images** from **5,749 identities**, collected under real-world conditions (lighting, pose, expression) fileciteturn0file1
- Kaggle mirror used in class slides (example): `https://www.kaggle.com/datasets/atulanandjha/lfwpeople` (for convenience)

Key challenge: **many identities have very few images** (often 1), so we must split carefully and avoid overfitting fileciteturn0file1

---

## 3) Preprocessing Pipeline (must follow)

### Step 1 — Face detection + alignment (MTCNN)
Use **MTCNN (Multi-task Cascaded CNN)** with 3 stages:
1. Proposal Network: find candidate face regions
2. Refine Network: reject non-face regions
3. Output Network: final bounding box + **5 landmarks** (eyes, nose, mouth corners) fileciteturn0file0 fileciteturn0file1

**Why**: Haar Cascade was unreliable in our trials; MTCNN is more robust “in the wild.” fileciteturn0file1

### Step 2 — Standardize image size + format
Slides mention two sizes (keep configurable in code):
- Option A (custom CNN): aligned **128×128×3** RGB fileciteturn0file0
- Option B (FaceNet baseline): crop/resize **160×160** RGB fileciteturn0file0

**Implementation rule**: define one config constant, e.g. `IMG_SIZE = 128 or 160`, and ensure every stage uses the same value.

### Step 3 — Pixel normalization
Normalize pixel values consistently (document exact method in code). fileciteturn0file0 fileciteturn0file1

### Step 4 — Data augmentation
Use light augmentation to increase robustness:
- Horizontal flip
- Small rotations fileciteturn0file0 fileciteturn0file1

---

## 4) Neural Network Design (planned)

Goal: map a standardized face image → **128-D embedding** fileciteturn0file0

Planned lightweight CNN (inspired by FaceNet/CNN-S) fileciteturn0file0:
- **Block 1**: Conv 3×3, 32 filters → ReLU → MaxPool 2×2
- **Block 2**: Conv 3×3, 64 filters → ReLU → MaxPool 2×2
- **Block 3**: Conv 3×3, 128 filters → ReLU → Global Average Pool
- **Head**: Fully connected layer → **128-D embedding**

Important: if we do siamese/triplet training, the **same network (shared weights)** is applied to both images so embeddings are comparable. fileciteturn0file0

---

## 5) Training Losses (two-phase plan)

### Phase 1 — Classification (cross-entropy)
Treat each identity in the training set as a class:
- Softmax + multi-class cross-entropy
- Purpose: learn **discriminative face features** (identity separation) fileciteturn0file0 fileciteturn0file1

### Phase 2 — Metric learning (triplet loss)
Triplet = (Anchor, Positive, Negative):
- Positive = same person as anchor
- Negative = different person
- Enforce: `d(A,P) < d(A,N) + margin` fileciteturn0file0

Purpose: directly shape embedding space so **distance-based verification** works well. fileciteturn0file0

---

## 6) Experiment Design (must follow)

### Split rule: split by identity (not by image)
We split people into:
- Train: 70% of identities
- Validation: 15% of identities
- Test: 15% of identities

**Same person must never appear in two different sets** (prevents leakage/cheating). fileciteturn0file0

### How to build verification pairs (val/test)
Create a balanced dataset of pairs (≈50% positive, 50% negative) fileciteturn0file0:
- **Positive pairs**: for each identity with ≥2 images, create pairs like (img1,img2), (img1,img3), …
- **Negative pairs**: for each positive pair, take the first image and pair it with a random image of a different identity

---

## 7) Evaluation Metrics (must report)

Primary:
- **Verification accuracy** = correctly classified pairs / total pairs fileciteturn0file0
- **Distance threshold**: classify “same” if distance ≤ threshold (tune on validation) fileciteturn0file0
- Distance measure referenced: **cosine distance** between embeddings (smaller = more similar) fileciteturn0file0
- Optional “confidence”: `confidence = 1 - distance` fileciteturn0file0

Recommended reporting (for codex agent):
- accuracy, precision, recall, F1 (if we compute them)
- threshold selected on validation and reused on test

---

## 8) Baselines + Related Work Context (for writeup)

Slides discuss:
- **DeepFace**: heavy 3D alignment + large private dataset; strong LFW accuracy but not easily reproducible fileciteturn0file1
- **DeepID**: many CNNs on patches, fused very large feature vectors; high compute cost fileciteturn0file1
- **Siamese/Metric learning**: learns similarity directly; threshold matters; can be costly at scale fileciteturn0file1

---

## 9) Presentation/Assignment Guidelines (keep in mind)

Presentation I (project overview) must include:
1. Motivation
2. Project description + expected outcomes + anticipated challenges
3. Related work (≥2 papers)
4. Contributions / what makes our method unique fileciteturn0file2

Presentation II (algorithm design) must include:
1. Neural network design
2. Loss function (and why it aligns with goals)
3. Preprocessing steps
4. Experiment design + metrics fileciteturn0file3

---

## 10) Codex/AI Agent Instructions (what to change vs not change)

### Do (safe improvements)
- Refactor into clear modules: `data/`, `preprocess/`, `model/`, `train/`, `eval/`
- Add reproducibility: fixed random seeds, consistent splits by identity
- Add unit tests for: pair generation, split leakage, embedding shape = 128
- Improve training stability: early stopping on validation, LR scheduling, better logging
- Make `IMG_SIZE` configurable (128 vs 160) and keep consistent end-to-end

### Don’t (violates project intent)
- Don’t split by image (must split by identity)
- Don’t allow any identity overlap across train/val/test
- Don’t skip MTCNN alignment if the pipeline is part of our intended design
- Don’t change embedding dimension away from 128 unless explicitly required

---

## 11) Suggested Repo Structure (recommended)

```
face-recognition/
  README.md
  requirements.txt
  data/
    raw/
    processed/
    splits/            # identity lists for train/val/test
    pairs/             # val/test pair lists
  preprocess/
    mtcnn_align.py
    augment.py
  model/
    cnn_embedding.py
    losses.py          # cross-entropy, triplet loss
  train/
    train_classify.py
    finetune_triplet.py
  eval/
    evaluate_pairs.py
    threshold_tuning.py
  utils/
    config.py
    seed.py
    io.py
```

---

## 12) Minimal “definition of done” checklist

- [ ] Identity-based 70/15/15 split saved to disk
- [ ] MTCNN align + resize + normalization works on full dataset
- [ ] Embedding model outputs 128-D vectors for any valid input
- [ ] Phase 1 training (cross-entropy) runs end-to-end
- [ ] Phase 2 training (triplet loss) runs end-to-end
- [ ] Validation threshold tuning implemented
- [ ] Test evaluation reports accuracy (and optional precision/recall/F1)
- [ ] No identity leakage (assertions in code)

