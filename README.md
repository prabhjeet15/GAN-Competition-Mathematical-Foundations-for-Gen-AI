# Overview

# GAN Challenge

Welcome to the GAN Challenge!

Your task is to train a GAN-style architecture to generate realistic 32×32 images using the provided dataset of base64-encoded image shards.

Your model’s performance will be evaluated using the Fréchet Inception Distance (FID) — lower is better.

---

# Objective

Train a Generative Adversarial Network (GAN) or any generative model (e.g., DCGAN, WGAN, Diffusion Model) to generate 32×32 images similar to the real dataset.

The goal is to minimize the FID score between generated and real image distributions.

---

# Step 1. Understand the Dataset

The dataset consists of several `.zip` files, each containing `.jsonl` files.

Each line represents one image with metadata and base64-encoded image data.

| Key | Description |
|---|---|
| `id` | Unique image ID |
| `mode` | `"I;16"` (16-bit grayscale) or `"RGBA"` |
| `size` | `[H, W]` |
| `exif_rot` | Rotation in degrees (`0`, `90`, `180`, `270`) |
| `invert` | Whether to invert pixel intensities |
| `img_b64 / img64` | Base64-encoded image bytes |
| `alpha_b64` | Optional transparency mask |

---

# Step 2. Decode the Images

You must decode and standardize all images before training.

## Steps

1. Decode Base64 → bytes
2. Open using PIL
3. Apply EXIF rotation and inversion
4. Handle alpha channels (if present)
5. Convert to RGB
6. Resize to 32×32
7. Normalize pixel values to `[-1, 1]`

---

# Start

Nov 12, 2025

# Close

Nov 17, 2025

---

# Description

You are given a dataset of image shards, each record stored as a JSONL entry containing base64-encoded pixel data, metadata like rotation, inversion, and size — and occasional chaos (missing keys, invalid encodings, random alpha masks).

Your job is to:

- Reconstruct valid images from these shards
- Handle 16-bit grayscale and RGBA modes correctly
- Crop to tight bounding boxes if alpha channels exist
- Standardize to 32×32 RGB tensors normalized to `[-1,1]`
- Train your own GAN-style generator
- Generate fixed-seed samples and compete for the lowest FID score

---

# Submission Format & Instructions

Each participant must submit a `submission.csv` containing 1000 rows — one per generated image — with Inception-V3 (`pool3` layer) feature vectors extracted from their generated samples.

Your task is to train a GAN (or any generative model) to learn the visual distribution of these images and generate new samples of similar quality.

Your generated images will be evaluated against real image features using the Fréchet Inception Distance (FID) metric.

---

# Submission File

- **Filename:** `submission.csv`
- **Number of rows:** `1000` (exactly same as `solution.csv`)
- **Number of columns:** `1 id column + 2048 feature columns (f0 … f2047)`

---

# Expected Format

```csv
id,f0,f1,f2,...,f2047
dig-000000,0.012,-0.034,0.020,...,0.015
dig-000001,0.051,0.011,-0.007,...,-0.012
...
dig-000999,-0.002,0.005,-0.011,...,0.017
```

# How to Generate Features

```python
import torch
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from PIL import Image

# Load Inception-V3 pretrained on ImageNet
m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
m.fc = torch.nn.Identity()
m.eval()

preproc = transforms.Compose([
  transforms.Resize((299, 299)),
  transforms.ToTensor(),
])

def extract_feature(img_path):
  img = Image.open(img_path).convert("RGB")
  x = preproc(img).unsqueeze(0)

  with torch.no_grad():
      feat = m(x)

      if isinstance(feat, (list, tuple)):
          feat = feat[0]

  return feat.squeeze().numpy()
```

Then, for all your generated images (`1000` samples):

```python
import pandas as pd
import numpy as np

ids = [f"dig-{i:06d}" for i in range(1000)]

features = []

for i, img_path in enumerate(generated_image_paths):
  feats = extract_feature(img_path)
  features.append(feats)

df = pd.DataFrame(features, columns=[f"f{i}" for i in range(2048)])
df.insert(0, "id", ids)

df.to_csv("submission.csv", index=False)
```

---

# Evaluation

Your performance will be measured using **Frechet Inception Distance (FID)**:

- FID compares statistics (mean, covariance) of generated images vs. real ones
- For grayscale images, triplicate channels before computing features
- Reference mean (`ref_mu`) and covariance (`ref_sigma`) are precomputed on a hidden test subset
- Lower FID = Better visual realism & distribution alignment
