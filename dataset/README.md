# UWrites Handwriting Dataset

This directory contains the original raw handwriting scan collection used to train and evaluate the UWrites personalized handwriting synthesis model.

---

## 1. Directory Structure

The dataset is partitioned into six lexical categories across two page formats (`ruled` and `unruled`):

```text
dataset/
├── capital letters/
│   ├── ruled/          # 1 scan (1.jpeg)
│   └── unruled/        # 4 scans (2.jpeg - 5.jpeg)
├── characters/
│   ├── ruled/          # 1 scan (1.jpeg)
│   └── unruled/        # 4 scans (2.jpeg - 5.jpeg)
├── collection/
│   ├── ruled/          # 2 scans (2.jpeg, 3.jpeg)
│   └── unruled/        # 3 scans (1.jpeg, 4.jpeg, 5.jpeg)
├── numbers/
│   ├── ruled/          # 1 scan (1.jpeg)
│   └── unruled/        # 3 scans (2.jpeg - 4.jpeg)
├── sentences/
│   ├── ruled/          # 6 scans (15.jpeg, 31.jpeg, etc.)
│   └── unruled/        # 40 scans (1.jpeg - 40.jpeg)
└── small letters/
    ├── ruled/          # 1 scan (1.jpeg)
    └── unruled/        # 3 scans (2.jpeg - 4.jpeg)
```

**Dataset Summary:**
- **Total Scans:** 69 `.jpeg` images (~7.28 MB)
- **Primary Category:** Sentences (46 scans / 66.7% of total) for continuous handwriting flow
- **Vocabulary Coverage:** Uppercase (A-Z), lowercase (a-z), digits (0-9), punctuation, and special symbols

---

## 2. Data Acquisition & Characteristics

- **Writing Medium:** Natural pen-and-paper writing using standard ballpoint / gel pens.
- **Paper Types:**
  - `ruled`: Notebook paper with pre-printed horizontal baseline ruling.
  - `unruled`: Blank white paper allowing natural slant and line drift.
- **Image Resolutions:** Range from `960 x 1600` to `2083 x 400` pixels depending on page orientation and capture framing.
- **Condition:** Includes real-world natural variations such as uneven illumination, slight page tilt, baseline slant, and ink stroke thickness differences.

---

## 3. Preprocessing Workflow

Each raw page scan is ingested by the preprocessing pipeline (`src/preprocessing.py` or the notebook) through the following operations:

1. **Deskewing:** Rotates the page using the minimum-area bounding box angle computed from foreground ink pixels.
2. **Adaptive Binarization:** Applies Gaussian adaptive thresholding (`cv2.adaptiveThreshold`) to separate ink from paper across non-uniform lighting conditions.
3. **Ruled Line Removal:** For scans under `ruled/`, horizontal morphological opening isolates straight guidelines and removes them to prevent line artifacts from attaching to character descenders/ascenders.
4. **Line & Word Segmentation:** Uses projection profiling to segment continuous writing into individual line strips and discrete word/glyph crops.

---

## 4. Usage in Training

To re-run the full segmentation and generate labeled training samples:
```bash
python -c "from src.preprocessing import batch_segment_directory; batch_segment_directory('dataset', 'segmented')"
```
Transcriptions are then mapped via `all_words_labeled.csv` or `all_words_template.csv` before training the GAN.
