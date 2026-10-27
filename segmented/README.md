# UWrites Segmented Word & Character Images

This directory stores the segmented word and glyph image crops extracted from the raw page scans in `dataset/`. These images serve as the ground truth training targets for the generative model.

---

## 1. Directory Structure & Inventory

```text
segmented/
├── capital_letters/      # 76 images  (ruled: 3,   unruled: 73)
├── mixed_case/           # 189 images (ruled: 17,  unruled: 172)
├── numbers/              # 163 images (ruled: 41,  unruled: 122)
├── sentences/            # 908 images (ruled: 17,  unruled: 891)
├── small_letters/        # 52 images  (ruled: 3,   unruled: 49)
└── special_characters/   # 131 images (ruled: 25,  unruled: 106)
```

**Total Dataset Size:** 1,519 `.png` images (~14.35 MB)

---

## 2. File Naming Convention

Each image follows a deterministic coordinate naming schema:

```text
{page_id}_line{line_index:02d}_word{word_index:02d}.png
```

- `page_id`: Identifier corresponding to the source page scan in `dataset/` (e.g., `15`, `2`, `10`).
- `line_index`: Zero-padded index indicating the horizontal line strip from top to bottom.
- `word_index`: Zero-padded index indicating the horizontal word position within that line from left to right.

*Example:* `15_line00_word02.png` represents the 3rd word extracted from the 1st line of page `15.jpeg`.

---

## 3. Labeling and Dataset Alignment

When training the conditional GAN:
1. `all_words_template.csv` is generated with columns: `filename`, `category`, `ruling`, `relpath`, `transcription`.
2. Manual/semi-automated transcription populates the `transcription` field.
3. In the initial project benchmark:
   - **Total segmented samples:** 1,519
   - **Unlabeled / fragmented crops filtered:** 207
   - **Usable annotated training samples:** 1,312

---

## 4. Quality & Engineering Findings

During empirical inspection of the 1,519 crops:
- **Median Crop Dimensions:** 52 px width × 60 px height.
- **Speckle Sensitivity in Projection Profiling:** Strict `v > 0` thresholds on raw adaptive binarization outputs caused stray background noise to merge multiple lines into a single page-height bounding box (e.g., on `2.jpeg`).
- **Aspect Ratio Normalization:** Forceful resizing to `(256, 64)` distorted single character glyphs (e.g., "u" or "1"). The updated preprocessing pipeline in `src/preprocessing.py` introduces aspect-ratio-preserving padding and adaptive noise-floor projection filtering.
