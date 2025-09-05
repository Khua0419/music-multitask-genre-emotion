# DEAM / EmoMusic preprocessing notes

**Goal**: Explain where to place audio and annotations and how clips are segmented.

## Expected local layout

## Segmentation plan
- Resample to 44.1 kHz (if needed).
- Convert to mono (optional).
- Extract 10–30 s clips with fixed hop (e.g., 10 s).
- Save clip index CSV: `path, start_sec, end_sec, label(s)`.

## Notes
- Do **NOT** commit `data/` to Git.
- This file is for reproducibility only.
