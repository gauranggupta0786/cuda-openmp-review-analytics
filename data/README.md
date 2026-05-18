# Dataset Information

## Full Dataset (Not in Repository)

The complete Amazon Movies & TV reviews dataset is ~3GB and cannot be committed to GitHub.

**Download and prepare it with these commands:**

```bash
# Download
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz

# Decompress
gunzip reviews_Movies_and_TV_5.json.gz

# Preprocess into reviews.csv
python3 ../src/preprocess.py
```

**Output:** `reviews.csv` with columns: `asin, rating, reviewerID, reviewText`

**Stats after preprocessing:**
- Total reviews : 1,697,533
- Unique movies : 50,052
- Avg reviews/movie : ~34

---

## Sample Files in This Folder

| File | Purpose | Records |
|---|---|---|
| `reviews_sentiment.csv` | Input for `cuda_reviewanalysis` | 4 reviews |
| `reviews_text.csv` | Input for `c_elaborate` / `c_elaborate_openmp` | 5 reviews (reviewer A1) |

These are minimal demo files only — not representative of the full dataset.
The full programs require the real `reviews.csv` generated above.

---

## VADER Lexicon (for Sentiment Analysis)

Also not committed — download separately:

```bash
wget https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/vader_lexicon.txt
```

**Stats:** 7,520 words with sentiment scores.
Place `vader_lexicon.txt` in the same directory where you run `./sentiment`.
