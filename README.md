# INFO284 Machine Learning – Group Exam (Spring 2026)

A two-task machine learning project completed as a group exam in the course **INFO284 Machine Learning**. The project covers sentiment analysis of WhatsApp app reviews and binary classification of AI-generated vs. real artwork.

---

## Tasks

### Task 1 – Sentiment Analysis of WhatsApp Reviews

Predicts the star rating (1–5) of WhatsApp Google Play reviews from review text. Four models are trained and evaluated on the same data split.

**Dataset:** `reviews.csv` – 6 210 reviews with columns `review_id`, `rating`, `review_text`, `review_date`, `helpful`. After cleaning (removing gibberish, spam repetition, emoji-only, and non-English reviews), 5 657 reviews remain.

**Models:**

| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Logistic Regression | 0.493 | 0.33 | TF-IDF (unigrams + bigrams), C=1.0, L2, grid search |
| LinearSVC (SVM) | 0.601 | 0.35 | Best overall; C=0.1 via 5-fold CV |
| LightGBM | 0.535 | 0.30 | Class-weighted; grid search over depth and estimators |
| Bidirectional LSTM | 0.527 | 0.33 | Most balanced minority-class coverage; data augmentation |

**Key design decisions:**
- Class weighting applied to all four models to address the heavy imbalance (rating 5 = ~49% of data)
- Macro F1 used as primary metric rather than accuracy
- Shared text cleaning pipeline: lowercasing, URL/email removal, stopword filtering (with negation words kept), domain noise (`app`, `whatsapp`) added to stoplist
- 80/20 stratified train/test split with `random_state=42`; hyperparameters tuned on training data only via cross-validation
- LSTM only: synonym-based data augmentation (`nlpaug`) to balance minority classes before training

---

### Task 2 – AI vs. Real Art Classifier

Binary image classifier that distinguishes AI-generated art from human-made art, using transfer learning on a small dataset.

**Dataset:** `Art_shuffled/` – 1 012 images (559 AI, 453 real), split into `AiArtData/` and `RealArt/` subfolders.

**Model:** EfficientNetB0 pretrained on ImageNet, with a custom classification head:
- `GlobalAveragePooling2D` → `Dense(128, ReLU)` → `Dropout(0.3)` → `Dense(1, sigmoid)`

**Training:** Two-phase approach:
1. **Phase 1 (frozen base):** Only the classification head is trained (Adam lr=1e-3, up to 20 epochs, EarlyStopping patience=3)
2. **Phase 2 (fine-tuning):** Last 20 layers of EfficientNetB0 unfrozen and trained at a lower rate (Adam lr=1e-5, EarlyStopping patience=3)

**Results (validation set, 194 images):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AI Art | 0.75 | 0.85 | 0.79 |
| Real Art | 0.78 | 0.64 | 0.70 |
| **Overall accuracy** | | | **0.76** |

**Task 2b – New images:** The model was tested on 5 self-sourced images (3 AI, 2 real). Accuracy was 40% (2/5), with both real photographs misclassified as AI. Photorealistic AI images also caused misclassification, as the training set skews toward illustrative AI art.

---

## Project Structure

```
├── reviews.csv               # WhatsApp review dataset (Task 1)
├── Art_shuffled/
│   ├── AiArtData/            # AI-generated images
│   └── RealArt/              # Human-made art images
├── task2_new_images/         # 5 self-sourced images for Task 2b
│   ├── AiArtData/
│   └── RealArt/
├── art_classifier.keras      # Saved EfficientNetB0 model (generated on run)
└── final_submission.pdf      # Full notebook export (this file)
```

---

## Dependencies

```
tensorflow
scikit-learn
lightgbm
nltk
nlpaug
pandas
numpy
matplotlib
seaborn
```

Install with:

```bash
pip install tensorflow scikit-learn lightgbm nltk nlpaug pandas numpy matplotlib seaborn
```

Additional NLTK data downloads are handled automatically in the notebook:

```python
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
```

---

## Usage

Run the notebook top-to-bottom. Both tasks are contained in a single notebook.

- Task 1 expects `reviews.csv` in the working directory
- Task 2 expects `Art_shuffled/` in the working directory
- Task 2b expects `task2_new_images/` with `AiArtData/` and `RealArt/` subfolders

The trained image classifier is saved to `art_classifier.keras` after Phase 2 and reloaded automatically for Task 2b if already present.

---

## Use of AI Tools

AI assistants (ChatGPT GPT-5.3 and Claude Sonnet 4.6) were used as productivity tools for routine implementation tasks: data cleaning regex patterns, boilerplate EDA code, plot formatting, and debugging. All generated code was reviewed and tested. Model selection, evaluation strategy, and interpretation of results were done independently by the group.

---

## References

- Müller, A. C. & Guido, S. – *Introduction to Machine Learning with Python*
- Géron, A. – *Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow*
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [LightGBM documentation](https://lightgbm.readthedocs.io/)
- [TensorFlow / Keras API](https://www.tensorflow.org/api_docs/python/tf/keras)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
