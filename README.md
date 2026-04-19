# IMDb Review Sentiment Classification

## Business Objective
This project builds a sentiment classification system to detect negative movie reviews from IMDb text data. The business goal is to help a content platform prioritize moderation, quality monitoring, and user-experience analysis by automatically flagging reviews with negative sentiment.

The target performance threshold is an F1 score of at least `0.85`, with a preference for models that are reliable, interpretable, and practical to maintain in production.

## Project Scope
The notebook evaluates several classic NLP approaches for binary text classification and compares them against a simple business benchmark. The final deliverable is not just the best score, but a model recommendation that balances performance and operational simplicity.

Main asset:
- Notebook: [notebooks/imdb_sentiment_classification.ipynb](notebooks/imdb_sentiment_classification.ipynb)

## Methodology
The workflow follows a business-oriented modeling pipeline:

1. Understand the problem and define the sentiment classification target.
2. Load and inspect the IMDb review dataset.
3. Run exploratory analysis to validate class balance and temporal coverage.
4. Normalize review text to reduce surface-level noise.
5. Reuse the predefined train/test split included in the dataset.
6. Evaluate multiple NLP pipelines using a shared evaluation framework.
7. Compare model quality using F1, ROC AUC, Average Precision, and threshold behavior.
8. Recommend the most practical model for initial deployment.

## Models Evaluated
The notebook compares the following approaches:

### 1. Business Benchmark: Majority Classifier
A dummy classifier that always predicts the majority class. This establishes the minimum acceptable baseline for business value.

### 2. TF-IDF + Logistic Regression
A lightweight and highly interpretable text-classification pipeline using TF-IDF features with class-balanced logistic regression.

### 3. TF-IDF + Calibrated Linear SVC
A linear support vector classifier calibrated to produce probabilities, allowing consistent threshold-based evaluation and curve analysis.

### 4. spaCy Lemmatization + TF-IDF + Logistic Regression
A variation of the logistic regression pipeline that adds lemmatization with spaCy before vectorization.

## Results Summary
The evaluated production candidates perform very similarly on the test set:

- `TF-IDF + Logistic Regression`: F1 `0.88`
- `TF-IDF + Calibrated Linear SVC`: F1 `0.88`
- `spaCy Lemmatization + TF-IDF + Logistic Regression`: F1 `0.88`

All three models exceed the target threshold of `0.85` and clearly outperform the business benchmark.

## Final Recommendation
The recommended model is **TF-IDF + Logistic Regression**.

Why this model is the best choice for a first production version:
- It matches the best observed test performance.
- It exceeds the target F1 threshold with margin.
- It is easier to explain, maintain, and retrain than the alternatives.
- It avoids the extra processing cost of spaCy without sacrificing quality.
- It delivers similar performance to the calibrated Linear SVC with a simpler operational profile.

For an initial deployment focused on moderation support or user-feedback triage, this is the strongest balance of quality and practicality.

## Reproducibility
The original dataset used in this project, `imdb_reviews.tsv`, was provided through TripleTen and is **not included in this repository**.

Because of that, the notebook cannot be executed end-to-end unless the dataset is supplied locally.

### Expected dataset locations
The notebook is configured to look for the dataset in these portable locations:

1. `data/imdb_reviews.tsv`
2. `imdb_reviews.tsv`
3. `/datasets/imdb_reviews.tsv`

If you want to run the notebook locally, place the file in any one of those locations. The recommended option for this repository is:

- `data/imdb_reviews.tsv`

## Dependencies
The notebook requires the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `spacy`
- `tqdm`

It also requires the spaCy English model:

- `en_core_web_sm`

## Notes
- The notebook narrative was rewritten for portfolio presentation.
- The modeling logic and evaluated algorithms were not changed.
- The current limitation for full execution is the absence of the original TripleTen dataset.
