# Zomato Add-On Recommendation System

A machine learning project that predicts whether a customer will accept a recommended add-on item (beverage, dessert, side, etc.) during checkout, using a synthetically generated food-delivery dataset. Includes model training/comparison, ranking and business-impact evaluation, and a Gradio demo app.

## Overview

The notebook (`zomato.ipynb`) builds an end-to-end pipeline for an "add-on recommendation" use case similar to what a food delivery app shows at checkout ("Add a drink?", "Add dessert?"):

1. **Synthetic data generation** — simulates 1,000,000 rows of user, restaurant, cart, candidate-item, and contextual features, with a rule-based acceptance label.
2. **Model building** — trains and compares Logistic Regression, Random Forest, XGBoost, and LightGBM classifiers to predict add-on acceptance probability.
3. **Evaluation** — ranking metrics (Precision@k, Recall@k, NDCG@k), classification metrics (AUC, accuracy, F1 with threshold tuning), and business metrics (acceptance rate, AOV lift, CTR, cart-to-order rate, latency, coverage).
4. **Gradio frontend** — an interactive demo where a user builds a cart from a sample menu and gets a live add-on recommendation.

## Pipeline

### 1. Synthetic Dataset Generation
Generates features across five groups:
- **User features:** segment (Budget/Premium/Occasional), order frequency, recency, average order value, preferred cuisine/zone, session clicks
- **Restaurant features:** cuisine type, price range, rating, order volume, chain flag, city/zone
- **Cart context:** item counts by category, cart total value
- **Candidate item features:** category, price, veg flag, popularity score
- **Contextual features:** hour, day of week, meal time, weekend flag
- **Interaction features:** cuisine/zone match flags, missing beverage/dessert flags, price-to-cart ratio, affinity scores

A rule-based probability model then generates the binary `label` (add-on accepted or not).

### 2. Model Building
- Data is grouped into **sessions** of 20 candidate items each, with a **session-safe train/test split** (no session leaks across sets).
- A second, more realistic label (`prob_accept` via sigmoid + binomial sampling) is generated for training.
- Three separate preprocessing pipelines are built (scaled + one-hot for linear models, unscaled + one-hot for trees, raw categorical/native handling for boosting models).
- Models trained: **Logistic Regression**, **Random Forest**, **XGBoost**, **LightGBM**.

### 3. Evaluation
- **Ranking metrics:** Precision@k, Recall@k, NDCG@k (computed per session)
- **Classification metrics:** AUC, accuracy, best-threshold F1 search
- **Business metrics:** Add-on acceptance rate, AOV (average order value) lift vs. random baseline, click-through rate, cart-to-order rate, inference latency, feature dimensionality, session coverage

### 4. Gradio App
An interactive demo: users pick items from a sample menu, and the app recommends add-ons using the trained models via wrapper prediction functions for XGBoost and LightGBM.

## Requirements

\`\`\`
numpy
pandas
tqdm
scikit-learn
xgboost
lightgbm
gradio
\`\`\`

## Usage

1. Install dependencies:
   \`\`\`bash
   pip install numpy pandas tqdm scikit-learn xgboost lightgbm gradio
   \`\`\`
2. Run the notebook cells in order: data generation → model training → evaluation → Gradio app.
3. Launch the Gradio app cell to try the interactive add-on recommender in your browser.

## Notes

- The dataset is fully **synthetic**, generated with a fixed random seed (`42`) for reproducibility — it does not contain real Zomato data.
- `N_ROWS` (default 1,000,000) can be reduced for faster iteration during development.
