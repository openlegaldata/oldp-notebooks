import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')


class OutcomePredictor:
    """
    Uses textual features from legal opinions to predict judicial outcomes
    and identifies the most influential words and phrases.
    """

    def __init__(self, processed_df):
        self.df = processed_df.copy()

    def _prepare_data(self, target_outcome):
        """Prepares data for modeling, ensuring no data leakage."""
        print(f"Preparing data to predict outcome: '{target_outcome}'")
        self.df.dropna(subset=['focused_document', 'outcome'], inplace=True)
        X = self.df['focused_document']
        y = (self.df['outcome'] == target_outcome).astype(int)
        if np.sum(y) < 10:
            print(f"Warning: Very few samples for the target outcome '{target_outcome}'. Model may not be reliable.")
            return None, None, None, None
        return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    def train_and_evaluate_model(self, X_train, X_test, y_train, y_test):
        """Trains a TF-IDF and Logistic Regression model and evaluates its performance."""
        print("Training outcome prediction model...")
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=5)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        self.model = LogisticRegression(random_state=42, class_weight='balanced', C=1.0)
        self.model.fit(X_train_tfidf, y_train)
        y_pred = self.model.predict(X_test_tfidf)
        print("\n--- Outcome Prediction Model Report ---")
        print(classification_report(y_test, y_pred))

    def identify_predictive_features(self, target_outcome, save_path):
        """Identifies and visualizes the most predictive words/phrases for the target outcome."""
        print(f"Identifying most predictive textual features for '{target_outcome}'...")
        if not hasattr(self, 'model') or not hasattr(self, 'vectorizer'):
            print("Model has not been trained yet. Skipping feature identification.")
            return

        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]
        feature_importance = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
        top_predictors = feature_importance.sort_values(by='coefficient', ascending=False).head(15)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='coefficient', y='feature', data=top_predictors, palette='Blues_r')
        plt.title(f'Top 15 Textual Predictors of "{target_outcome}" Outcome', fontsize=16, fontweight='bold')
        plt.xlabel('Predictive Weight (Coefficient)', fontsize=12)
        plt.ylabel('Word / Phrase', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Predictive features chart saved to: {save_path}")

    def run_prediction_analysis(self, output_dir, target_outcome):
        """Runs the full outcome prediction pipeline."""
        X_train, X_test, y_train, y_test = self._prepare_data(target_outcome=target_outcome)
        if X_train is not None:
            self.train_and_evaluate_model(X_train, X_test, y_train, y_test)
            self.identify_predictive_features(
                target_outcome=target_outcome,
                save_path=os.path.join(output_dir, f'predictive_features_for_{target_outcome.replace("/", "")}.png')
            )
