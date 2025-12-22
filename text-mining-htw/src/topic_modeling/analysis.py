import os
import re
import warnings
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK for text processing
import nltk
from nltk.corpus import stopwords

# Topic modeling libraries
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Machine learning for outcome modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
nltk.download('stopwords', quiet=True)


class AdvancedLegalAnalytics:
    """
    An advanced class for legal text analysis, focusing on topic modeling, trend analysis, and outcome modeling.
    """

    def __init__(self):
        # Expanded stopword list for better legal text processing
        self.legal_stopwords = list(stopwords.words('english')) + [
            'court', 'case', 'judge', 'plaintiff', 'defendant', 'v', 'vs', 'et', 'al',
            'opinion', 'decision', 'order', 'judgment', 'decree', 'holding', 'held',
            'see', 'also', 'however', 'therefore', 'furthermore', 'accordingly', 'moreover',
            'said', 'pursuant', 'herein', 'thereof', 'district', 'circuit', 'honorable',
            'appeal', 'appeals', 'appellant', 'appellee', 'argued', 'decided', 'law',
            'claims', 'filed', 'motion', 'summary', 'granted', 'denied', 'state', 'federal'
        ]
        self.processed_df = None
        self.topic_model = None

    def prepare_documents(self, df: pd.DataFrame, text_column: str, date_column: str) -> List[str]:
        """Prepares documents for topic modeling using the FULL text."""
        print("Preparing and cleaning documents (using full text)...")
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df.dropna(subset=[date_column, text_column], inplace=True)

        df['focused_document'] = df[text_column]

        df.dropna(subset=['focused_document'], inplace=True)
        df = df[df['focused_document'].str.len() > 50].copy()

        print(f"Processed {len(df)} cases using their full text.")
        self.processed_df = df
        return df['focused_document'].tolist()

    def create_outcome_variable(self):
        """Engineers the 'outcome' variable from the full document text."""
        print("\nEngineering outcome variable from text...")
        if self.processed_df is None: return

        def determine_outcome(text):
            text = text.lower()
            if 'reverse' in text or 'vacate' in text or 'set aside' in text: return 'Reversed/Vacated'
            if 'remand' in text: return 'Remanded'
            if 'affirm' in text or 'uphold' in text or 'we agree' in text: return 'Affirmed'
            if 'dismiss' in text: return 'Dismissed'
            if 'deny' in text or 'denied' in text or 'we disagree' in text: return 'Denied'
            return 'Other'

        self.processed_df['outcome'] = self.processed_df['focused_document'].apply(determine_outcome)
        print("Outcome variable created successfully.")
        print("Distribution of outcomes:")
        print(self.processed_df['outcome'].value_counts())

    def fit_topic_model(self, documents: List[str], min_topic_size: int = 20):
        """
        Trains the BERTopic model with enhanced configuration for better topic representation.
        """
        print("\nTraining Advanced BERTopic model...")

        vectorizer_model = CountVectorizer(stop_words=self.legal_stopwords, ngram_range=(1, 2))
        representation_model = KeyBERTInspired()
        embedding_model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")

        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            min_topic_size=min_topic_size,
            verbose=True,
            calculate_probabilities=True,
            nr_topics='auto'
        )
        topics, probabilities = self.topic_model.fit_transform(documents)

        self.processed_df['topic_id'] = topics
        self.processed_df['topic_probabilities'] = list(probabilities)
        topic_info = self.topic_model.get_topic_info()
        topic_map = topic_info.set_index('Topic')['Name'].to_dict()
        self.processed_df['topic_label'] = self.processed_df['topic_id'].map(topic_map)
        print("BERTopic model trained successfully.")

    def map_topics_to_judicial_classification(self):
        """
        Maps the machine-generated topics to a human-readable, standard judicial classification.
        """
        print("\nMapping topics to a standard judicial classification...")
        if self.topic_model is None: return

        classification_map = {
            "Criminal Law": ["crime", "sentence", "conviction", "guilty", "inmate", "prison", "warrant", "arrest"],
            "Contract Law": ["contract", "breach", "agreement", "damages", "parties", "insurance"],
            "Civil Procedure": ["summary judgment", "motion to dismiss", "jurisdiction", "discovery", "appeal",
                                "procedural"],
            "Employment Law": ["employment", "discrimination", "retaliation", "labor", "employee", "harassment", "ada"],
            "Constitutional & Civil Rights": ["constitutional", "due process", "first amendment", "equal protection",
                                              "civil rights", "speech"],
            "Tort Law": ["negligence", "injury", "tort", "damages", "liability"],
            "Intellectual Property": ["patent", "copyright", "trademark", "intellectual property"]
        }

        topic_info = self.topic_model.get_topic_info()
        id_to_label_map = {}
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1: continue
            topic_keywords = self.topic_model.get_topic(topic_id)
            if not topic_keywords: continue
            topic_words_str = ' '.join([word for word, _ in topic_keywords])

            mapped_category = "General/Other"
            for category, keywords in classification_map.items():
                if any(keyword in topic_words_str for keyword in keywords):
                    mapped_category = category
                    break
            id_to_label_map[topic_id] = mapped_category

        self.processed_df['judicial_category'] = self.processed_df['topic_id'].map(id_to_label_map).fillna(
            'General/Other')
        print("Judicial classification mapping complete.")

    def map_to_sustainability_topics(self):
        """
        Performs a second, more specific classification focused on Sustainability (ESG) topics.
        """
        print("\nClassifying cases by Sustainability (ESG) topics...")
        if 'focused_document' not in self.processed_df.columns:
            print("Focused document text not found. Skipping sustainability mapping.")
            return

        esg_map = {
            "Environmental/Climate": r'\b(climate|environmental|pollution|epa|conservation|emissions|green|sustainable|water|land|energy|resource)\b',
            "Social/Human Rights": r'\b(discrimination|labor|employee rights|civil rights|human rights|safety|consumer protection|community|public health)\b',
            "Corporate Governance/Finance": r'\b(governance|shareholder|board|fraud|securities|disclosure|compliance|fiduciary|investment|financial)\b'
        }

        def classify_esg(text):
            text = text.lower()
            for category, pattern in esg_map.items():
                if re.search(pattern, text):
                    return category
            return 'Non-ESG'

        self.processed_df['sustainability_category'] = self.processed_df['focused_document'].apply(classify_esg)
        print("Sustainability (ESG) classification complete.")
        print("Distribution of ESG categories:")
        print(self.processed_df['sustainability_category'].value_counts())

    def analyze_topic_trends(self, time_column: str, jurisdiction_column: str, output_dir: str):
        """Performs trend analysis of topics over time and by jurisdiction."""
        if self.topic_model is None: return
        print("\nAnalyzing topic trends...")
        try:
            timestamps = self.processed_df[time_column].to_list()
            docs = self.processed_df['focused_document'].to_list()
            topics_over_time = self.topic_model.topics_over_time(docs, timestamps, nr_bins=20)
            fig = self.topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
            fig.write_html(os.path.join(output_dir, "topic_trends_over_time.html"))
        except Exception as e:
            print(f"  - Could not create trends-over-time chart: {e}")

    def model_outcomes(self, positive_outcome: str):
        """Models the relationship between topics and case outcomes."""
        if self.topic_model is None or 'outcome' not in self.processed_df.columns: return
        print("\nModeling case outcomes...")
        X = np.array(self.processed_df['topic_probabilities'].to_list())
        y = (self.processed_df['outcome'] == positive_outcome).astype(int)
        if len(np.unique(y)) < 2: return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        model = LogisticRegression(random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        print("\n--- Outcome Model Classification Report ---")
        print(classification_report(y_test, model.predict(X_test)))

    def create_sample_data(self, n_samples=1000):
        """Creates sample data for demonstration purposes."""
        np.random.seed(42)
        opinion_texts = [
            "The district court's summary judgment is affirmed. The employment contract was unambiguous.",
            "We reverse the conviction and remand for a new trial. The defendant's constitutional rights were violated due to illegal search.",
            "The motion to dismiss is granted. The plaintiff failed to state a claim regarding EPA environmental protection regulations on water pollution.",
            "Finding no breach of contract, we affirm the lower court's decision in favor of the appellee. The shareholder agreement was valid.",
            "The inmate's petition is denied. The sentence was within the statutory limits for the crime committed.",
            "The board's decision was a breach of its fiduciary duty. We find for the shareholders and reverse."
        ]
        sample_df = pd.DataFrame({
            'date_filed': pd.to_datetime(np.random.choice(pd.date_range('2015-01-01', '2023-12-31'), n_samples)),
            'court_jurisdiction': np.random.choice(['NY District Court', 'CA 9th Circuit', 'Federal District TX'],
                                                   n_samples),
            'opinion_text': np.random.choice(opinion_texts, n_samples)
        })
        return sample_df
