import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

# Enable parallel processing
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count()))

# For force cleanup and parallel processing
import gc
import joblib
from joblib import Parallel, delayed
import atexit
from contextlib import contextmanager

# Context manager for proper resource cleanup
@contextmanager
def managed_joblib_context():
    """Context manager that ensures proper cleanup of joblib resources"""
    try:
        yield
    finally:
        try:
            # Comprehensive cleanup of joblib/loky resources
            from joblib.externals import loky
            
            # Clean up loky executors
            if hasattr(loky, 'get_reusable_executor'):
                try:
                    executor = loky.get_reusable_executor()
                    if executor is not None:
                        executor.shutdown(wait=True)
                except:
                    pass
            
            # Clean up any process pools
            if hasattr(loky.process_executor, '_process_pools'):
                for pool in list(loky.process_executor._process_pools.values()):
                    try:
                        pool.shutdown(wait=False)
                    except:
                        pass
                loky.process_executor._process_pools.clear()
            
            # Clean up resource tracker
            try:
                from joblib.externals.loky.backend import resource_tracker
                if hasattr(resource_tracker, '_resource_tracker') and resource_tracker._resource_tracker is not None:
                    resource_tracker._resource_tracker._stop()
                    resource_tracker._resource_tracker = None
            except:
                pass
            
            # Force garbage collection
            gc.collect()
        except:
            # Fallback cleanup
            gc.collect()

# Text preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Global function for parallel preprocessing with fallback stemmer
def preprocess_text_parallel(text: str, legal_stopwords: set) -> str:
    """Standalone function for parallel text preprocessing with fallback stemmer"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Use Porter Stemmer instead of WordNet Lemmatizer for parallel processing
    # Porter Stemmer is simpler and doesn't have serialization issues
    stemmer = PorterStemmer()
    
    # Convert to lowercase
    text = text.lower()
    
    # Enhanced citation removal patterns
    text = re.sub(r'\b\d+\s+[a-z\.]+\s*\d+[a-z]*\b', '', text)  # "123 F.3d 456"
    text = re.sub(r'\b\d{4}\s+[a-z]{2,4}\s+\d+\b', '', text)  # "2023 WL 123456"
    text = re.sub(r'\b\d+\s+[a-z]{2,4}\s+\d+d\s+\d+\b', '', text)  # "123 Ohio 3d 456"
    
    # Remove reporter metadata terms
    text = re.sub(r'\b(advance sheet|reporter|citation|slip opinion)s?\b', '', text)
    text = re.sub(r'\b(westlaw|lexis|findlaw)\b', '', text)
    
    # Remove state names and common geographic terms to reduce noise
    us_states = r'\b(alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|hampshire|jersey|mexico|york|carolina|dakota|ohio|oklahoma|oregon|pennsylvania|island|tennessee|texas|utah|vermont|virginia|washington|wisconsin|wyoming)\b'
    text = re.sub(us_states, '', text)
    
    # Remove case names in parentheses
    text = re.sub(r'\([^)]*v[^)]*\)', '', text)
    
    # Remove excessive whitespace and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenize and stem
    tokens = word_tokenize(text)
    
    # Filter tokens
    processed_tokens = []
    for token in tokens:
        # Skip short tokens and numbers
        if len(token) < 3 or token.isdigit():
            continue
        
        # Skip stop words
        if token in legal_stopwords:
            continue
        
        # Stem the token (simpler than lemmatization)
        stemmed = stemmer.stem(token)
        processed_tokens.append(stemmed)
    
    return ' '.join(processed_tokens)

class LDATopicModeling:
    """Latent Dirichlet Allocation (LDA) topic modeling for legal case analysis"""
    
    def __init__(self):
        # Note: Don't initialize lemmatizer here due to parallel processing issues
        # Comprehensive legal-specific stop words for better topic quality
        self.legal_stopwords = set(stopwords.words('english')).union({
            # Basic legal terms that appear in all cases
            'case', 'court', 'judge', 'justice', 'plaintiff', 'defendant', 'v', 'vs', 'versus',
            'appellant', 'appellee', 'petitioner', 'respondent', 'party', 'parties',
            
            # Geographic/jurisdictional terms (often not topic-specific)
            'state', 'county', 'city', 'district', 'circuit', 'municipal', 'federal',
            'united', 'states', 'america', 'usa', 'us',
            
            # US States to reduce geographic noise
            'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
            'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
            'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
            'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
            'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
            'hampshire', 'jersey', 'mexico', 'york', 'carolina', 'dakota',
            'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'island',
            'tennessee', 'texas', 'utah', 'vermont', 'virginia',
            'washington', 'wisconsin', 'wyoming',
            
            # Reporter and citation metadata terms
            'advance', 'sheet', 'reporter', 'citation', 'slip', 'opinion',
            'westlaw', 'lexis', 'findlaw', 'volume', 'page', 'published',
            'report', 'cite', 'app', 'tex', 'del', 'ohio', 'st',
            
            # Jurisdiction-specific artifacts
            'commonwealth', 'pcra', 'nebraska', 'pennsylvania', 'texas',
            'superior', 'appellate', 'supreme', 'dist', 'cir', 'ct',
            'div', 'dept', 'admin', 'rev', 'supp', 'misc',
            
            # Business entities (generic)
            'inc', 'corp', 'corporation', 'ltd', 'llc', 'company', 'co', 'enterprises',
            
            # Legal document structure terms
            'decision', 'order', 'holding', 'finding', 'conclusion',
            'section', 'subsection', 'paragraph', 'clause', 'footnote',
            'exhibit', 'appendix', 'attachment', 'schedule',
            
            # Common legal procedure terms (too generic)
            'motion', 'petition', 'complaint', 'answer', 'brief', 'memorandum',
            'record', 'transcript', 'hearing', 'trial', 'proceeding',
            
            # Legal transitional phrases
            'see', 'also', 'however', 'therefore', 'furthermore', 'moreover',
            'nevertheless', 'accordingly', 'thus', 'hence', 'whereas', 'wherein',
            'herein', 'thereof', 'therefor', 'pursuant', 'notwithstanding',
            
            # Numbers and ordinals
            'one', 'two', 'three', 'four', 'five', 'first', 'second', 'third',
            'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
            
            # Legal modals and auxiliaries
            'would', 'could', 'should', 'may', 'must', 'shall', 'might',
            'said', 'such', 'same', 'aforesaid', 'aforementioned',
            
            # Generic legal actions (too broad)
            'filed', 'granted', 'denied', 'dismissed', 'reversed', 'affirmed',
            'remanded', 'vacated', 'stayed', 'enjoined', 'ordered',
            
            # Citations and references
            'id', 'ibid', 'supra', 'infra', 'cf', 'eg', 'ie', 'etc', 'et', 'al',
            
            # Time references (often not substantive)
            'year', 'month', 'day', 'date', 'time', 'period', 'term',
            'years', 'months', 'days', 'dates', 'times', 'periods', 'terms'
        })
        
        # Legal domain patterns to preserve
        self.legal_patterns = [
            r'\b(?:constitutional|criminal|civil|contract|tort|property)\b',
            r'\b(?:due process|equal protection|probable cause|search and seizure)\b',
            r'\b(?:first amendment|second amendment|fourth amendment|fifth amendment)\b',
            r'\b(?:evidence|testimony|discovery|motion|appeal|jurisdiction)\b',
            r'\b(?:liability|damages|injunction|remedy|statute|regulation)\b',
            r'\b(?:precedent|stare decisis|dicta|ratio decidendi)\b'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess legal text for topic modeling"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Create lemmatizer locally to avoid serialization issues
        lemmatizer = WordNetLemmatizer()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove jurisdiction-specific terms and artifacts
        text = re.sub(r'\b(tex app|ohio st|penn super|del super|commonwealth|pcra)\b', '', text)
        text = re.sub(r'\b\w{2,}\s+(app|ct|cir|dist|super|supp)\b', '', text)  # "texas app", "ohio ct", etc.
        
        # Remove jurisdiction-specific terms and artifacts
        text = re.sub(r'\b(tex app|ohio st|penn super|del super|commonwealth|pcra)\b', '', text)
        text = re.sub(r'\b\w{2,}\s+(app|ct|cir|dist|super|supp)\b', '', text)  # "texas app", "ohio ct", etc.
        
        # Enhanced citation removal patterns
        text = re.sub(r'\b\d+\s+[a-z\.]+\s*\d+[a-z]*\b', '', text)  # "123 F.3d 456"
        text = re.sub(r'\b\d{4}\s+[a-z]{2,4}\s+\d+\b', '', text)  # "2023 WL 123456"
        text = re.sub(r'\b\d+\s+[a-z]{2,4}\s+\d+d\s+\d+\b', '', text)  # "123 Ohio 3d 456"
        
        # Remove reporter metadata terms
        text = re.sub(r'\b(advance sheet|reporter|citation|slip opinion)s?\b', '', text)
        text = re.sub(r'\b(westlaw|lexis|findlaw)\b', '', text)
        
        # Remove state names and common geographic terms to reduce noise
        us_states = r'\b(alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|hampshire|jersey|mexico|york|carolina|dakota|ohio|oklahoma|oregon|pennsylvania|island|tennessee|texas|utah|vermont|virginia|washington|wisconsin|wyoming)\b'
        text = re.sub(us_states, '', text)
        
        # Remove case names in parentheses
        text = re.sub(r'\([^)]*v[^)]*\)', '', text)
        
        # Remove excessive whitespace and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        
        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Skip short tokens and numbers
            if len(token) < 3 or token.isdigit():
                continue
            
            # Skip stop words
            if token in self.legal_stopwords:
                continue
            
            # Lemmatize
            lemmatized = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def prepare_documents(self, df: pd.DataFrame, text_column: str = 'opinion_text', 
                         cache_file: str = 'data/processed/preprocessed_docs.pkl', 
                         force_reprocess: bool = False) -> List[str]:
        """Prepare documents for topic modeling with high-quality preprocessing"""
        import pickle
        import os
        
        # Option to force reprocessing (useful when changing preprocessing algorithms)
        if force_reprocess and os.path.exists(cache_file):
            print(f"🔄 Force reprocessing - removing cache: {cache_file}")
            os.remove(cache_file)
        
        # Check if cached preprocessed documents exist
        if os.path.exists(cache_file):
            print(f"📁 Loading preprocessed documents from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    documents = pickle.load(f)
                print(f"✓ Loaded {len(documents)} documents from cache")
                return documents
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}. Reprocessing...")
        
        print("📝 Preprocessing documents with high-quality lemmatization...")
        print("ℹ️  Using WordNet Lemmatizer for proper word forms (slower but better quality)")
        
        all_texts = df[text_column].tolist()
        documents = []
        
        # Use sequential processing with WordNet Lemmatizer for better quality
        for idx, text in enumerate(all_texts):
            processed = self.preprocess_text(text)
            
            # Additional noise filtering for LDA
            tokens = processed.split()
            
            # Filter out HTML artifacts and encoding issues
            clean_tokens = []
            for token in tokens:
                # Skip HTML artifacts
                if any(artifact in token for artifact in ['quot', 'x27', 'amp', 'lt', 'gt']):
                    continue
                # Skip common names/locations that create noise topics
                if token in {'miller', 'smith', 'jones', 'brown', 'davis', 'wilson', 'moore', 'taylor', 'anderson', 'thomas',
                           'north', 'south', 'east', 'west', 'los', 'san', 'new', 'que', 'del', 'las', 'des'}:
                    continue
                # Skip very short or numeric tokens
                if len(token) < 3 or token.isdigit():
                    continue
                clean_tokens.append(token)
            
            cleaned_processed = ' '.join(clean_tokens)
            if len(clean_tokens) > 10:
                documents.append(cleaned_processed)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} documents...")
        
        # Cache the preprocessed documents
        if documents:  # Only cache if we have documents
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(documents, f)
                print(f"💾 Cached preprocessed documents to: {cache_file}")
            except Exception as e:
                print(f"⚠️ Failed to cache documents: {e}")
        
        print(f"✓ Prepared {len(documents)} documents for topic modeling")
        return documents
    
    def fit_lda_model(self, documents: List[str], n_topics: int = 12, 
                      max_features: int = 500) -> Tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
        """Fit LDA topic model with optimized parameters"""
        print(f"🔍 Fitting LDA model with {n_topics} topics...")
        
        # Vectorize documents with enhanced noise filtering
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=3,                   
            max_df=0.6,                 # More restrictive to filter common terms
            stop_words=list(self.legal_stopwords),  # Use our comprehensive legal stopwords
            ngram_range=(1, 2),
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic words, min 2 chars
        )
        
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        # Fit LDA with optimized parameters and proper resource management
        with joblib.parallel_backend('threading'):  # Use threading backend to avoid process leaks
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=50,                # Reduced from 100 to 50
                learning_method='online',   # Faster than 'batch'
                learning_offset=50.0,
                n_jobs=1                   # Use single job to avoid resource leaks
            )
            
            lda_output = lda.fit_transform(doc_term_matrix)
        
        print("✓ LDA model fitted successfully")
        return lda, vectorizer, lda_output
    
    def get_lda_topics(self, lda: LatentDirichletAllocation, 
                       vectorizer: CountVectorizer, n_words: int = 10) -> List[Dict]:
        """Extract topics from LDA model with normalized weights"""
        try:
            # Try the newer method first
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            # Fall back to older method for older sklearn versions
            feature_names = vectorizer.get_feature_names()
        
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            # Normalize topic weights to sum to 1
            normalized_topic = topic / topic.sum()
            
            top_words_idx = normalized_topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = [normalized_topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': top_weights,
                'top_words_string': ', '.join(top_words[:5])
            })
        
        return topics
    
    def analyze_topic_coherence(self, lda: LatentDirichletAllocation, 
                               doc_term_matrix, n_topics_range: range = range(5, 21)) -> Dict:
        """Analyze topic coherence for different numbers of topics"""
        print("📊 Analyzing topic coherence...")
        
        coherence_scores = []
        perplexity_scores = []
        
        for n_topics in n_topics_range:
            lda_temp = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=50
            )
            lda_temp.fit(doc_term_matrix)
            
            perplexity = lda_temp.perplexity(doc_term_matrix)
            perplexity_scores.append(perplexity)
            
        return {
            'n_topics': list(n_topics_range),
            'perplexity_scores': perplexity_scores
        }
    
    def visualize_lda_topics(self, topics: List[Dict], save_plot: bool = True):
        """Enhanced visualization of LDA topics with professional styling and proper legends"""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Legal Case Topic Analysis - LDA Model Results', fontsize=24, fontweight='bold', y=0.95)
        
        # Professional color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # 1. Topic word charts (top words bar chart)
        n_topics_to_show = min(4, len(topics))
        for i in range(n_topics_to_show):
            ax = axes[i//2, i%2]
            topic = topics[i]
            
            words = topic['words'][:8]
            weights = topic['weights'][:8]
            
            # Create horizontal bar chart with consistent colors
            bars = ax.barh(words, weights, color=colors[i % len(colors)], alpha=0.8, 
                          edgecolor='white', linewidth=1)
            
            # Style improvements with proper labels
            topic_title = f'Topic {i}: {topic["top_words_string"]}'
            ax.set_title(topic_title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Normalized Topic Weight (0-1)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Legal Keywords', fontsize=12, fontweight='bold')
            
            # Add value labels on bars
            for j, (word, weight) in enumerate(zip(words, weights)):
                ax.text(weight + max(weights) * 0.01, j, f'{weight:.3f}', 
                       va='center', fontweight='bold', fontsize=10, color='darkblue')
            
            # Enhanced styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_color('gray')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(0, max(weights) * 1.15)
            
            # Add legend for topic interpretation
            legal_categories = {
                0: 'Property/Zoning Law', 1: 'Civil Litigation', 2: 'Criminal Procedure', 3: 'Family Law',
                4: 'Legal Procedure', 5: 'Professional Ethics', 6: 'Employment Law', 7: 'Criminal Sentencing'
            }
            if i in legal_categories:
                ax.text(0.02, 0.98, f'Category: {legal_categories[i]}', 
                       transform=ax.transAxes, fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3),
                       verticalalignment='top')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_plot:
            plt.savefig('lda_topics_enhanced.png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print("📊 Enhanced LDA topics visualization saved as 'lda_topics_enhanced.png'")
        
        # Create a comprehensive topic overview with legend
        self.create_topic_overview(topics, save_plot)
        
        #plt.show()
    
    def create_topic_overview(self, topics: List[Dict], save_plot: bool = True):
        """Create a comprehensive overview of all topics with proper legends"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create a comprehensive topic words visualization
        n_topics = len(topics)
        y_positions = np.arange(n_topics)
        
        # Professional color scheme
        colors = plt.cm.Set3(np.linspace(0, 1, n_topics))
        
        # Legal domain categories for legend
        legal_domains = {
            'Property/Zoning': ['property', 'zoning', 'board', 'public', 'use'],
            'Civil Litigation': ['judgment', 'plaintiff', 'defendant', 'summary'],
            'Criminal Law': ['officer', 'police', 'search', 'evidence', 'sentence'],
            'Family Law': ['child', 'mother', 'father', 'parent', 'family'],
            'Employment': ['employee', 'work', 'employment', 'duty'],
            'Legal Process': ['counsel', 'attorney', 'rule', 'notice', 'file'],
            'Medical/Injury': ['medical', 'injury', 'claimant', 'board'],
            'Contract': ['contract', 'agreement', 'claim']
        }
        
        topic_categories = []
        for i, topic in enumerate(topics):
            # Categorize each topic
            topic_words = ' '.join(topic['words'][:5]).lower()
            category = 'General'
            
            for domain, keywords in legal_domains.items():
                if any(keyword in topic_words for keyword in keywords):
                    category = domain
                    break
            topic_categories.append(category)
            
            # Get top 3 words for display
            top_words = ', '.join(topic['words'][:3])
            avg_weight = np.mean(topic['weights'][:5])
            
            # Create horizontal bars with category-based styling
            bar = ax.barh(y_positions[i], avg_weight, color=colors[i], 
                         alpha=0.7, height=0.8, edgecolor='white', linewidth=1,
                         label=category if category not in [tc for tc in topic_categories[:i]] else "")
            
            # Add topic labels with category information
            label_text = f'Topic {i}: {top_words} ({category})'
            ax.text(avg_weight + max([np.mean(t['weights'][:5]) for t in topics]) * 0.02, 
                   y_positions[i], label_text, 
                   va='center', fontweight='bold', fontsize=11)
        
        # Styling with proper labels and legend
        ax.set_xlabel('Normalized Topic Strength (0-1)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Legal Topics', fontsize=14, fontweight='bold')
        ax.set_title('Complete Legal Topic Analysis - All Topics by Strength and Category', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Enhanced styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f'T{i}' for i in range(n_topics)])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Add legend for categories
        unique_categories = list(set(topic_categories))
        legend_elements = []
        category_colors = {cat: colors[topic_categories.index(cat)] for cat in unique_categories}
        
        for category in unique_categories:
            legend_elements.append(plt.Rectangle((0,0),1,1, 
                                               facecolor=category_colors[category], 
                                               alpha=0.7, label=category))
        
        ax.legend(handles=legend_elements, loc='lower right', 
                 title='Legal Domains', title_fontsize=12, fontsize=10,
                 frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('topic_overview_complete.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print("📊 Complete topic overview with legends saved as 'topic_overview_complete.png'")
        
        plt.close()

    def visualize_topic_distribution(self, lda_output: np.ndarray, topics: List[Dict], 
                                   save_plot: bool = True):
        """Enhanced visualization of topic distribution with proper legends and labels"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Legal Topic Distribution Analysis - Document Coverage and Patterns', 
                    fontsize=24, fontweight='bold', y=0.95)
        
        # Professional color palette
        colors_discrete = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                          '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', 
                          '#f7b6d3', '#c7c7c7']
        
        # 1. Average topic weights with enhanced styling and legend
        avg_weights = np.mean(lda_output, axis=0)
        topic_labels = [f"T{i}: {topics[i]['words'][0]}" for i in range(len(topics))]  # Use first keyword
        
        bars1 = axes[0, 0].bar(range(len(avg_weights)), avg_weights, 
                              color=colors_discrete[:len(avg_weights)], 
                              alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add value labels and proper formatting
        for i, (bar, weight, topic) in enumerate(zip(bars1, avg_weights, topics)):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(avg_weights) * 0.01,
                          f'{weight:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        axes[0, 0].set_title('Average Normalized Topic Weights Across All Legal Documents', 
                           fontsize=16, fontweight='bold', pad=20)
        axes[0, 0].set_xlabel('Legal Topic Categories', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Average Normalized Weight (0-1)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(range(len(topic_labels)))
        axes[0, 0].set_xticklabels([f'T{i}' for i in range(len(topics))], rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 2. Enhanced topic distribution heatmap with proper labels
        sample_size = min(30, lda_output.shape[0])
        sample_docs = lda_output[:sample_size]
        
        im = axes[0, 1].imshow(sample_docs.T, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        axes[0, 1].set_title(f'Topic Distribution Heatmap (Sample of {sample_size} Documents)', 
                           fontsize=16, fontweight='bold', pad=20)
        axes[0, 1].set_xlabel('Document Index', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Legal Topics', fontsize=12, fontweight='bold')
        axes[0, 1].set_yticks(range(len(topics)))
        axes[0, 1].set_yticklabels([f'T{i}' for i in range(len(topics))])
        
        # Enhanced colorbar with proper labeling
        cbar1 = plt.colorbar(im, ax=axes[0, 1], shrink=0.8)
        cbar1.set_label('Topic Probability', fontsize=11, fontweight='bold')
        
        # 3. Document count by dominant topic with category labels
        dominant_topics = np.argmax(lda_output, axis=1)
        topic_counts = Counter(dominant_topics)
        
        topics_list = list(topic_counts.keys())
        counts_list = list(topic_counts.values())
        
        bars3 = axes[1, 0].bar(topics_list, counts_list, 
                              color=[colors_discrete[i % len(colors_discrete)] for i in topics_list],
                              alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add count and percentage labels
        total_docs = sum(counts_list)
        for i, (topic_id, count) in enumerate(zip(topics_list, counts_list)):
            percentage = (count / total_docs) * 100
            axes[1, 0].text(topic_id, count + max(counts_list) * 0.01,
                          f'{count}\n({percentage:.1f}%)', 
                          ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        axes[1, 0].set_title('Document Distribution by Dominant Legal Topic', 
                           fontsize=16, fontweight='bold', pad=20)
        axes[1, 0].set_xlabel('Legal Topic Categories', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Number of Legal Documents', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 4. Topic diversity with enhanced labeling
        topic_diversities = []
        for i in range(lda_output.shape[1]):
            topic_docs = lda_output[:, i]
            topic_docs_norm = topic_docs / np.sum(topic_docs) if np.sum(topic_docs) > 0 else topic_docs
            diversity = -np.sum(topic_docs_norm * np.log(topic_docs_norm + 1e-10))
            topic_diversities.append(diversity)
        
        bars4 = axes[1, 1].bar(range(len(topic_diversities)), topic_diversities,
                              color=colors_discrete[:len(topic_diversities)],
                              alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add diversity score labels
        for i, (bar, diversity) in enumerate(zip(bars4, topic_diversities)):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(topic_diversities) * 0.01,
                          f'{diversity:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        axes[1, 1].set_title('Topic Diversity Scores (Higher = More Varied Coverage)', 
                           fontsize=16, fontweight='bold', pad=20)
        axes[1, 1].set_xlabel('Legal Topic Categories', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Diversity Score (Entropy)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(range(len(topic_diversities)))
        axes[1, 1].set_xticklabels([f'T{i}' for i in range(len(topic_diversities))], rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Style all subplots consistently
        for ax in axes.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_color('gray')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_plot:
            plt.savefig('topic_distribution_enhanced.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print("📊 Enhanced topic distribution with legends saved as 'topic_distribution_enhanced.png'")
        
        plt.close()
    
    def cluster_similar_cases(self, lda_output: np.ndarray, df: pd.DataFrame, 
                            n_clusters: int = 5, topics: List[Dict] = None) -> Dict:
        """Cluster cases based on topic distributions"""
        print(f"🔍 Clustering cases into {n_clusters} clusters...")
        
        # K-means clustering on topic distributions with proper resource management
        with joblib.parallel_backend('threading'):  # Use threading to avoid resource leaks
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(lda_output)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(lda_output, cluster_labels)
        
        # Analyze clusters
        cluster_analysis = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_sizes': Counter(cluster_labels),
            'clusters': {}
        }
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_docs = lda_output[cluster_mask]
            
            # Get cluster characteristics
            avg_topic_dist = np.mean(cluster_docs, axis=0)
            dominant_topic = np.argmax(avg_topic_dist)
            
            # Get sample cases from this cluster
            cluster_indices = np.where(cluster_mask)[0]
            sample_cases = []
            
            for idx in cluster_indices[:3]:  # Get first 3 cases as samples
                if idx < len(df):
                    case_info = self.clean_case_info(df.iloc[idx])
                    sample_cases.append(case_info)
            
            cluster_analysis['clusters'][cluster_id] = {
                'size': int(np.sum(cluster_mask)),
                'dominant_topic': int(dominant_topic),
                'avg_topic_distribution': avg_topic_dist.tolist(),
                'sample_cases': sample_cases
            }
        
        print(f"✓ Clustering complete (Silhouette score: {silhouette_avg:.3f})")
        
        # Create cluster visualization
        if topics is not None:
            self.visualize_clusters(lda_output, cluster_labels, cluster_analysis, topics)
        
        return cluster_analysis

    def visualize_clusters(self, lda_output: np.ndarray, cluster_labels: np.ndarray, 
                          cluster_analysis: Dict, topics: List[Dict]):
        """Create enhanced visualization of case clusters"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Legal Case Clustering Analysis', fontsize=24, fontweight='bold', y=0.95)
        
        # Color palette for clusters
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        
        # 1. Cluster size distribution
        cluster_sizes = [cluster_analysis['clusters'][i]['size'] for i in range(cluster_analysis['n_clusters'])]
        cluster_ids = list(range(cluster_analysis['n_clusters']))
        
        bars1 = axes[0, 0].bar(cluster_ids, cluster_sizes, 
                              color=colors[:len(cluster_ids)], 
                              alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add labels
        for i, (bar, size) in enumerate(zip(bars1, cluster_sizes)):
            height = bar.get_height()
            percentage = (size / sum(cluster_sizes)) * 100
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(cluster_sizes) * 0.01,
                          f'{size}\n({percentage:.1f}%)', 
                          ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        axes[0, 0].set_title('Cluster Size Distribution', fontsize=16, fontweight='bold', pad=20)
        axes[0, 0].set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Dominant topics per cluster
        dominant_topics_per_cluster = [cluster_analysis['clusters'][i]['dominant_topic'] 
                                     for i in range(cluster_analysis['n_clusters'])]
        
        bars2 = axes[0, 1].bar(cluster_ids, dominant_topics_per_cluster,
                              color=colors[:len(cluster_ids)],
                              alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add topic labels
        for i, (bar, topic_id) in enumerate(zip(bars2, dominant_topics_per_cluster)):
            height = bar.get_height()
            topic_words = ', '.join(topics[topic_id]['words'][:2]) if topic_id < len(topics) else 'N/A'
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                          f'T{topic_id}\n{topic_words}', 
                          ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        axes[0, 1].set_title('Dominant Topic per Cluster', fontsize=16, fontweight='bold', pad=20)
        axes[0, 1].set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Dominant Topic ID', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. 2D visualization using PCA for topic space
        from sklearn.decomposition import PCA
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2, random_state=42)
        lda_2d = pca.fit_transform(lda_output)
        
        # Create scatter plot
        for i in range(cluster_analysis['n_clusters']):
            cluster_mask = cluster_labels == i
            axes[1, 0].scatter(lda_2d[cluster_mask, 0], lda_2d[cluster_mask, 1], 
                             c=colors[i], label=f'Cluster {i}', alpha=0.7, s=60, edgecolors='white')
        
        axes[1, 0].set_title('Case Clusters in 2D Topic Space (PCA)', fontsize=16, fontweight='bold', pad=20)
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12, fontweight='bold')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cluster quality metrics
        # Calculate average topic distributions for each cluster
        avg_topic_distributions = []
        cluster_coherence_scores = []
        
        for i in range(cluster_analysis['n_clusters']):
            cluster_mask = cluster_labels == i
            cluster_docs = lda_output[cluster_mask]
            
            if len(cluster_docs) > 0:
                avg_dist = np.mean(cluster_docs, axis=0)
                avg_topic_distributions.append(avg_dist)
                
                # Calculate intra-cluster coherence (average pairwise cosine similarity)
                if len(cluster_docs) > 1:
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarities = cosine_similarity(cluster_docs)
                    # Get upper triangle (excluding diagonal)
                    upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
                    coherence = np.mean(upper_tri) if len(upper_tri) > 0 else 0
                else:
                    coherence = 1.0  # Single document cluster is perfectly coherent
                cluster_coherence_scores.append(coherence)
            else:
                avg_topic_distributions.append(np.zeros(lda_output.shape[1]))
                cluster_coherence_scores.append(0)
        
        bars4 = axes[1, 1].bar(cluster_ids, cluster_coherence_scores,
                              color=colors[:len(cluster_ids)],
                              alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add coherence score labels
        for i, (bar, score) in enumerate(zip(bars4, cluster_coherence_scores)):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(cluster_coherence_scores) * 0.01,
                          f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        axes[1, 1].set_title('Cluster Coherence Scores', fontsize=16, fontweight='bold', pad=20)
        axes[1, 1].set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Average Intra-cluster Similarity', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Style all subplots
        for ax in axes.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_color('gray')
        
        plt.tight_layout()
        plt.savefig('cluster_analysis_enhanced.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("📊 Enhanced cluster analysis saved as 'cluster_analysis_enhanced.png'")
    
    def clean_case_info(self, case_row: pd.Series) -> Dict:
        """Extract and clean case information from a dataframe row"""
        try:
            # Extract relevant information from the case row
            case_info = {
                'case_name': str(case_row.get('case_name', 'Unknown')),
                'court_type': str(case_row.get('court_type', 'Unknown')),
                'year_filed': str(case_row.get('year_filed', 'Unknown')),
                'case_id': str(case_row.get('id', case_row.name if hasattr(case_row, 'name') else 'Unknown'))
            }
            
            # Clean up the case name if it's too long
            if len(case_info['case_name']) > 50:
                case_info['case_name'] = case_info['case_name'][:47] + "..."
                
            return case_info
        except Exception as e:
            # Return minimal info if there's an error
            return {
                'case_name': 'Error extracting case info',
                'court_type': 'Unknown',
                'year_filed': 'Unknown',
                'case_id': str(case_row.name if hasattr(case_row, 'name') else 'Unknown')
            }
    
    def save_lda_results(self, lda_topics: List[Dict], cluster_analysis: Dict,
                        output_dir: str = "data/processed/"):
        """Save LDA topic modeling results"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                # Handle NaN values
                if np.isnan(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                # Handle NaN values in arrays
                clean_array = obj.tolist()
                return clean_nan_values(clean_array)
            elif obj is np.nan or (isinstance(obj, float) and np.isnan(obj)):
                return None
            else:
                return obj
        
        def clean_nan_values(obj):
            """Recursively clean NaN values from nested structures"""
            if isinstance(obj, list):
                return [clean_nan_values(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: clean_nan_values(v) for k, v in obj.items()}
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            else:
                return obj
        
        # Convert numpy types in the data
        clean_lda_results = convert_numpy_types({
            'topics': lda_topics,
            'cluster_analysis': cluster_analysis
        })
        
        lda_file = os.path.join(output_dir, "lda_topic_modeling.json")
        with open(lda_file, 'w') as f:
            json.dump(clean_lda_results, f, indent=2)
        print(f"💾 LDA results saved: {lda_file}")

def load_data(file_path: str) -> pd.DataFrame:
    """Load the legal cases dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded: {df.shape[0]} cases, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return pd.DataFrame()

def main():
    """Main execution function for LDA topic modeling"""
    with managed_joblib_context():
        print("🚀 LDA Legal Topic Modeling")
        print("="*50)
        
        # Initialize LDA topic modeling
        lda_modeler = LDATopicModeling()
        
        # Load data
        data_file = "/Users/liuyafei/Text_mining/Text-mining-HTW/data/processed/cleaned/parse_legal_cases.csv"
        df = load_data(data_file)
        
        if df.empty:
            print("❌ No data loaded. Please check the file path.")
            return
        
        # Random sample for topic modeling
       # max_cases = min(3000, len(df))
        #df_subset = df.sample(n=max_cases, random_state=42).copy()
        df_subset = df.copy()
        print(f"Processing {len(df_subset)} cases for LDA topic modeling")
        
        # Prepare documents
        documents = lda_modeler.prepare_documents(df_subset, 'opinion_text', force_reprocess=False)
        
        if len(documents) < 10:
            print("❌ Insufficient documents for LDA topic modeling")
            return
        
        print(f"\n📊 Starting LDA topic modeling with {len(documents)} documents...")
        
        # Fit LDA model
        lda, vectorizer, lda_output = lda_modeler.fit_lda_model(
            documents, n_topics=10, max_features=1000
        )
        
        # Extract topics
        lda_topics = lda_modeler.get_lda_topics(lda, vectorizer, n_words=10)
        
        # Print LDA topics
        print("\n🔍 LDA TOPICS:")
        for topic in lda_topics:
            print(f"Topic {topic['topic_id']}: {topic['top_words_string']}")
        
        # Visualize LDA results
        print(f"\n📊 Creating LDA visualizations...")
        lda_modeler.visualize_lda_topics(lda_topics)
        lda_modeler.visualize_topic_distribution(lda_output, lda_topics)
        
        # Cluster similar cases
        print(f"\n🔍 Clustering similar legal cases...")
        cluster_analysis = lda_modeler.cluster_similar_cases(lda_output, df_subset, n_clusters=5, topics=lda_topics)
        
        print(f"\n📋 CLUSTER ANALYSIS:")
        print(f"Silhouette Score: {cluster_analysis['silhouette_score']:.3f}")
        
        for cluster_id, cluster_info in cluster_analysis['clusters'].items():
            print(f"\nCluster {cluster_id} (Size: {cluster_info['size']}):")
            print(f"  Dominant Topic: {cluster_info['dominant_topic']}")
            if cluster_info['sample_cases']:
                print("  Sample Cases:")
                for case in cluster_info['sample_cases']:
                    print(f"    - {case['case_name']} ({case['court_type']}, {case['year_filed']})")
        
        # Save results
        print(f"\n💾 Saving LDA results...")
        lda_modeler.save_lda_results(lda_topics, cluster_analysis)
        
        print(f"\n✅ LDA topic modeling complete!")
        print(f"📊 Found {len(lda_topics)} LDA topics")
        print(f"🔍 Identified {cluster_analysis['n_clusters']} case clusters")
        
        print(f"\n🧹 Resource cleanup completed")

if __name__ == "__main__":
    main()