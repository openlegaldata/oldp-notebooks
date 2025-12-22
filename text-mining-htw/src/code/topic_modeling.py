import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
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

# Text preprocessing and machine learning
from sklearn.feature_extraction.text import CountVectorizer
import umap

# BERTopic
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from bertopic.representation import KeyBERTInspired
    BERTOPIC_AVAILABLE = True
    print("✓ BERTopic available")
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("❌ BERTopic not available. Install with: pip install bertopic")

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

class LegalTopicModeling:
    """Topic modeling for legal case analysis"""
    
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
    
    
    def prepare_documents_for_bertopic(self, df: pd.DataFrame, text_column: str = 'opinion_text') -> List[str]:
        """Prepare documents for BERTopic with legal-optimized preprocessing"""
        print("📝 Preparing documents for BERTopic (legal-optimized preprocessing)...")
        
        documents = []
        for idx, text in enumerate(df[text_column]):
            if pd.isna(text) or not isinstance(text, str):
                continue
            
            # Enhanced preprocessing for better legal topic extraction
            processed = text.lower()
            
            # Preserve key legal phrases before token filtering
            legal_phrases = [
                'summary judgment', 'due process', 'probable cause', 'search warrant',
                'miranda rights', 'habeas corpus', 'class action', 'preliminary injunction',
                'temporary restraining', 'punitive damages', 'attorney fees', 'motion dismiss',
                'personal jurisdiction', 'subject matter jurisdiction', 'res judicata',
                'collateral estoppel', 'statute limitations', 'qualified immunity',
                'sovereign immunity', 'commerce clause', 'equal protection', 'first amendment',
                'fourth amendment', 'fifth amendment', 'fourteenth amendment',
                'breach contract', 'specific performance', 'injunctive relief',
                'criminal law', 'civil procedure', 'constitutional law', 'contract law',
                'tort law', 'property law', 'family law', 'employment law'
            ]
            
            # Replace phrases with underscore versions to preserve them
            phrase_map = {}
            for phrase in legal_phrases:
                if phrase in processed:
                    underscore_phrase = phrase.replace(' ', '_')
                    phrase_map[underscore_phrase] = phrase
                    processed = processed.replace(phrase, underscore_phrase)
            
            # Remove legal citations more comprehensively
            processed = re.sub(r'\b\d+\s+[a-z\.]+\s*\d+[a-z]*\b', '', processed)  # e.g., "123 F.3d 456"
            processed = re.sub(r'\b\d{4}\s+[a-z]{2,4}\s+\d+\b', '', processed)    # e.g., "2023 WL 123456"
            processed = re.sub(r'\b\d+\s+[a-z]{2,4}\s+\d+d\s+\d+\b', '', processed)  # e.g., "123 Ohio 3d 456"
            
            # Remove case names and party references
            processed = re.sub(r'\([^)]*v\.?\s+[^)]*\)', '', processed)             # Case names in parens
            processed = re.sub(r'\b[a-z]+\s+v\.?\s+[a-z]+\b', '', processed)       # Simple "A v. B" patterns
            
            # Remove court-specific abbreviations that create noise topics
            processed = re.sub(r'\b(dist|cir|app|ct|sup|fed|admin)\b\.?', '', processed)
            processed = re.sub(r'\b\d+(st|nd|rd|th)\s+(dist|cir|app|ct)\b\.?', '', processed)
            
            # Remove excessive punctuation and normalize spaces
            processed = re.sub(r'[^\w\s_]', ' ', processed)  # Keep underscores for phrases
            processed = re.sub(r'\s+', ' ', processed).strip()
            
            # Filter out very short tokens that don't add semantic value
            tokens = processed.split()
            filtered_tokens = [token for token in tokens if len(token) > 2 and not token.isdigit()]
            
            # Remove legal stopwords but preserve legal phrases
            meaningful_tokens = []
            for token in filtered_tokens:
                if token in phrase_map:  # It's a preserved legal phrase
                    meaningful_tokens.append(token)
                elif token not in self.legal_stopwords:
                    meaningful_tokens.append(token)
            
            # Restore original phrases
            processed_doc = ' '.join(meaningful_tokens)
            for underscore_phrase, original_phrase in phrase_map.items():
                processed_doc = processed_doc.replace(underscore_phrase, original_phrase)
            
            # Keep documents with substantial meaningful content
            if len(meaningful_tokens) > 25:  # Slightly lower threshold to include more documents
                documents.append(processed_doc)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} documents for BERTopic...")
        
        print(f"✓ Prepared {len(documents)} documents for BERTopic with enhanced preprocessing")
        return documents
    
    def preprocess_text_for_bertopic(self, text: str) -> str:
        """Preprocess legal text specifically for BERTopic with legal phrase preservation"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Enhanced preprocessing for better legal topic extraction
        processed = text.lower()
        
        # Preserve key legal phrases before token filtering
        legal_phrases = [
            'summary judgment', 'due process', 'probable cause', 'search warrant',
            'miranda rights', 'habeas corpus', 'class action', 'preliminary injunction',
            'temporary restraining', 'punitive damages', 'attorney fees', 'motion dismiss',
            'personal jurisdiction', 'subject matter jurisdiction', 'res judicata',
            'collateral estoppel', 'statute limitations', 'qualified immunity',
            'sovereign immunity', 'commerce clause', 'equal protection', 'first amendment',
            'fourth amendment', 'fifth amendment', 'fourteenth amendment',
            'breach contract', 'specific performance', 'injunctive relief',
            'criminal law', 'civil procedure', 'constitutional law', 'contract law',
            'tort law', 'property law', 'family law', 'employment law'
        ]
        
        # Replace phrases with underscore versions to preserve them
        phrase_map = {}
        for phrase in legal_phrases:
            if phrase in processed:
                underscore_phrase = phrase.replace(' ', '_')
                phrase_map[underscore_phrase] = phrase
                processed = processed.replace(phrase, underscore_phrase)
        
        # Remove legal citations more comprehensively
        processed = re.sub(r'\b\d+\s+[a-z\.]+\s*\d+[a-z]*\b', '', processed)  # e.g., "123 F.3d 456"
        processed = re.sub(r'\b\d{4}\s+[a-z]{2,4}\s+\d+\b', '', processed)    # e.g., "2023 WL 123456"
        processed = re.sub(r'\b\d+\s+[a-z]{2,4}\s+\d+d\s+\d+\b', '', processed)  # e.g., "123 Ohio 3d 456"
        
        # Remove case names and party references
        processed = re.sub(r'\([^)]*v\.?\s+[^)]*\)', '', processed)             # Case names in parens
        processed = re.sub(r'\b[a-z]+\s+v\.?\s+[a-z]+\b', '', processed)       # Simple "A v. B" patterns
        
        # Remove court-specific abbreviations that create noise topics
        processed = re.sub(r'\b(dist|cir|app|ct|sup|fed|admin)\b\.?', '', processed)
        processed = re.sub(r'\b\d+(st|nd|rd|th)\s+(dist|cir|app|ct)\b\.?', '', processed)
        
        # Remove excessive punctuation and normalize spaces
        processed = re.sub(r'[^\w\s_]', ' ', processed)  # Keep underscores for phrases
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # Filter out very short tokens that don't add semantic value
        tokens = processed.split()
        filtered_tokens = [token for token in tokens if len(token) > 2 and not token.isdigit()]
        
        # Remove legal stopwords but preserve legal phrases
        meaningful_tokens = []
        for token in filtered_tokens:
            if token in phrase_map:  # It's a preserved legal phrase
                meaningful_tokens.append(token)
            elif token not in self.legal_stopwords:
                meaningful_tokens.append(token)
        
        # Restore original phrases
        processed_doc = ' '.join(meaningful_tokens)
        for underscore_phrase, original_phrase in phrase_map.items():
            processed_doc = processed_doc.replace(underscore_phrase, original_phrase)
        
        return processed_doc

    def fit_bertopic_model_with_optimization(self, documents: List[str], data_size: int = None, max_iterations: int = 3) -> Optional[BERTopic]:
        """Fit BERTopic model with automatic optimization to prevent dominant topics"""
        if not BERTOPIC_AVAILABLE:
            print("❌ BERTopic not available")
            return None
        
        # Check minimum dataset size
        if len(documents) < 20:
            print(f"⚠️ Dataset too small for BERTopic ({len(documents)} docs). Need at least 20 documents.")
            return None
        
        # Use the best performing parameters from optimization
        best_params = {'min_topic_size': 28, 'n_components': 8, 'n_neighbors': 12}
        print(f"🎯 Using optimized parameters: {best_params}")
        
        model = self.fit_single_bertopic_model(
            documents, 
            data_size, 
            min_topic_size=best_params['min_topic_size'],
            n_components=best_params['n_components'], 
            n_neighbors=best_params['n_neighbors']
        )
        
        if model is not None:
            # Evaluate the final model
            topics, _ = model.fit_transform(documents)
            quality_metrics = self.evaluate_topic_quality(model, documents, topics)
            print(f"📊 Final Model Quality Score: {quality_metrics['overall_score']:.1f}/100")
        
        return model

    def fit_single_bertopic_model(self, documents: List[str], data_size: int = None, 
                                 min_topic_size: int = 10, n_components: int = 12, 
                                 n_neighbors: int = 8) -> Optional[BERTopic]:
        """Fit a single BERTopic model with specified parameters"""
        if not BERTOPIC_AVAILABLE:
            print("❌ BERTopic not available")
            return None
        
        if data_size is None:
            data_size = len(documents)
        
        print(f"🤖 Fitting BERTopic model with min_topic_size={min_topic_size}...")
        
        # Use legal-BERT model for better legal text understanding
        try:
            # Try to load a legal-specific model - using a known working legal model
            print("🔍 Attempting to load legal-BERT model...")
            
            # Try different legal BERT model options (sentence-transformers compatible)
            legal_models = [
                'sentence-transformers/all-mpnet-base-v2',  # Best general model for semantic search
                'sentence-transformers/all-MiniLM-L12-v2',  # Good balance of speed/quality
                'sentence-transformers/all-MiniLM-L6-v2',   # Faster, still good quality
                'nlpaueb/legal-bert-base-uncased'  # Legal BERT (will auto-create pooling)
            ]
            
            sentence_model = None
            for model_name in legal_models:
                try:
                    print(f"🔍 Trying {model_name}...")
                    sentence_model = SentenceTransformer(model_name)
                    print(f"✓ Successfully loaded {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {model_name}: {e}")
                    continue
            
            # If no legal model works, use general model
            if sentence_model is None:
                raise Exception("No legal BERT models available")
                
        except Exception as e:
            print(f"⚠️ Legal BERT models not available: {e}")
            # Fallback to all-MiniLM-L6-v2
            try:
                print("🔄 Falling back to all-MiniLM-L6-v2...")
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ Using all-MiniLM-L6-v2 as fallback")
            except Exception as e2:
                print(f"⚠️ Failed to load fallback model: {e2}")
                return None
        
        # Optimized vectorizer parameters to prevent topic dominance
        print(f"📊 Processing {data_size} documents, {len(documents)} preprocessed documents")
        
        num_docs = len(documents)
        
        # More restrictive parameters to create balanced topics
        if num_docs < 50:
            min_df = 1
            max_df = max(2, int(num_docs * 0.8))  # More restrictive max_df
            max_features = min(150, num_docs * 4)
        elif num_docs < 100:
            min_df = 1
            max_df = max(3, int(num_docs * 0.7))  # More restrictive
            max_features = min(300, num_docs * 6)
        elif num_docs < 200:
            min_df = 2
            max_df = max(5, int(num_docs * 0.6))  # More restrictive
            max_features = min(500, num_docs * 8)
        elif num_docs < 500:
            min_df = 2
            max_df = 0.5  # Much more restrictive to prevent common words dominating
            max_features = 800
        elif num_docs < 1000:
            min_df = 3
            max_df = 0.4  # Very restrictive
            max_features = 1500
        else:
            min_df = 3
            max_df = 0.35  # Very restrictive to force topic specialization
            max_features = 2500
        
        print(f"📊 Vectorizer settings: min_df={min_df}, max_df={max_df}, max_features={max_features}")
        
        # Optimized vectorizer for legal text with legal phrases
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 3),  # Include 3-grams for complex legal phrases
            stop_words=list(self.legal_stopwords),  # Use our comprehensive legal stopwords
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b',  # Only words with 3+ characters
            analyzer='word',
            strip_accents='unicode'
        )
        
        # Set up UMAP for better dimensionality reduction
        try:
            import umap
            # More aggressive UMAP parameters to prevent dominant clusters
            umap_model = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.1,
                metric='cosine',
                spread=1.5,
                random_state=42
            )
            print("✓ Using optimized UMAP parameters")
        except ImportError:
            umap_model = None
            print("⚠️ UMAP not available, using default dimensionality reduction")
        
        # Set up HDBSCAN for better clustering
        try:
            import hdbscan
            # More sensitive HDBSCAN to create balanced clusters
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=min_topic_size,
                min_samples=max(2, min_topic_size // 3),  # Adaptive min_samples
                metric='euclidean',
                cluster_selection_method='leaf',  # More sensitive to smaller clusters
                alpha=1.5,  # Higher alpha for more conservative clustering
                prediction_data=True
            )
            print(f"✓ Using HDBSCAN with min_cluster_size={min_topic_size}")
        except ImportError:
            hdbscan_model = None
            print("⚠️ HDBSCAN not available, using default clustering")
        
        # Set up KeyBERTInspired for better keyword quality
        try:
            representation_model = KeyBERTInspired()
            print("✓ Using KeyBERTInspired for better keyword quality")
        except:
            representation_model = None
            print("⚠️ KeyBERTInspired not available, using default representation")
        
        # Force more topics to prevent dominance
        target_topics = max(8, min(25, data_size // 150))  # Adaptive topic count
        print(f"🎯 Target number of topics: {target_topics}")
        
        # Initialize BERTopic with optimized parameters for legal text
        topic_model_params = {
            "language": "english",
            "embedding_model": sentence_model,
            "vectorizer_model": vectorizer_model,
            "calculate_probabilities": True,
            "verbose": True,
            "nr_topics": target_topics  # Force specific number of topics
        }
        
        if representation_model is not None:
            topic_model_params["representation_model"] = representation_model
        
        if umap_model is not None:
            topic_model_params["umap_model"] = umap_model
        
        if hdbscan_model is not None:
            topic_model_params["hdbscan_model"] = hdbscan_model
        else:
            topic_model_params["min_topic_size"] = min_topic_size
        
        topic_model = BERTopic(**topic_model_params)
        
        # Fit model with error handling
        print("🔄 Training BERTopic model...")
        try:
            topics, probs = topic_model.fit_transform(documents)
            
            n_topics = len(set(topics))
            n_outliers = sum(1 for t in topics if t == -1)
            
            print(f"✅ BERTopic model training complete!")
            print(f"📊 Found {n_topics} topics ({n_outliers} outlier documents)")
            
            return topic_model
            
        except Exception as e:
            print(f"❌ Error training BERTopic model: {e}")
            return None

    def evaluate_topic_quality(self, topic_model: BERTopic, documents: List[str], topics: List[int]):
        """Evaluate the quality of generated topics for legal text"""
        print("\n🔍 EVALUATING TOPIC QUALITY:")
        
        topic_info = topic_model.get_topic_info()
        total_topics = len(topic_info) - 1  # Exclude outlier topic (-1)
        
        # 1. Topic Balance Analysis
        topic_counts = Counter(topics)
        topic_sizes = [topic_counts[i] for i in range(0, total_topics)]
        
        if topic_sizes:
            largest_topic_pct = max(topic_sizes) / len(documents) * 100
            smallest_topic_pct = min(topic_sizes) / len(documents) * 100
            
            print(f"📊 Topic Balance:")
            print(f"   • Total topics: {total_topics}")
            print(f"   • Largest topic covers: {largest_topic_pct:.1f}% of documents")
            print(f"   • Smallest topic covers: {smallest_topic_pct:.1f}% of documents")
            
            if largest_topic_pct > 35:
                print(f"   ⚠️  Warning: Largest topic is too dominant ({largest_topic_pct:.1f}%)")
            else:
                print(f"   ✓ Good topic balance")
        
        # 2. Legal Domain Analysis
        legal_domains = {
            'criminal': ['criminal', 'sentence', 'plea', 'defendant', 'prosecution', 'guilty', 'conviction'],
            'civil': ['civil', 'damages', 'liability', 'negligence', 'tort', 'plaintiff'],
            'contract': ['contract', 'agreement', 'breach', 'performance', 'consideration'],
            'property': ['property', 'real estate', 'lease', 'landlord', 'tenant', 'ownership'],
            'family': ['custody', 'divorce', 'marriage', 'child', 'parent', 'support'],
            'constitutional': ['constitutional', 'amendment', 'rights', 'freedom', 'due process'],
            'employment': ['employment', 'employee', 'employer', 'workplace', 'discrimination'],
            'corporate': ['corporation', 'shareholder', 'director', 'securities', 'merger']
        }
        
        domain_topics = {}
        print(f"\n🏛️ Legal Domain Coverage:")
        
        for topic_id in range(min(10, total_topics)):  # Check top 10 topics
            topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
            topic_words_str = ', '.join(topic_words[:5])
            
            # Find matching legal domains
            matching_domains = []
            for domain, keywords in legal_domains.items():
                if any(keyword in ' '.join(topic_words) for keyword in keywords):
                    matching_domains.append(domain)
            
            if matching_domains:
                domain_topics[topic_id] = matching_domains
                print(f"   Topic {topic_id}: {topic_words_str} → {', '.join(matching_domains)}")
            else:
                print(f"   Topic {topic_id}: {topic_words_str} → [Unclear domain]")
        
        # 3. Topic Coherence Assessment
        coherent_topics = 0
        print(f"\n🎯 Topic Coherence Assessment:")
        
        for topic_id in range(min(5, total_topics)):  # Check top 5 topics
            topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:8]]
            
            # Simple coherence check: do words relate to similar legal concepts?
            if len(set(topic_words).intersection(set([
                # Group related legal terms
                'sentence', 'plea', 'criminal', 'defendant',  # Criminal
                'contract', 'agreement', 'breach', 'performance',  # Contract
                'property', 'lease', 'tenant', 'landlord',  # Property
                'custody', 'child', 'parent', 'family'  # Family
            ]))) >= 2:
                coherent_topics += 1
                print(f"   ✓ Topic {topic_id}: Coherent ({', '.join(topic_words[:4])})")
            else:
                print(f"   ? Topic {topic_id}: Mixed ({', '.join(topic_words[:4])})")
        
        # 4. Overall Quality Score
        balance_score = min(100, (100 - largest_topic_pct + 10)) if topic_sizes else 50
        domain_score = len(domain_topics) / min(10, total_topics) * 100
        coherence_score = coherent_topics / min(5, total_topics) * 100
        
        overall_score = (balance_score + domain_score + coherence_score) / 3
        
        print(f"\n📈 QUALITY METRICS:")
        print(f"   • Balance Score: {balance_score:.1f}/100")
        print(f"   • Domain Coverage: {domain_score:.1f}/100")
        print(f"   • Coherence Score: {coherence_score:.1f}/100")
        print(f"   • Overall Quality: {overall_score:.1f}/100")
        
        if overall_score >= 75:
            print("   🎉 Excellent topic quality!")
        elif overall_score >= 60:
            print("   👍 Good topic quality")
        elif overall_score >= 45:
            print("   ⚠️  Fair topic quality - consider parameter tuning")
        else:
            print("   ❌ Poor topic quality - model needs optimization")
        
        return {
            'total_topics': total_topics,
            'balance_score': balance_score,
            'domain_score': domain_score,
            'coherence_score': coherence_score,
            'overall_score': overall_score,
            'domain_topics': domain_topics
        }
    
    
    

    
    def visualize_bertopic_results(self, topic_model: BERTopic, documents: List[str], 
                                 save_plot: bool = True):
        """Visualize BERTopic results"""
        if not topic_model:
            return
        
        try:
            # Topic visualization
            fig1 = topic_model.visualize_topics()
            if save_plot:
                fig1.write_html("bertopic_topics.html")
                print("📊 BERTopic topics saved as 'bertopic_topics.html'")
            #fig1.show()
            
            # Topic hierarchy
            hierarchical_topics = topic_model.hierarchical_topics(documents)
            fig2 = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
            if save_plot:
                fig2.write_html("bertopic_hierarchy.html")
                print("📊 BERTopic hierarchy saved as 'bertopic_hierarchy.html'")
            #fig2.show()
            
            # Topic heatmap
            fig3 = topic_model.visualize_heatmap()
            if save_plot:
                fig3.write_html("bertopic_heatmap.html")
                print("📊 BERTopic heatmap saved as 'bertopic_heatmap.html'")
            #fig3.show()
            
        except Exception as e:
            print(f"Error creating BERTopic visualizations: {e}")
    

    
    
    def save_results(self, bertopic_model: Optional[BERTopic] = None, 
                    output_dir: str = "data/processed/"):
        """Save BERTopic modeling results"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save BERTopic results
        if bertopic_model:
            bertopic_file = os.path.join(output_dir, "bertopic_model")
            bertopic_model.save(bertopic_file)
            print(f"💾 BERTopic model saved: {bertopic_file}")
            
            # Save topic info
            topics_info = bertopic_model.get_topic_info()
            topics_file = os.path.join(output_dir, "bertopic_topics.csv")
            topics_info.to_csv(topics_file, index=False)
            print(f"💾 BERTopic topics saved: {topics_file}")
        else:
            print("⚠️ No BERTopic model to save")

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
    """Main execution function with proper resource management for BERT topic modeling"""
    with managed_joblib_context():
        print("🚀 Legal BERT Topic Modeling")
        print("="*60)
        
        # Initialize topic modeling
        topic_modeler = LegalTopicModeling()
        
        # Load data
        data_file = "Text-mining-HTW/data/processed/cleaned/parse_legal_cases.csv"
        df = load_data(data_file)
        
        if df.empty:
            print("❌ No data loaded. Please check the file path.")
            return
        
        # Random sample for topic modeling
        max_cases = min(3000, len(df))
        df_subset = df.sample(n=max_cases, random_state=42).copy()
        print(f"Processing {len(df_subset)} cases for BERT topic modeling")
        
        # BERTopic Modeling
        if BERTOPIC_AVAILABLE:
            print("\n" + "="*40)
            print("BERTOPIC MODELING")
            print("="*40)
            
            # Prepare documents with legal-optimized preprocessing for BERTopic
            bertopic_documents = topic_modeler.prepare_documents_for_bertopic(df_subset, 'opinion_text')
            
            if len(bertopic_documents) < 20:
                print("❌ Insufficient documents for BERTopic modeling (need at least 20)")
                return
            
            bertopic_model = topic_modeler.fit_bertopic_model_with_optimization(
                bertopic_documents, 
                data_size=len(df_subset)
            )
            
            if bertopic_model:
                # Print enhanced BERTopic topics analysis
                topic_info = bertopic_model.get_topic_info()
                n_topics = len(topic_info) - 1  # Exclude outlier topic
                
                print(f"\n🔍 BERTOPIC LEGAL TOPICS ANALYSIS:")
                print(f"📊 Generated {n_topics} distinct legal topics")
                
                # Show top topics with legal domain interpretation
                print(f"\n📋 TOP 15 LEGAL TOPICS:")
                displayed_topics = 0
                
                for _, row in topic_info.iterrows():
                    topic_id = row['Topic']
                    if topic_id != -1 and displayed_topics < 15:  # Skip outlier topic, show top 15
                        words = bertopic_model.get_topic(topic_id)
                        top_words = [word for word, _ in words[:8]]
                        word_weights = [f"{word}({weight:.3f})" for word, weight in words[:5]]
                            
                        # Attempt to categorize the topic
                        legal_category = "General"
                        topic_text = ' '.join(top_words)
                            
                        if any(term in topic_text for term in ['criminal', 'sentence', 'plea', 'defendant']):
                            legal_category = "🔒 Criminal Law"
                        elif any(term in topic_text for term in ['contract', 'agreement', 'breach']):
                            legal_category = "📄 Contract Law"
                        elif any(term in topic_text for term in ['property', 'real estate', 'lease']):
                            legal_category = "🏠 Property Law"
                        elif any(term in topic_text for term in ['family', 'custody', 'divorce', 'child']):
                            legal_category = "👨‍👩‍👧‍👦 Family Law"
                        elif any(term in topic_text for term in ['constitutional', 'amendment', 'rights']):
                            legal_category = "⚖️ Constitutional Law"
                        elif any(term in topic_text for term in ['employment', 'workplace', 'employee']):
                            legal_category = "💼 Employment Law"
                        elif any(term in topic_text for term in ['corporate', 'securities', 'shareholder']):
                            legal_category = "🏢 Corporate Law"
                        elif any(term in topic_text for term in ['tort', 'negligence', 'damages']):
                            legal_category = "⚡ Tort Law"
                            
                        topic_size = row['Count']
                        print(f"   Topic {topic_id:2d} ({topic_size:3d} cases) - {legal_category}")
                        print(f"      Keywords: {', '.join(top_words[:6])}")
                        print(f"      Weights:  {', '.join(word_weights)}")
                        print()
                        
                        displayed_topics += 1
                
                # Create BERTopic visualizations
                print(f"📊 Creating enhanced BERTopic visualizations...")
                topic_modeler.visualize_bertopic_results(bertopic_model, bertopic_documents)
                
                # Save results
                print(f"\n💾 Saving BERTopic results...")
                topic_modeler.save_results(bertopic_model)
                
                print(f"\n✅ BERT topic modeling complete!")
                print(f"🤖 Found {n_topics} BERTopic topics")
            else:
                print("❌ BERTopic model training failed")
        else:
            print("\n❌ BERTopic not available. Install with: pip install bertopic")
        
        # Final cleanup message
        print(f"\n🧹 Automatic resource cleanup completed")

if __name__ == "__main__":
    main()