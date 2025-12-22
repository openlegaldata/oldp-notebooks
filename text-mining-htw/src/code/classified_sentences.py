import pandas as pd
import spacy
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy model loaded successfully")
except OSError:
    print("❌ Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

class LegalSentenceClassifier:
    """Classifier for legal sentence types in opinion text"""
    
    def __init__(self):
        # Define patterns for different sentence types
        self.patterns = {
            'procedural': [
                r'\b(moved?|filed?|appealed?|petitioned?|requested?)\b',
                r'\b(motion|petition|appeal|request|filing)\b',
                r'\b(court ordered?|granted?|denied?)\b',
                r'\b(hearing|trial|proceeding|action)\b',
                r'\b(served|served upon|notice)\b'
            ],
            'holding': [
                r'\b(held?|holding|find|found|conclude|concluded?)\b',
                r'\b(determine|determined?|decide|decided?)\b',
                r'\b(rule|ruled?|ruling)\b',
                r'we (hold|find|conclude|determine|decide|rule)',
                r'\b(affirm|reverse|remand|dismiss|vacate)\b'
            ],
            'reasoning': [
                r'\b(because|since|therefore|thus|accordingly)\b',
                r'\b(rationale|basis|reason|justification)\b',
                r'\b(analysis|examine|consider|evaluation)\b',
                r'\b(however|moreover|furthermore|nevertheless)\b',
                r'the (court|law|statute|precedent) (requires?|states?|provides?)'
            ],
            'citation': [
                r'\d+\s+[A-Z][a-z]*\.?\s+\d+',  # Case citations
                r'\d+\s+U\.S\.C\.\s+§\s+\d+',   # USC references
                r'§\s*\d+[\d\.]*',               # Section references
                r'\b(cite|citing|cited|see|accord|cf\.)\b.*\d+',
                r'\b(supra|infra|id\.)\b'
            ],
            'factual': [
                r'\b(plaintiff|defendant|party|parties)\b.*\b(alleges?|claims?|argues?|contends?)\b',
                r'\b(evidence|testimony|witness|document)\b',
                r'\b(occurred?|happened?|took place|incident)\b',
                r'on\s+(\w+\s+)?\d{1,2},?\s+\d{4}',  # Dates
                r'\$\d+|\d+\s+dollars?'  # Money amounts
            ]
        }
    
    def classify_sentence(self, sentence: str) -> str:
        """Classify a single sentence into legal categories"""
        sentence_lower = sentence.lower()
        scores = {}
        
        for category, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, sentence_lower, re.IGNORECASE))
                score += matches
            scores[category] = score
        
        # Return category with highest score, default to 'other'
        max_score = max(scores.values())
        if max_score == 0:
            return 'other'
        
        return max(scores, key=scores.get)
    
    def segment_and_classify(self, text: str) -> List[Dict[str, str]]:
        """Segment text into sentences and classify each"""
        if not text or not isinstance(text, str):
            return []
        
        # Use spaCy for sentence segmentation
        if nlp is not None:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to regex-based segmentation
            sentences = re.split(r'(?<=\.)\s+', text)
        
        results = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 10:  # Filter very short sentences
                classification = self.classify_sentence(sentence)
                results.append({
                    'sentence_id': i,
                    'sentence': sentence.strip(),
                    'classification': classification,
                    'length': len(sentence)
                })
        
        return results

def load_data(file_path: str) -> pd.DataFrame:
    """Load the legal cases dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded: {df.shape[0]} cases, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return pd.DataFrame()

def process_cases(df: pd.DataFrame, classifier: LegalSentenceClassifier, num_cases: int = 100) -> List[Dict]:
    """Process multiple cases and return aggregated results"""
    all_results = []
    processed_count = 0
    
    for idx, row in df.head(num_cases).iterrows():
        if pd.isna(row['opinion_text']):
            continue
            
        results = classifier.segment_and_classify(str(row['opinion_text']))
        
        # Add case metadata to each sentence
        for result in results:
            result['case_id'] = row['id']
            result['court_type'] = row.get('court_type', 'Unknown')
            result['case_name_short'] = row.get('case_name_short', 'Unknown')
            result['year_filed'] = row.get('year_filed', None)
        
        all_results.extend(results)
        processed_count += 1
        
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} cases...")
    
    print(f"✓ Completed processing {processed_count} cases")
    return all_results

def analyze_results(sentences_df: pd.DataFrame):
    """Generate comprehensive analysis of classification results"""
    print("\n" + "="*50)
    print("SENTENCE CLASSIFICATION ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print(f"Total sentences analyzed: {len(sentences_df):,}")
    print(f"Unique cases: {sentences_df['case_id'].nunique()}")
    print(f"Average sentences per case: {len(sentences_df) / sentences_df['case_id'].nunique():.1f}")
    print(f"Average sentence length: {sentences_df['length'].mean():.1f} characters")
    
    # Classification distribution
    print("\nClassification Distribution:")
    class_counts = sentences_df['classification'].value_counts()
    for cls, count in class_counts.items():
        percentage = (count / len(sentences_df)) * 100
        print(f"  {cls}: {count:,} ({percentage:.1f}%)")
    
    # Court type analysis
    if 'court_type' in sentences_df.columns:
        print("\nSentences by Court Type:")
        court_counts = sentences_df['court_type'].value_counts()
        for court, count in court_counts.head().items():
            print(f"  {court}: {count:,}")
    
    return class_counts

def visualize_results(sentences_df: pd.DataFrame, save_plots: bool = True):
    """Create visualizations of the classification results"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall classification distribution
    class_counts = sentences_df['classification'].value_counts()
    axes[0, 0].bar(class_counts.index, class_counts.values, color='steelblue')
    axes[0, 0].set_title('Overall Sentence Classification Distribution', fontsize=14)
    axes[0, 0].set_xlabel('Classification Type')
    axes[0, 0].set_ylabel('Number of Sentences')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(class_counts.values):
        axes[0, 0].text(i, v + max(class_counts.values)*0.01, str(v), 
                       ha='center', va='bottom')
    
    # 2. Sentence length by classification
    sns.boxplot(data=sentences_df, x='classification', y='length', ax=axes[0, 1])
    axes[0, 1].set_title('Sentence Length Distribution by Classification', fontsize=14)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Classification by court type (if available)
    if 'court_type' in sentences_df.columns and sentences_df['court_type'].nunique() > 1:
        court_class = pd.crosstab(sentences_df['court_type'], sentences_df['classification'])
        court_class.plot(kind='bar', stacked=True, ax=axes[1, 0])
        axes[1, 0].set_title('Classification Distribution by Court Type', fontsize=14)
        axes[1, 0].set_xlabel('Court Type')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[1, 0].text(0.5, 0.5, 'Court type data not available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Classification by Court Type (N/A)', fontsize=14)
    
    # 4. Average sentence length by classification
    avg_length = sentences_df.groupby('classification')['length'].mean().sort_values(ascending=False)
    axes[1, 1].bar(avg_length.index, avg_length.values, color='lightcoral')
    axes[1, 1].set_title('Average Sentence Length by Classification', fontsize=14)
    axes[1, 1].set_xlabel('Classification Type')
    axes[1, 1].set_ylabel('Average Length (characters)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(avg_length.values):
        axes[1, 1].text(i, v + max(avg_length.values)*0.01, f'{v:.0f}', 
                       ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('sentence_classification_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Visualization saved as 'sentence_classification_analysis.png'")
    
    plt.show()

def show_examples(sentences_df: pd.DataFrame, num_examples: int = 2):
    """Display examples of each sentence classification type"""
    print("\n" + "="*60)
    print("EXAMPLES OF CLASSIFIED SENTENCES")
    print("="*60)
    
    for classification in sorted(sentences_df['classification'].unique()):
        examples = sentences_df[sentences_df['classification'] == classification]['sentence'].head(num_examples)
        print(f"\n{classification.upper()} SENTENCES:")
        print("-" * 30)
        
        for i, example in enumerate(examples, 1):
            # Truncate long sentences for display
            display_text = example[:300] + "..." if len(example) > 300 else example
            print(f"{i}. {display_text}")
            print()

def save_results(sentences_df: pd.DataFrame, output_dir: str = "data/processed/"):
    """Save classification results to CSV files"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed sentence results
    sentences_file = os.path.join(output_dir, "classified_sentences_detailed.csv")
    sentences_df.to_csv(sentences_file, index=False)
    print(f"💾 Detailed results saved: {sentences_file}")
    
    # Create case summary with simple approach
    case_summary_data = []
    
    for case_id in sentences_df['case_id'].unique():
        case_data = sentences_df[sentences_df['case_id'] == case_id]
        
        # Basic case info
        summary_row = {
            'case_id': case_id,
            'case_name_short': case_data['case_name_short'].iloc[0],
            'court_type': case_data['court_type'].iloc[0],
            'total_sentences': len(case_data),
            'avg_sentence_length': case_data['length'].mean()
        }
        
        # Add classification counts
        class_counts = case_data['classification'].value_counts()
        for classification in ['procedural', 'holding', 'reasoning', 'citation', 'factual', 'other']:
            summary_row[classification] = class_counts.get(classification, 0)
        
        case_summary_data.append(summary_row)
    
    case_summary = pd.DataFrame(case_summary_data)
    
    summary_file = os.path.join(output_dir, "case_classification_summary.csv")
    case_summary.to_csv(summary_file, index=False)
    print(f"💾 Case summary saved: {summary_file}")
    
    return sentences_file, summary_file

def main():
    """Main execution function"""
    print("🚀 Legal Opinion Sentence Classification")
    print("="*50)
    
    # Initialize classifier
    classifier = LegalSentenceClassifier()
    
    # Load data
    data_file = "Text-mining-HTW/data/processed/cleaned/parse_legal_cases.csv"
    df = load_data(data_file)
    
    if df.empty:
        print("❌ No data loaded. Please check the file path.")
        return
    
    # Process cases
    print(f"\n📊 Processing cases for sentence classification...")
    sentence_results = process_cases(df, classifier, num_cases=100)
    
    if not sentence_results:
        print("❌ No sentences were processed.")
        return
    
    # Create DataFrame
    sentences_df = pd.DataFrame(sentence_results)
    
    # Analyze results
    class_counts = analyze_results(sentences_df)
    
    # Show examples
    show_examples(sentences_df, num_examples=2)
    
    # Create visualizations
    print(f"\n📈 Generating visualizations...")
    visualize_results(sentences_df)
    
    # Save results
    print(f"\n💾 Saving results...")
    save_results(sentences_df)
    
    print(f"\n✅ Analysis complete!")
    print(f"   - Processed {len(sentences_df):,} sentences from {sentences_df['case_id'].nunique()} cases")
    print(f"   - Found {len(class_counts)} different sentence types")
    print(f"   - Most common type: {class_counts.index[0]} ({class_counts.iloc[0]:,} sentences)")

if __name__ == "__main__":
    main()