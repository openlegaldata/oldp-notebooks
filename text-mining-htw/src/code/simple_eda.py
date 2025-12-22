#!/usr/bin/env python3
"""
Simple Exploratory Data Analysis for Legal Cases Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

def load_and_analyze_data():
    """Load the legal cases dataset and perform basic EDA"""
    
    print("=" * 60)
    print("LEGAL CASES DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Load the dataset
    file_path = "/data/processed/cleaned/parse_legal_cases.csv"
    print(f"Loading data from: {file_path}")
    
    try:
        # Load first chunk to understand structure
        df_sample = pd.read_csv(file_path, nrows=10000, low_memory=False)
        print(f"✅ Sample loaded successfully: {df_sample.shape}")
        
        # Load full dataset
        print("Loading full dataset...")
        df = pd.read_csv(file_path, low_memory=False)
        print(f"✅ Full dataset loaded: {df.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Basic dataset information
    print(f"\n1. DATASET OVERVIEW")
    print(f"-" * 30)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Data types
    print(f"\nData types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Missing data analysis
    print(f"\n2. MISSING DATA ANALYSIS")
    print(f"-" * 30)
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Missing_Percentage', ascending=False)
    
    print("Top 10 columns with missing data:")
    print(missing_data.head(10).to_string(index=False))
    
    # Date analysis
    if 'date_filed' in df.columns:
        print(f"\n3. TEMPORAL ANALYSIS")
        print(f"-" * 30)
        df['date_filed'] = pd.to_datetime(df['date_filed'], errors='coerce')
        df['year_filed'] = df['date_filed'].dt.year
        df['month_filed'] = df['date_filed'].dt.month
        
        print(f"Date range: {df['date_filed'].min()} to {df['date_filed'].max()}")
        print(f"Approximate dates: {df['date_filed_is_approximate'].sum():,} cases")
        
        # Cases by year
        yearly_counts = df['year_filed'].value_counts().sort_index()
        print(f"\nCases by year:")
        for year, count in yearly_counts.items():
            if pd.notna(year):
                print(f"  {int(year)}: {count:,} cases")
    
    # Court system analysis
    if 'court_type' in df.columns:
        print(f"\n4. COURT SYSTEM ANALYSIS")
        print(f"-" * 30)
        court_type_counts = df['court_type'].value_counts()
        print("Court types:")
        for court_type, count in court_type_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {court_type}: {count:,} cases ({percentage:.1f}%)")
        
        if 'court_jurisdiction' in df.columns:
            print(f"\nTop 10 jurisdictions:")
            jurisdiction_counts = df['court_jurisdiction'].value_counts().head(10)
            for jurisdiction, count in jurisdiction_counts.items():
                print(f"  {jurisdiction}: {count:,} cases")
    
    # Citation analysis
    if 'citation_count' in df.columns:
        print(f"\n5. CITATION ANALYSIS")
        print(f"-" * 30)
        citation_stats = df['citation_count'].describe()
        print("Citation statistics:")
        print(f"  Mean: {citation_stats['mean']:.2f}")
        print(f"  Median: {citation_stats['50%']:.0f}")
        print(f"  Max: {citation_stats['max']:.0f}")
        print(f"  Cases with 0 citations: {(df['citation_count'] == 0).sum():,} ({(df['citation_count'] == 0).sum()/len(df)*100:.1f}%)")
        print(f"  Cases with 10+ citations: {(df['citation_count'] >= 10).sum():,}")
        print(f"  Cases with 50+ citations: {(df['citation_count'] >= 50).sum():,}")
    
    # Case name analysis
    if 'case_name' in df.columns:
        print(f"\n6. CASE NAME ANALYSIS")
        print(f"-" * 30)
        df['case_name_length'] = df['case_name'].str.len()
        print(f"Case name length statistics:")
        print(f"  Average length: {df['case_name_length'].mean():.0f} characters")
        print(f"  Max length: {df['case_name_length'].max():.0f} characters")
        print(f"  Min length: {df['case_name_length'].min():.0f} characters")
        
        # Common patterns
        vs_pattern = df['case_name'].str.contains(' v\. | vs\. ', case=False, na=False).sum()
        print(f"  Cases with 'v.' pattern: {vs_pattern:,} ({vs_pattern/len(df)*100:.1f}%)")
        
        # Legal abbreviations
        legal_abbrevs = {
            'Inc.': df['case_name'].str.contains('Inc\.', case=False, na=False).sum(),
            'Corp.': df['case_name'].str.contains('Corp\.', case=False, na=False).sum(),
            'LLC': df['case_name'].str.contains('LLC', case=False, na=False).sum(),
            'Co.': df['case_name'].str.contains(' Co\.', case=False, na=False).sum()
        }
        print(f"  Legal abbreviations:")
        for abbrev, count in legal_abbrevs.items():
            print(f"    {abbrev}: {count:,} cases")
    
    # Opinion text analysis
    if 'opinion_text' in df.columns:
        print(f"\n7. OPINION TEXT ANALYSIS")
        print(f"-" * 30)
        opinion_available = df['opinion_text'].notna().sum()
        print(f"Cases with opinion text: {opinion_available:,} ({opinion_available/len(df)*100:.1f}%)")
        
        if opinion_available > 0:
            df['opinion_length'] = df['opinion_text'].str.len()
            opinion_stats = df['opinion_length'].describe()
            print(f"Opinion text length statistics:")
            print(f"  Average length: {opinion_stats['mean']:.0f} characters")
            print(f"  Median length: {opinion_stats['50%']:.0f} characters")
            print(f"  Max length: {opinion_stats['max']:.0f} characters")
    
    # Sample data
    print(f"\n8. SAMPLE DATA")
    print(f"-" * 30)
    print("First 3 case names:")
    for i, case_name in enumerate(df['case_name'].head(3), 1):
        print(f"  {i}. {case_name}")
    
    if 'court_short_name' in df.columns:
        print(f"\nTop 5 most active courts:")
        top_courts = df['court_short_name'].value_counts().head(5)
        for court, count in top_courts.items():
            print(f"  {court}: {count:,} cases")
    
    # Data quality assessment
    print(f"\n9. DATA QUALITY SUMMARY")
    print(f"-" * 30)
    
    # Check for duplicates
    duplicate_ids = df['id'].duplicated().sum() if 'id' in df.columns else 0
    print(f"Duplicate IDs: {duplicate_ids}")
    
    # Completeness score for key fields
    key_fields = ['case_name', 'court_type', 'date_filed']
    completeness_scores = {}
    for field in key_fields:
        if field in df.columns:
            completeness = (df[field].notna().sum() / len(df)) * 100
            completeness_scores[field] = completeness
            print(f"{field} completeness: {completeness:.1f}%")
    
    print(f"\n✅ Analysis complete! Dataset has {len(df):,} legal cases with rich metadata.")
    
    return df

if __name__ == "__main__":
    df = load_and_analyze_data()