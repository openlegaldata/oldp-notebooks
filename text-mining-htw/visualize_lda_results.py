import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_lda_results(file_path: str) -> Dict[str, Any]:
    """Load LDA results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_topic_importance(topics: List[Dict]) -> List[Dict]:
    """Calculate topic importance based on average weights and add rankings"""
    # Calculate average weight for each topic (importance metric)
    for topic in topics:
        topic['avg_weight'] = np.mean(topic['weights'][:5])  # Top 5 words average
        topic['total_weight'] = sum(topic['weights'])
        
    # Sort by importance (average weight)
    topics_ranked = sorted(topics, key=lambda x: x['avg_weight'], reverse=True)
    
    # Add rankings
    for i, topic in enumerate(topics_ranked):
        topic['importance_rank'] = i + 1
        
    return topics_ranked

def categorize_topics(topics: List[Dict]) -> Dict[str, List[int]]:
    """Categorize topics into legal domains"""
    legal_domains = {
        '⚖️ Criminal Law': ['sentence', 'counsel', 'plea', 'conviction', 'criminal', 'officer', 'police', 'evidence', 'jury'],
        '🏢 Employment Law': ['employee', 'work', 'injury', 'claimant', 'employer', 'employment', 'compensation', 'board'],
        '👨‍👩‍👧‍👦 Family Law': ['child', 'mother', 'father', 'family', 'parent', 'custody', 'abuse', 'care'],
        '🏠 Property/Contract Law': ['property', 'agreement', 'contract', 'fee', 'interest', 'tax', 'payment', 'sale'],
        '📋 Civil Litigation': ['claim', 'action', 'policy', 'damage', 'liability', 'summary', 'judgment'],
        '🏛️ Administrative Law': ['government', 'agency', 'information', 'document', 'public', 'service', 'review'],
        '⚖️ Legal Procedure': ['rule', 'attorney', 'notice', 'file', 'counsel', 'jurisdiction', 'relief'],
        '📜 Constitutional/Statutory': ['statute', 'act', 'right', 'statutory', 'public', 'amendment', 'provision'],
        '🔍 Evidence/Trial': ['evidence', 'jury', 'officer', 'testimony', 'testified', 'witness', 'reasonable', 'statement'],
        '🏛️ Civil Procedure': ['summary', 'judgment', 'department', 'arbitration', 'division', 'entered', 'decided']
    }
    
    topic_categories = {}
    
    for topic in topics:
        topic_words = set([word.lower() for word in topic['words'][:5]])
        category = 'General'
        max_matches = 0
        
        for domain, keywords in legal_domains.items():
            matches = len(topic_words.intersection(set(keywords)))
            if matches > max_matches:
                max_matches = matches
                category = domain
                
        if category not in topic_categories:
            topic_categories[category] = []
        topic_categories[category].append(topic['topic_id'])
        
        # Add category to topic
        topic['category'] = category
    
    return topic_categories

def create_comprehensive_visualization(results: Dict[str, Any]):
    """Create comprehensive visualization of LDA results ranked by importance"""
    
    topics = results['topics']
    cluster_analysis = results['cluster_analysis']
    
    # Calculate importance and rankings
    topics_ranked = calculate_topic_importance(topics)
    topic_categories = categorize_topics(topics_ranked)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 20))
    
    # Title
    fig.suptitle('📊 LDA Legal Topic Analysis - Comprehensive Results Dashboard', 
                fontsize=28, fontweight='bold', y=0.98)
    
    # Color schemes
    colors_main = plt.cm.Set3(np.linspace(0, 1, 10))
    colors_accent = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#AED6F1', '#F8BBD9']
    
    # 1. Topic Importance Ranking (Main visualization)
    ax1 = plt.subplot(2, 3, (1, 2))
    
    # Extract data for ranking plot
    topic_ids = [t['topic_id'] for t in topics_ranked]
    avg_weights = [t['avg_weight'] for t in topics_ranked]
    categories = [t['category'] for t in topics_ranked]
    top_words = [', '.join(t['words'][:3]) for t in topics_ranked]
    
    # Create horizontal bar chart
    bars = ax1.barh(range(len(topics_ranked)), avg_weights, 
                   color=colors_main, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Customize the bars with gradients
    for i, (bar, category) in enumerate(zip(bars, categories)):
        # Add value labels
        ax1.text(bar.get_width() + max(avg_weights) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{avg_weights[i]:.4f}', va='center', fontweight='bold', fontsize=10)
        
        # Add category emoji
        category_emoji = category.split(' ')[0] if ' ' in category else '📋'
        ax1.text(-max(avg_weights) * 0.08, bar.get_y() + bar.get_height()/2,
                category_emoji, va='center', fontsize=16)
    
    # Labels and formatting
    ax1.set_yticks(range(len(topics_ranked)))
    ax1.set_yticklabels([f'#{i+1}: Topic {topic_ids[i]}\n({top_words[i]})' 
                        for i in range(len(topics_ranked))], fontsize=11)
    ax1.set_xlabel('Average Topic Importance Score', fontsize=14, fontweight='bold')
    ax1.set_title('🏆 Topics Ranked by Importance Level', fontsize=18, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Topic Categories Distribution
    ax2 = plt.subplot(2, 3, 3)
    
    category_counts = {cat: len(topics) for cat, topics in topic_categories.items()}
    categories_clean = [cat.split(' ', 1)[1] if ' ' in cat else cat for cat in category_counts.keys()]
    
    wedges, texts, autotexts = ax2.pie(category_counts.values(), 
                                      labels=categories_clean,
                                      autopct='%1.1f%%',
                                      colors=colors_accent[:len(category_counts)],
                                      explode=[0.05] * len(category_counts))
    
    ax2.set_title('🎯 Legal Domain Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # 3. Topic Word Clouds (Top 3 most important topics)
    for i in range(3):
        ax = plt.subplot(2, 6, 7 + i)
        topic = topics_ranked[i]
        
        # Create word importance visualization
        words = topic['words'][:6]
        weights = topic['weights'][:6]
        
        # Horizontal bar chart for words
        y_pos = np.arange(len(words))
        bars_words = ax.barh(y_pos, weights, color=colors_main[i], alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10)
        ax.set_xlabel('Weight', fontsize=10)
        ax.set_title(f'#{i+1}: Topic {topic["topic_id"]}\n{topic["category"].split(" ", 1)[1] if " " in topic["category"] else topic["category"]}', 
                    fontsize=12, fontweight='bold')
        
        # Add value labels
        for j, bar in enumerate(bars_words):
            ax.text(bar.get_width() + max(weights) * 0.02, bar.get_y() + bar.get_height()/2,
                   f'{weights[j]:.3f}', va='center', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 4. Cluster Analysis
    ax4 = plt.subplot(2, 6, (10, 12))
    
    # Extract cluster data
    cluster_sizes = [cluster_analysis['clusters'][str(i)]['size'] for i in range(cluster_analysis['n_clusters'])]
    cluster_topics = [cluster_analysis['clusters'][str(i)]['dominant_topic'] for i in range(cluster_analysis['n_clusters'])]
    
    # Create cluster visualization
    cluster_labels = [f'Cluster {i}\n(Topic {cluster_topics[i]})' for i in range(len(cluster_sizes))]
    
    bars_cluster = ax4.bar(range(len(cluster_sizes)), cluster_sizes, 
                          color=colors_accent[:len(cluster_sizes)], alpha=0.8)
    
    # Add percentage labels
    total_cases = sum(cluster_sizes)
    for i, (bar, size) in enumerate(zip(bars_cluster, cluster_sizes)):
        height = bar.get_height()
        percentage = (size / total_cases) * 100
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(cluster_sizes) * 0.01,
                f'{size:,}\n({percentage:.1f}%)', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    ax4.set_xticks(range(len(cluster_sizes)))
    ax4.set_xticklabels(cluster_labels, rotation=45, ha='right')
    ax4.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax4.set_title(f'📊 Case Clustering Analysis\n(Silhouette Score: {cluster_analysis["silhouette_score"]:.3f})', 
                 fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('lda_comprehensive_analysis.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("📊 Comprehensive LDA analysis saved as 'lda_comprehensive_analysis.png'")
    
    return topics_ranked, topic_categories

def create_detailed_topic_table(topics_ranked: List[Dict], topic_categories: Dict[str, List[int]]):
    """Create detailed topic analysis table"""
    
    print("\n" + "="*100)
    print("🏆 DETAILED TOPIC ANALYSIS - RANKED BY IMPORTANCE")
    print("="*100)
    
    for i, topic in enumerate(topics_ranked):
        print(f"\n#{i+1:2d} │ TOPIC {topic['topic_id']:2d} │ {topic['category']}")
        print("─" * 80)
        print(f"    📊 Importance Score: {topic['avg_weight']:.4f}")
        print(f"    🎯 Total Weight: {topic['total_weight']:.4f}")
        
        # Top words with weights
        print("    🔍 Keywords & Weights:")
        for j, (word, weight) in enumerate(zip(topic['words'][:8], topic['weights'][:8])):
            print(f"        {j+1:2d}. {word:<20} ({weight:.4f})")
        
        print(f"    📝 Summary: {topic['top_words_string']}")
        print()
    
    # Category summary
    print("\n" + "="*60)
    print("🎯 TOPIC CATEGORIES SUMMARY")
    print("="*60)
    
    for category, topic_ids in topic_categories.items():
        print(f"{category}: {len(topic_ids)} topics (IDs: {topic_ids})")

def create_importance_heatmap(topics_ranked: List[Dict]):
    """Create heatmap showing topic-word importance relationships"""
    
    # Prepare data for heatmap
    n_topics = len(topics_ranked)
    n_words = 8
    
    # Create matrix
    word_weights_matrix = []
    topic_labels = []
    all_words = []
    
    for topic in topics_ranked:
        topic_labels.append(f"#{topic['importance_rank']} T{topic['topic_id']}")
        words = topic['words'][:n_words]
        weights = topic['weights'][:n_words]
        
        # Normalize weights for better visualization
        normalized_weights = [w / max(topic['weights']) for w in weights]
        word_weights_matrix.append(normalized_weights)
        
        if not all_words:  # First iteration
            all_words = words
    
    # Create heatmap
    plt.figure(figsize=(16, 12))
    
    # Convert to numpy array for heatmap
    heatmap_data = np.array(word_weights_matrix)
    
    # Create heatmap with better color scheme
    sns.heatmap(heatmap_data, 
                xticklabels=all_words,
                yticklabels=topic_labels,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Word Importance'},
                linewidths=0.5)
    
    plt.title('🔥 Topic-Word Importance Heatmap\n(Ranked by Topic Importance)', 
             fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Top Keywords', fontsize=14, fontweight='bold')
    plt.ylabel('Topics (Ranked by Importance)', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig('lda_importance_heatmap.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("🔥 Topic importance heatmap saved as 'lda_importance_heatmap.png'")

def main():
    """Main function to load and visualize LDA results"""
    
    print("🚀 Loading and Visualizing LDA Results")
    print("="*50)
    
    # Load results
    results_file = "/Users/liuyafei/Text_mining/data/processed/lda_topic_modeling.json"
    results = load_lda_results(results_file)
    
    print(f"✓ Loaded {len(results['topics'])} topics")
    print(f"✓ Loaded cluster analysis with {results['cluster_analysis']['n_clusters']} clusters")
    print(f"✓ Silhouette Score: {results['cluster_analysis']['silhouette_score']:.3f}")
    
    # Create comprehensive visualizations
    print("\n📊 Creating comprehensive visualizations...")
    topics_ranked, topic_categories = create_comprehensive_visualization(results)
    
    # Create detailed topic table
    create_detailed_topic_table(topics_ranked, topic_categories)
    
    # Create importance heatmap
    print("\n🔥 Creating importance heatmap...")
    create_importance_heatmap(topics_ranked)
    
    plt.show()
    
    print("\n✅ All visualizations created successfully!")
    print("📁 Files generated:")
    print("   • lda_comprehensive_analysis.png")
    print("   • lda_importance_heatmap.png")

if __name__ == "__main__":
    main()