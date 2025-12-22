import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from matplotlib.patches import Rectangle

# Set style for clean plots
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def load_lda_results():
    """Load LDA results from JSON file"""
    with open('/Users/liuyafei/Text_mining/data/processed/lda_topic_modeling.json', 'r') as f:
        return json.load(f)

def plot_topics_importance():
    """Plot 1: Topics Ranked by Importance Level"""
    # Load actual model results
    lda_results = load_lda_results()
    topics = lda_results['topics']
    
    # Calculate importance scores (average of top 3 word weights)
    topic_importance = []
    topic_names = []
    topic_descriptions = []
    
    # Remove unused variables to clean up the code
    # topic_descriptions is removed as we now use actual topic words
    # descriptions_map is removed as we use model data directly
    
    for topic in topics:
        topic_id = topic['topic_id']
        # Use average of top 3 weights as importance score
        importance = np.mean(topic['weights'][:10])
        topic_importance.append((importance, topic_id, topic['top_words_string']))
    
    # Sort by importance (descending)
    topic_importance.sort(reverse=True)
    
    importance_scores = [item[0] for item in topic_importance]
    topic_names = [f"Topic {item[1]}" for item in topic_importance]
    topic_words = [item[2] for item in topic_importance]
    
    # Create colors - different color for each topic
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#CDDC39', '#FF5722', '#795548', '#607D8B']
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create horizontal bars with labels for legend
    bars = ax.barh(range(len(topic_names)), importance_scores, color=colors, alpha=0.8)
    
    # Customize the plot
    ax.set_yticks(range(len(topic_names)))
    ax.set_yticklabels([f'#{i+1}: {name}' for i, name in enumerate(topic_names)], fontsize=11)
    ax.set_xlabel('Average Topic Importance Score', fontsize=14, fontweight='bold')
    ax.set_title('Topics Ranked by Importance Level', fontsize=18, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, importance_scores)):
        ax.text(score + 0.0005, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', ha='left', fontweight='bold')
    
    # Add legend using actual topic words
    legend_labels = [f'{name}: {words}' for name, words in zip(topic_names, topic_words)]
    legend = ax.legend(bars, legend_labels, loc='upper right', bbox_to_anchor=(1.0, 1.0), 
                      fontsize=9, title='Legal Topic Categories', title_fontsize=12,
                      frameon=True, fancybox=True, shadow=True)
    legend.get_title().set_fontweight('bold')
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(importance_scores) * 1.1)
    
    plt.tight_layout()
    plt.savefig('/Users/liuyafei/Text_mining/plot1_topics_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_legal_domain_distribution():
    """Plot 2: Legal Domain Distribution (Pie Chart)"""
    # Load actual model results
    lda_results = load_lda_results()
    topics = lda_results['topics']
    
    # Calculate topic weights based on average word weights
    domain_weights = []
    domains = []
    
    descriptions_map = {
        0: 'Employment/Worker Compensation',
        1: 'Civil Procedure', 
        2: 'Criminal Law',
        3: 'Constitutional/Statutory Law',
        4: 'Legal Process',
        5: 'Government/Administrative Law',
        6: 'Family Law',
        7: 'Property/Contract Law',
        8: 'Civil Litigation',
        9: 'Evidence/Trial Procedure'
    }
    
    for topic in topics:
        topic_id = topic['topic_id']
        # Use average of all weights as domain size
        weight = np.mean(topic['weights'])
        domain_weights.append(weight)
        domains.append(f"Topic {topic_id}")
    
    # Normalize to percentages
    total_weight = sum(domain_weights)
    sizes = [(w/total_weight) * 100 for w in domain_weights]
    
    # Professional color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#AED6F1', '#F8C471']
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create pie chart without labels on the pie (we'll use legend instead)
    wedges, texts, autotexts = ax.pie(sizes, colors=colors, autopct='%1.1f%%',
                                     startangle=90, textprops={'fontsize': 11})
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # Add legend outside the pie chart using actual topic words
    legend_labels = [f'Topic {i}: {topics[i]["top_words_string"]}' for i in range(len(topics))]
    ax.legend(wedges, legend_labels, title="Legal Topics", loc="upper right", 
              bbox_to_anchor=(1.3, 1.0), fontsize=10, title_fontsize=13,
              frameon=True, fancybox=True, shadow=True)
    
    ax.set_title('Legal Domain Distribution', fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/Users/liuyafei/Text_mining/plot2_legal_domain_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_topic_word_clouds():
    """Plot 3: Word clouds for top 3 topics (same ranking as Plot 1)"""
    # Load actual model results
    lda_results = load_lda_results()
    topics = lda_results['topics']
    
    # Use EXACT SAME importance calculation as Plot 1
    topic_importance = []
    for topic in topics:
        topic_id = topic['topic_id']
        # Use average of top 10 weights as importance score (SAME AS PLOT 1)
        importance = np.mean(topic['weights'][:10])
        topic_importance.append((importance, topic_id, topic))
    
    # Sort by importance (descending) - SAME AS PLOT 1
    topic_importance.sort(reverse=True)
    
    # Get top 3 topics from Plot 1 ranking
    top_3_topics = [item[2] for item in topic_importance[:3]]  # item[2] is the topic object
    
    print(f"Plot 1 ranking order: {[item[1] for item in topic_importance]}")
    print(f"Plot 3 using top 3: {[topic['topic_id'] for topic in top_3_topics]}")
    
    topics_data = []
    for i, topic in enumerate(top_3_topics):
        topic_id = topic['topic_id']
        topics_data.append({
            'title': f'#{i+1}: Topic {topic_id}\n({topic["top_words_string"]})',
            'words': topic['words'][:10],
            'weights': topic['weights'][:10]
        })
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    colors_set = [['#4CAF50', '#8BC34A', '#CDDC39', '#689F38', '#558B2F', '#33691E'], 
                  ['#2196F3', '#03A9F4', '#00BCD4', '#0288D1', '#0277BD', '#01579B'],
                  ['#FF9800', '#FF5722', '#795548', '#F57C00', '#E65100', '#BF360C']]
    
    for i, (ax, topic, colors) in enumerate(zip(axes, topics_data, colors_set)):
        # Create word size visualization
        y_pos = np.arange(len(topic['words']))
        bars = ax.barh(y_pos, topic['weights'], color=colors, alpha=0.8, 
                      label=[f'{word}: {weight:.3f}' for word, weight in zip(topic['words'], topic['weights'])])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(topic['words'], fontsize=11)
        ax.set_xlabel('Weight', fontsize=12, fontweight='bold')
        ax.set_title(topic['title'], fontsize=14, fontweight='bold')
        
        # Add value labels
        for j, (bar, weight) in enumerate(zip(bars, topic['weights'])):
            ax.text(weight + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{weight:.3f}', va='center', ha='left', fontsize=9, fontweight='bold')
        
        # Add legend showing word importance ranking using actual topic words
        legend_labels = [f'{word} ({weight:.3f})' for word, weight in zip(topics_data[i]['words'], topics_data[i]['weights'])]
        ax.legend(bars, legend_labels, loc='upper right', fontsize=8, 
                 title=f'Topic Words & Weights', title_fontsize=9, frameon=True, 
                 fancybox=True, shadow=True)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.suptitle('Top 3 Topics from Plot 1 Ranking - Key Legal Terms', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/liuyafei/Text_mining/plot3_topic_words.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return topics_data  # Return for debugging

def plot_case_clustering():
    """Plot 4: Case Clustering Analysis"""
    # Load actual model results
    lda_results = load_lda_results()
    cluster_analysis = lda_results['cluster_analysis']
    
    # Extract cluster data
    cluster_sizes = cluster_analysis['cluster_sizes']
    clusters_data = cluster_analysis['clusters']
    
    # Map cluster to descriptions based on actual dominant topics from model
    topics = lda_results['topics']
    topic_words_map = {topic['topic_id']: topic['top_words_string'] for topic in topics}
    
    cluster_names = []
    cluster_descriptions = []
    counts = []
    percentages = []
    
    total_cases = sum(cluster_sizes.values())
    
    for cluster_id in sorted(cluster_sizes.keys(), key=int):
        cluster_names.append(f'Cluster {cluster_id}')
        count = cluster_sizes[cluster_id]
        counts.append(count)
        percentage = (count / total_cases) * 100
        percentages.append(percentage)
        
        # Get dominant topic words for description
        dominant_topic = clusters_data[cluster_id]['dominant_topic']
        description = topic_words_map.get(dominant_topic, f'Topic {dominant_topic}')
        cluster_descriptions.append(description)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.bar(range(len(cluster_names)), counts, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=2, 
                  label=[f'{name}: {desc}' for name, desc in zip(cluster_names, cluster_descriptions)])
    
    # Add labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{count}\n({pct}%)', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    ax.set_xticks(range(len(cluster_names)))
    ax.set_xticklabels([f'{name}' for name in cluster_names], fontsize=12)
    ax.set_ylabel('Number of Cases', fontsize=14, fontweight='bold')
    silhouette_score = cluster_analysis['silhouette_score']
    ax.set_title(f'Case Clustering Analysis\n(Silhouette Score: {silhouette_score:.3f})', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_labels = [f'{name}: {desc}' for name, desc in zip(cluster_names, cluster_descriptions)]
    legend = ax.legend(bars, legend_labels, loc='upper right', fontsize=9, 
                      title='Cluster Categories', title_fontsize=11,
                      frameon=True, fancybox=True, shadow=True)
    legend.get_title().set_fontweight('bold')
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    plt.savefig('/Users/liuyafei/Text_mining/plot4_case_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate all plots
if __name__ == "__main__":
    print("Generating individual plots...")
    plot_topics_importance()
    plot_legal_domain_distribution()  
    plot_topic_word_clouds()
    plot_case_clustering()
    print("All plots generated successfully!")