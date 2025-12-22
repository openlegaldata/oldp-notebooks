"""
Enhanced BERT Topic Modeling Visualization
Loads saved BERTopic model and creates comprehensive visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from bertopic import BERTopic
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BERTTopicVisualizer:
    """Enhanced visualization for BERT topic modeling results"""
    
    def __init__(self, model_path: str = "data/processed/bertopic_model"):
        """Load the saved BERTopic model"""
        try:
            self.topic_model = BERTopic.load(model_path)
            print(f"✅ BERTopic model loaded from {model_path}")
            
            # Get topic information
            self.topic_info = self.topic_model.get_topic_info()
            self.n_topics = len(self.topic_info) - 1  # Exclude outlier topic (-1)
            print(f"📊 Model contains {self.n_topics} topics")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.topic_model = None
            
        # Legal domain mappings for better categorization
        self.legal_domains = {
            'Criminal Law': ['criminal', 'sentence', 'plea', 'defendant', 'prosecution', 'guilty', 'conviction', 'murder', 'assault'],
            'Civil Law': ['civil', 'damages', 'liability', 'negligence', 'tort', 'plaintiff', 'compensation'],
            'Contract Law': ['contract', 'agreement', 'breach', 'performance', 'consideration'],
            'Property Law': ['property', 'real estate', 'lease', 'landlord', 'tenant', 'ownership', 'zoning'],
            'Family Law': ['custody', 'divorce', 'marriage', 'child', 'parent', 'support', 'termination'],
            'Constitutional Law': ['constitutional', 'amendment', 'rights', 'freedom', 'due process', 'equal protection'],
            'Employment Law': ['employment', 'employee', 'employer', 'workplace', 'discrimination', 'title vii'],
            'Corporate Law': ['corporation', 'shareholder', 'director', 'securities', 'merger', 'business'],
            'Administrative Law': ['agency', 'regulation', 'administrative', 'government', 'federal', 'state'],
            'Tort Law': ['negligence', 'liability', 'damages', 'injury', 'malpractice', 'wrongful']
        }
        
        # Color palette for legal domains
        self.domain_colors = {
            'Criminal Law': '#FF6B6B',
            'Civil Law': '#4ECDC4',
            'Contract Law': '#45B7D1',
            'Property Law': '#96CEB4',
            'Family Law': '#FFEAA7',
            'Constitutional Law': '#DDA0DD',
            'Employment Law': '#98D8C8',
            'Corporate Law': '#F7DC6F',
            'Administrative Law': '#BB8FCE',
            'Tort Law': '#F8C471',
            'General': '#BDC3C7'
        }
    
    def categorize_topic(self, topic_words: list) -> str:
        """Categorize a topic based on its keywords"""
        topic_text = ' '.join(topic_words).lower()
        
        domain_scores = {}
        for domain, keywords in self.legal_domains.items():
            score = sum(1 for keyword in keywords if keyword in topic_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return 'General'
    
    def create_meaningful_topic_labels(self):
        """Create meaningful labels for each topic"""
        topic_labels = {}
        topic_descriptions = {}
        
        for _, row in self.topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Skip outlier topic
                # Get top words and their weights
                words_weights = self.topic_model.get_topic(topic_id)[:8]
                top_words = [word for word, _ in words_weights]
                category = self.categorize_topic(top_words)
                
                # Create meaningful label based on top words
                if 'criminal' in top_words or 'conviction' in top_words or 'sentence' in top_words:
                    if 'murder' in top_words or 'homicide' in top_words:
                        label = "🔒 Criminal: Homicide Cases"
                    elif 'assault' in top_words:
                        label = "🔒 Criminal: Assault Cases"
                    elif 'conviction' in top_words:
                        label = "🔒 Criminal: Post-Conviction"
                    else:
                        label = "🔒 Criminal Law"
                elif 'custody' in top_words or 'child' in top_words or 'termination parental' in ' '.join(top_words):
                    label = "👨‍👩‍👧‍👦 Family: Child Custody"
                elif 'zoning' in top_words or 'land use' in top_words:
                    label = "🏠 Property: Zoning Law"
                elif 'foreclosure' in top_words:
                    label = "🏠 Property: Foreclosure"
                elif 'employment' in top_words or 'title vii' in top_words:
                    label = "💼 Employment Law"
                elif 'negligence' in top_words or 'damages' in top_words:
                    label = "⚡ Tort: Negligence"
                elif 'workers compensation' in ' '.join(top_words):
                    label = "💼 Workers Compensation"
                elif 'disciplinary' in top_words and 'attorney' in top_words:
                    label = "⚖️ Attorney Discipline"
                elif 'appeal' in top_words or 'precedent' in top_words:
                    label = "📋 Appellate Procedure"
                else:
                    # Create custom label from top 2-3 most descriptive words
                    descriptive_words = [w for w in top_words[:3] if len(w) > 3][:2]
                    label = f"{self.get_domain_emoji(category)} {category}: {' & '.join(descriptive_words).title()}"
                
                topic_labels[topic_id] = label
                
                # Create detailed description
                description = f"Keywords: {', '.join(top_words[:5])}"
                topic_descriptions[topic_id] = description
        
        return topic_labels, topic_descriptions
    
    def get_domain_emoji(self, domain: str) -> str:
        """Get emoji for legal domain"""
        emoji_map = {
            'Criminal Law': '🔒',
            'Civil Law': '⚖️', 
            'Contract Law': '📄',
            'Property Law': '🏠',
            'Family Law': '👨‍👩‍👧‍👦',
            'Constitutional Law': '📜',
            'Employment Law': '💼',
            'Corporate Law': '🏢',
            'Administrative Law': '🏛️',
            'Tort Law': '⚡',
            'General': '📊'
        }
        return emoji_map.get(domain, '📊')
    
    def create_topic_size_distribution(self, show_plot: bool = True):
        """Create topic size distribution chart with meaningful labels"""
        if not self.topic_model:
            return
            
        # Get meaningful topic labels
        topic_labels, topic_descriptions = self.create_meaningful_topic_labels()
        
        # Prepare data
        topic_data = []
        for _, row in self.topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Skip outlier topic
                topic_size = row['Count']
                topic_words = [word for word, _ in self.topic_model.get_topic(topic_id)[:5]]
                category = self.categorize_topic(topic_words)
                
                topic_data.append({
                    'Topic': topic_labels.get(topic_id, f"Topic {topic_id}"),
                    'Topic_ID': topic_id,
                    'Size': topic_size,
                    'Category': category,
                    'Description': topic_descriptions.get(topic_id, ""),
                    'Percentage': (topic_size / self.topic_info['Count'].sum()) * 100
                })
        
        # Sort by size for better visualization
        topic_data = sorted(topic_data, key=lambda x: x['Size'], reverse=True)
        df = pd.DataFrame(topic_data)
        
        # Create interactive bar chart
        fig = px.bar(
            df, 
            x='Size', 
            y='Topic',
            color='Category',
            color_discrete_map=self.domain_colors,
            hover_data=['Description', 'Percentage'],
            title='📊 BERT Legal Topics - Distribution by Case Count',
            labels={'Size': 'Number of Cases', 'Topic': 'Legal Topics'},
            orientation='h'  # Horizontal bars for better label readability
        )
        
        fig.update_layout(
            title_font_size=20,
            title_x=0.5,
            height=max(600, len(df) * 40),  # Dynamic height based on number of topics
            width=1400,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            yaxis=dict(
                categoryorder='total ascending'  # Order by size
            ),
            margin=dict(l=300)  # More space for labels
        )
        
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>" +
                         "Cases: %{x}<br>" +
                         "Category: %{customdata[1]}<br>" +
                         "%{customdata[0]}<br>" +
                         "Percentage: %{customdata[2]:.1f}%<br>" +
                         "<extra></extra>",
            customdata=df[['Description', 'Category', 'Percentage']].values
        )
        
        if show_plot:
            fig.write_image("bert_topic_distribution.png", width=1400, height=max(600, len(df) * 40), scale=2)
            print("📊 BERT topic distribution saved as 'bert_topic_distribution.png'")
        
        return fig
    
    def create_topic_summary_table(self, show_plot: bool = True):
        """Create a comprehensive topic summary table"""
        if not self.topic_model:
            return
            
        # Get meaningful topic labels
        topic_labels, topic_descriptions = self.create_meaningful_topic_labels()
        
        # Prepare data for table
        table_data = []
        for _, row in self.topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Skip outlier topic
                topic_size = row['Count']
                words_weights = self.topic_model.get_topic(topic_id)[:5]
                top_words = [f"{word} ({weight:.3f})" for word, weight in words_weights]
                category = self.categorize_topic([word for word, _ in words_weights])
                
                table_data.append([
                    f"Topic {topic_id}",
                    topic_labels.get(topic_id, f"Topic {topic_id}"),
                    category,
                    topic_size,
                    f"{(topic_size / self.topic_info['Count'].sum()) * 100:.1f}%",
                    "<br>".join(top_words)
                ])
        
        # Sort by size
        table_data = sorted(table_data, key=lambda x: x[3], reverse=True)
        
        # Create table figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['ID', 'Topic Label', 'Legal Domain', 'Cases', '%', 'Top Keywords (weights)'],
                fill_color='#4472C4',
                font=dict(color='white', size=12),
                align='left',
                height=40
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color=[['#E8F1FF' if i % 2 == 0 else '#FFFFFF' for i in range(len(table_data))] for _ in range(6)],
                font=dict(color='black', size=11),
                align='left',
                height=30
            )
        )])
        
        fig.update_layout(
            title='📋 BERT Topic Analysis - Comprehensive Summary Table',
            title_font_size=20,
            title_x=0.5,
            height=max(400, len(table_data) * 35 + 150),
            width=1600,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        if show_plot:
            fig.write_image("bert_topic_summary_table.png", width=1600, height=max(400, len(table_data) * 35 + 150), scale=2)
            print("📋 BERT topic summary table saved as 'bert_topic_summary_table.png'")
        
        return fig
    
    def create_topic_keywords_heatmap(self, show_plot: bool = True):
        """Create topic keywords heatmap with legal domain annotations"""
        if not self.topic_model:
            return
            
        # Get top words for each topic
        topic_words_data = []
        all_words = set()
        
        for topic_id in range(min(15, self.n_topics)):  # Top 15 topics
            words_weights = self.topic_model.get_topic(topic_id)[:10]
            topic_words = {word: weight for word, weight in words_weights}
            topic_words_data.append(topic_words)
            all_words.update(topic_words.keys())
        
        # Create matrix
        word_list = list(all_words)[:30]  # Top 30 unique words
        topic_ids = list(range(min(15, self.n_topics)))
        
        matrix = []
        for topic_id in topic_ids:
            row = []
            for word in word_list:
                weight = topic_words_data[topic_id].get(word, 0)
                row.append(weight)
            matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=word_list,
            y=[f'Topic {i}' for i in topic_ids],
            colorscale='Viridis',
            hovertemplate='Topic: %{y}<br>Word: %{x}<br>Weight: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='🔥 Topic-Keywords Heatmap - Word Importance Across Topics',
            title_font_size=20,
            title_x=0.5,
            xaxis_title='Keywords',
            yaxis_title='Topics',
            height=800,
            width=1400,
            xaxis_tickangle=-45
        )
        
        if show_plot:
            # Save as image instead of using fig.show()
            fig.write_image("topic_keywords_heatmap.png", width=1400, height=800, scale=2)
            print("📊 Topic keywords heatmap saved as 'topic_keywords_heatmap.png'")
        
        return fig
    
    def create_legal_domain_pie_chart(self, show_plot: bool = True):
        """Create pie chart showing distribution of legal domains"""
        if not self.topic_model:
            return
            
        domain_counts = Counter()
        
        for _, row in self.topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Skip outlier topic
                topic_words = [word for word, _ in self.topic_model.get_topic(topic_id)[:8]]
                category = self.categorize_topic(topic_words)
                domain_counts[category] += row['Count']
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(domain_counts.keys()),
            values=list(domain_counts.values()),
            hole=.3,
            marker_colors=[self.domain_colors.get(domain, '#BDC3C7') for domain in domain_counts.keys()],
            hovertemplate='<b>%{label}</b><br>Cases: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title='⚖️ Legal Domain Distribution - Cases by Legal Area',
            title_font_size=20,
            title_x=0.5,
            height=600,
            width=800,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            )
        )
        
        if show_plot:
            # Save as image instead of using fig.show()
            fig.write_image("legal_domain_distribution.png", width=800, height=600, scale=2)
            print("📊 Legal domain distribution saved as 'legal_domain_distribution.png'")
        
        return fig
    
    def create_topic_similarity_network(self, show_plot: bool = True):
        """Create network graph showing topic similarities"""
        if not self.topic_model:
            return
            
        # Get topic similarities (using topic embeddings if available)
        try:
            # Get topic representations
            topic_embeddings = []
            topic_labels = []
            topic_categories = []
            topic_sizes = []
            
            for _, row in self.topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id != -1 and topic_id < 10:  # Top 10 topics for clarity
                    # Get topic vector (c-TF-IDF representation)
                    topic_vector = self.topic_model.c_tf_idf_[topic_id + 1]  # +1 because -1 is at index 0
                    topic_embeddings.append(topic_vector.toarray().flatten())
                    
                    topic_words = [word for word, _ in self.topic_model.get_topic(topic_id)[:3]]
                    topic_labels.append(f"Topic {topic_id}<br>{'<br>'.join(topic_words)}")
                    topic_categories.append(self.categorize_topic(topic_words))
                    topic_sizes.append(row['Count'])
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(topic_embeddings)
            
            # Create network layout
            import networkx as nx
            
            G = nx.Graph()
            
            # Add nodes
            for i, label in enumerate(topic_labels):
                G.add_node(i, label=label, category=topic_categories[i], size=topic_sizes[i])
            
            # Add edges for similar topics (threshold > 0.1)
            for i in range(len(similarities)):
                for j in range(i+1, len(similarities)):
                    if similarities[i][j] > 0.1:
                        G.add_edge(i, j, weight=similarities[i][j])
            
            # Get layout positions
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Prepare data for plotly
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(G.nodes[node]['label'])
                node_colors.append(self.domain_colors.get(G.nodes[node]['category'], '#BDC3C7'))
                node_sizes.append(max(20, G.nodes[node]['size'] / 10))  # Scale node size
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                showlegend=False
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                showlegend=False
            ))
            
            fig.update_layout(
                title='🕸️ Topic Similarity Network - Related Legal Topics',
                title_font_size=20,
                title_x=0.5,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Topics connected by similarity. Node size represents number of cases.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='#888', size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,
                width=1000
            )
            
            if show_plot:
                # Save as image instead of using fig.show()
                fig.write_image("topic_similarity_network.png", width=1000, height=700, scale=2)
                print("📊 Topic similarity network saved as 'topic_similarity_network.png'")
            
            return fig
            
        except Exception as e:
            print(f"⚠️ Could not create similarity network: {e}")
            return None
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all visualizations"""
        if not self.topic_model:
            return
            
        print("🎨 Creating comprehensive BERT topic visualization dashboard...")
        
        # Create all visualizations
        fig1 = self.create_topic_size_distribution(save_html=False)
        fig2 = self.create_legal_domain_pie_chart(save_html=False)
        fig3 = self.create_topic_keywords_heatmap(save_html=False)
        fig4 = self.create_topic_similarity_network(save_html=False)
        
        # Create subplot dashboard
        from plotly.subplots import make_subplots
        
        # Create a 2x2 subplot layout
        dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Topic Size Distribution', 'Legal Domain Distribution', 
                          'Topic Keywords Heatmap', 'Topic Similarity Network'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # Add traces from individual figures
        if fig1:
            for trace in fig1.data:
                dashboard.add_trace(trace, row=1, col=1)
        
        if fig2:
            for trace in fig2.data:
                dashboard.add_trace(trace, row=1, col=2)
        
        if fig3:
            for trace in fig3.data:
                dashboard.add_trace(trace, row=2, col=1)
        
        if fig4:
            for trace in fig4.data:
                dashboard.add_trace(trace, row=2, col=2)
        
        dashboard.update_layout(
            title='🔍 Legal BERT Topic Modeling - Comprehensive Analysis Dashboard',
            title_font_size=24,
            title_x=0.5,
            height=1200,
            width=1600,
            showlegend=False
        )
        
        dashboard.write_html("bert_topic_dashboard.html")
        print("💾 Comprehensive dashboard saved as 'bert_topic_dashboard.html'")
        
        return dashboard
    
    def generate_all_visualizations(self):
        """Generate all individual visualizations and save as PNG images"""
        if not self.topic_model:
            print("❌ No model loaded. Cannot create visualizations.")
            return
        
        print("\n🎨 Generating enhanced BERT topic visualizations with clear legends...")
        print("="*70)
        
        # Generate and save individual plots as images
        print("1. 📋 Creating comprehensive topic summary table...")
        self.create_topic_summary_table()
        
        print("2. 📊 Creating topic distribution chart...")
        self.create_topic_size_distribution()
        
        print("3. ⚖️ Creating legal domain pie chart...")
        self.create_legal_domain_pie_chart()
        
        print("4. 🔥 Creating topic keywords heatmap...")
        self.create_topic_keywords_heatmap()
        
        print("5. 🕸️ Creating topic similarity network...")
        self.create_topic_similarity_network()
        
        print("\n✅ All BERT topic visualizations saved as high-quality PNG images!")
        print("="*70)
        print("📁 Files created with clear topic legends:")
        print("   • bert_topic_summary_table.png      - Complete topic overview with labels")
        print("   • bert_topic_distribution.png       - Topic sizes with meaningful names")
        print("   • legal_domain_distribution.png     - Legal area breakdown") 
        print("   • topic_keywords_heatmap.png        - Word importance matrix")
        print("   • topic_similarity_network.png      - Topic relationship network")
        print("\n🎯 Each visualization includes:")
        print("   ✅ Clear topic labels (e.g., '🔒 Criminal: Homicide Cases')")
        print("   ✅ Legal domain categorization with emojis")
        print("   ✅ Professional legends and annotations")
        print("   ✅ High-resolution PNG format")
        print("\n🖼️ Open the PNG files to view your enhanced BERT topic analysis!")

def main():
    """Main function to generate all BERT topic visualizations"""
    print("🎨 BERT Topic Visualization Dashboard")
    print("="*50)
    
    # Initialize visualizer
    visualizer = BERTTopicVisualizer()
    
    # Generate all visualizations
    if visualizer.topic_model:
        visualizer.generate_all_visualizations()
    else:
        print("❌ Could not load BERT topic model. Please check the model path.")

if __name__ == "__main__":
    main()