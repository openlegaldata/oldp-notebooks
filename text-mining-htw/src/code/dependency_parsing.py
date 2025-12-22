import pandas as pd
import spacy
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Set
import json

# Load spaCy model with dependency parsing
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy model loaded successfully")
except OSError:
    print("❌ Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

class LegalDependencyParser:
    """Extract subject-verb-object relationships and legal reasoning chains"""
    
    def __init__(self):
        # Legal entities and concepts for enhanced extraction
        self.legal_entities = {
            'court', 'judge', 'plaintiff', 'defendant', 'appellant', 'appellee',
            'petitioner', 'respondent', 'jury', 'attorney', 'counsel', 'party',
            'parties', 'witness', 'state', 'government', 'prosecution', 'defense'
        }
        
        self.legal_verbs = {
            'hold', 'held', 'find', 'found', 'rule', 'ruled', 'decide', 'decided',
            'determine', 'determined', 'conclude', 'concluded', 'affirm', 'affirmed',
            'reverse', 'reversed', 'remand', 'remanded', 'dismiss', 'dismissed',
            'grant', 'granted', 'deny', 'denied', 'order', 'ordered', 'require',
            'required', 'prohibit', 'prohibited', 'allow', 'allowed'
        }
        
        self.reasoning_markers = {
            'because', 'since', 'therefore', 'thus', 'accordingly', 'consequently',
            'however', 'nevertheless', 'moreover', 'furthermore', 'additionally',
            'given', 'considering', 'based on', 'in light of', 'as a result'
        }
    
    def extract_svo_relationships(self, text: str) -> List[Dict]:
        """Extract subject-verb-object relationships from text"""
        if not nlp or not text:
            return []
        
        doc = nlp(text)
        svo_relationships = []
        
        for sent in doc.sents:
            # Find the root verb
            root = None
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    root = token
                    break
            
            if not root:
                continue
            
            # Extract subject
            subjects = []
            for child in root.children:
                if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    subjects.append(self._get_full_phrase(child))
            
            # Extract objects
            objects = []
            for child in root.children:
                if child.dep_ in ["dobj", "pobj", "attr", "dative"]:
                    objects.append(self._get_full_phrase(child))
                elif child.dep_ == "prep":
                    # Get prepositional objects
                    for prep_child in child.children:
                        if prep_child.dep_ == "pobj":
                            objects.append(f"{child.text} {self._get_full_phrase(prep_child)}")
            
            # Create relationship record
            if subjects and root:
                relationship = {
                    'sentence': sent.text.strip(),
                    'subject': subjects,
                    'verb': root.lemma_,
                    'verb_text': root.text,
                    'objects': objects,
                    'is_legal_verb': root.lemma_.lower() in self.legal_verbs,
                    'is_legal_subject': any(self._contains_legal_entity(subj) for subj in subjects),
                    'sentence_start': sent.start_char,
                    'sentence_end': sent.end_char
                }
                
                svo_relationships.append(relationship)
        
        return svo_relationships
    
    def _get_full_phrase(self, token) -> str:
        """Get the full phrase for a token including its modifiers"""
        phrase_tokens = [token]
        
        # Add children (modifiers, determiners, etc.)
        for child in token.subtree:
            if child != token:
                phrase_tokens.append(child)
        
        # Sort by position in sentence
        phrase_tokens.sort(key=lambda t: t.i)
        return " ".join([t.text for t in phrase_tokens])
    
    def _contains_legal_entity(self, text: str) -> bool:
        """Check if text contains legal entities"""
        text_lower = text.lower()
        return any(entity in text_lower for entity in self.legal_entities)
    
    def extract_reasoning_chains(self, text: str) -> List[Dict]:
        """Extract legal reasoning chains from text"""
        if not nlp or not text:
            return []
        
        doc = nlp(text)
        reasoning_chains = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Look for reasoning markers
            reasoning_markers_found = []
            for marker in self.reasoning_markers:
                if marker.lower() in sent_text.lower():
                    reasoning_markers_found.append(marker)
            
            if not reasoning_markers_found:
                continue
            
            # Analyze sentence structure for reasoning
            premises = []
            conclusions = []
            
            # Split by reasoning markers
            for marker in reasoning_markers_found:
                pattern = rf'\b{re.escape(marker)}\b'
                parts = re.split(pattern, sent_text, flags=re.IGNORECASE)
                
                if len(parts) >= 2:
                    if marker.lower() in ['because', 'since', 'given', 'considering']:
                        # Premise comes after marker
                        premises.extend([p.strip() for p in parts[1:] if p.strip()])
                        conclusions.append(parts[0].strip())
                    else:
                        # Conclusion comes after marker
                        premises.append(parts[0].strip())
                        conclusions.extend([p.strip() for p in parts[1:] if p.strip()])
            
            # Extract legal concepts mentioned
            legal_concepts = self._extract_legal_concepts(sent_text)
            
            reasoning_chain = {
                'sentence': sent_text,
                'reasoning_markers': reasoning_markers_found,
                'premises': premises,
                'conclusions': conclusions,
                'legal_concepts': legal_concepts,
                'sentence_start': sent.start_char,
                'sentence_end': sent.end_char,
                'reasoning_type': self._classify_reasoning_type(reasoning_markers_found)
            }
            
            reasoning_chains.append(reasoning_chain)
        
        return reasoning_chains
    
    def _extract_legal_concepts(self, text: str) -> List[str]:
        """Extract legal concepts from text"""
        legal_patterns = [
            r'\b(?:statute|law|rule|regulation|code|act)\b',
            r'\b(?:precedent|case law|holding|decision)\b',
            r'\b(?:due process|equal protection|probable cause)\b',
            r'\b(?:contract|tort|criminal|civil|constitutional)\b',
            r'\b(?:evidence|testimony|discovery|motion)\b',
            r'\b(?:liability|damages|injunction|remedy)\b'
        ]
        
        concepts = []
        for pattern in legal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        return list(set(concepts))
    
    def _classify_reasoning_type(self, markers: List[str]) -> str:
        """Classify the type of legal reasoning"""
        causal_markers = {'because', 'since', 'given', 'due to', 'as a result'}
        conclusive_markers = {'therefore', 'thus', 'consequently', 'accordingly'}
        contrastive_markers = {'however', 'nevertheless', 'but', 'although'}
        additive_markers = {'moreover', 'furthermore', 'additionally', 'also'}
        
        marker_set = set(m.lower() for m in markers)
        
        if marker_set & causal_markers:
            return 'causal'
        elif marker_set & conclusive_markers:
            return 'conclusive'
        elif marker_set & contrastive_markers:
            return 'contrastive'
        elif marker_set & additive_markers:
            return 'additive'
        else:
            return 'other'
    
    def build_reasoning_graph(self, reasoning_chains: List[Dict]) -> nx.DiGraph:
        """Build a graph of reasoning relationships"""
        G = nx.DiGraph()
        
        for i, chain in enumerate(reasoning_chains):
            chain_id = f"chain_{i}"
            
            # Add nodes for premises and conclusions
            for premise in chain['premises']:
                if premise:
                    premise_id = f"premise_{hash(premise) % 10000}"
                    G.add_node(premise_id, text=premise, type='premise', chain=chain_id)
            
            for conclusion in chain['conclusions']:
                if conclusion:
                    conclusion_id = f"conclusion_{hash(conclusion) % 10000}"
                    G.add_node(conclusion_id, text=conclusion, type='conclusion', chain=chain_id)
                    
                    # Connect premises to conclusions
                    for premise in chain['premises']:
                        if premise:
                            premise_id = f"premise_{hash(premise) % 10000}"
                            G.add_edge(premise_id, conclusion_id, 
                                     reasoning_type=chain['reasoning_type'],
                                     markers=chain['reasoning_markers'])
        
        return G

def load_data(file_path: str) -> pd.DataFrame:
    """Load the legal cases dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded: {df.shape[0]} cases, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return pd.DataFrame()

def analyze_svo_relationships(svo_data: List[Dict]) -> Dict:
    """Analyze subject-verb-object relationships"""
    if not svo_data:
        return {}
    
    analysis = {
        'total_relationships': len(svo_data),
        'legal_verb_count': sum(1 for rel in svo_data if rel['is_legal_verb']),
        'legal_subject_count': sum(1 for rel in svo_data if rel['is_legal_subject']),
        'most_common_verbs': Counter([rel['verb'] for rel in svo_data]).most_common(10),
        'most_common_subjects': Counter([subj for rel in svo_data for subj in rel['subject']]).most_common(10),
        'verb_subject_pairs': Counter([(rel['verb'], ', '.join(rel['subject'])) for rel in svo_data]).most_common(10)
    }
    
    return analysis

def analyze_reasoning_chains(reasoning_data: List[Dict]) -> Dict:
    """Analyze legal reasoning chains"""
    if not reasoning_data:
        return {}
    
    analysis = {
        'total_chains': len(reasoning_data),
        'reasoning_types': Counter([chain['reasoning_type'] for chain in reasoning_data]),
        'most_common_markers': Counter([marker for chain in reasoning_data for marker in chain['reasoning_markers']]).most_common(10),
        'legal_concepts': Counter([concept for chain in reasoning_data for concept in chain['legal_concepts']]).most_common(10),
        'avg_premises_per_chain': sum(len(chain['premises']) for chain in reasoning_data) / len(reasoning_data),
        'avg_conclusions_per_chain': sum(len(chain['conclusions']) for chain in reasoning_data) / len(reasoning_data)
    }
    
    return analysis

def visualize_svo_analysis(svo_analysis: Dict, save_plots: bool = True):
    """Create visualizations for SVO analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Most common verbs
    verbs, counts = zip(*svo_analysis['most_common_verbs'][:10])
    axes[0, 0].barh(verbs, counts, color='steelblue')
    axes[0, 0].set_title('Most Common Legal Verbs', fontsize=14)
    axes[0, 0].set_xlabel('Frequency')
    
    # 2. Legal vs Non-legal relationships
    legal_counts = [
        svo_analysis['legal_verb_count'],
        svo_analysis['total_relationships'] - svo_analysis['legal_verb_count']
    ]
    axes[0, 1].pie(legal_counts, labels=['Legal Verbs', 'Other Verbs'], autopct='%1.1f%%')
    axes[0, 1].set_title('Legal vs Non-Legal Verbs', fontsize=14)
    
    # 3. Subject types
    legal_subject_counts = [
        svo_analysis['legal_subject_count'],
        svo_analysis['total_relationships'] - svo_analysis['legal_subject_count']
    ]
    axes[1, 0].pie(legal_subject_counts, labels=['Legal Subjects', 'Other Subjects'], autopct='%1.1f%%')
    axes[1, 0].set_title('Legal vs Non-Legal Subjects', fontsize=14)
    
    # 4. Top verb-subject pairs
    if svo_analysis['verb_subject_pairs']:
        pairs, pair_counts = zip(*svo_analysis['verb_subject_pairs'][:8])
        pair_labels = [f"{verb}\n({subj[:30]}...)" if len(subj) > 30 else f"{verb}\n({subj})" 
                      for verb, subj in pairs]
        axes[1, 1].bar(range(len(pair_labels)), pair_counts, color='lightcoral')
        axes[1, 1].set_title('Top Verb-Subject Pairs', fontsize=14)
        axes[1, 1].set_xticks(range(len(pair_labels)))
        axes[1, 1].set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('svo_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 SVO analysis saved as 'svo_analysis.png'")
    
    plt.show()

def visualize_reasoning_analysis(reasoning_analysis: Dict, save_plots: bool = True):
    """Create visualizations for reasoning chain analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Reasoning types distribution
    types = list(reasoning_analysis['reasoning_types'].keys())
    counts = list(reasoning_analysis['reasoning_types'].values())
    axes[0, 0].bar(types, counts, color='forestgreen')
    axes[0, 0].set_title('Reasoning Types Distribution', fontsize=14)
    axes[0, 0].set_xlabel('Reasoning Type')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Most common reasoning markers
    if reasoning_analysis['most_common_markers']:
        markers, marker_counts = zip(*reasoning_analysis['most_common_markers'][:10])
        axes[0, 1].barh(markers, marker_counts, color='orange')
        axes[0, 1].set_title('Most Common Reasoning Markers', fontsize=14)
        axes[0, 1].set_xlabel('Frequency')
    
    # 3. Legal concepts frequency
    if reasoning_analysis['legal_concepts']:
        concepts, concept_counts = zip(*reasoning_analysis['legal_concepts'][:10])
        axes[1, 0].barh(concepts, concept_counts, color='purple')
        axes[1, 0].set_title('Most Common Legal Concepts', fontsize=14)
        axes[1, 0].set_xlabel('Frequency')
    
    # 4. Chain complexity
    chain_metrics = [
        reasoning_analysis['avg_premises_per_chain'],
        reasoning_analysis['avg_conclusions_per_chain']
    ]
    axes[1, 1].bar(['Avg Premises', 'Avg Conclusions'], chain_metrics, color='teal')
    axes[1, 1].set_title('Average Chain Complexity', fontsize=14)
    axes[1, 1].set_ylabel('Average Count')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('reasoning_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Reasoning analysis saved as 'reasoning_analysis.png'")
    
    plt.show()

def visualize_reasoning_graph(graph: nx.DiGraph, max_nodes: int = 20, save_plot: bool = True):
    """Visualize reasoning graph"""
    if len(graph.nodes()) == 0:
        print("No reasoning graph to visualize")
        return
    
    # Limit nodes for visualization
    if len(graph.nodes()) > max_nodes:
        # Get subgraph with most connected nodes
        node_degrees = dict(graph.degree())
        top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:max_nodes]
        graph = graph.subgraph(top_nodes)
    
    plt.figure(figsize=(14, 10))
    
    # Create layout
    pos = nx.spring_layout(graph, k=1, iterations=50)
    
    # Color nodes by type
    node_colors = []
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type', 'unknown')
        if node_type == 'premise':
            node_colors.append('lightblue')
        elif node_type == 'conclusion':
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightgray')
    
    # Draw graph
    nx.draw(graph, pos, node_color=node_colors, node_size=1000, 
            with_labels=False, arrows=True, edge_color='gray', 
            arrowsize=20, arrowstyle='->')
    
    # Add labels with truncated text
    labels = {}
    for node in graph.nodes():
        text = graph.nodes[node].get('text', node)
        labels[node] = text[:30] + '...' if len(text) > 30 else text
    
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)
    
    plt.title('Legal Reasoning Graph\n(Blue: Premises, Red: Conclusions)', fontsize=16)
    plt.axis('off')
    
    if save_plot:
        plt.savefig('reasoning_graph.png', dpi=300, bbox_inches='tight')
        print("📊 Reasoning graph saved as 'reasoning_graph.png'")
    
    plt.tight_layout()
    plt.show()

def save_results(svo_data: List[Dict], reasoning_data: List[Dict], 
                svo_analysis: Dict, reasoning_analysis: Dict,
                output_dir: str = "data/processed/"):
    """Save dependency parsing results"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save SVO relationships
    svo_df = pd.DataFrame(svo_data)
    svo_file = os.path.join(output_dir, "svo_relationships.csv")
    svo_df.to_csv(svo_file, index=False)
    print(f"💾 SVO relationships saved: {svo_file}")
    
    # Save reasoning chains
    reasoning_df = pd.DataFrame(reasoning_data)
    reasoning_file = os.path.join(output_dir, "reasoning_chains.csv")
    reasoning_df.to_csv(reasoning_file, index=False)
    print(f"💾 Reasoning chains saved: {reasoning_file}")
    
    # Save analysis results
    analysis_file = os.path.join(output_dir, "dependency_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump({
            'svo_analysis': svo_analysis,
            'reasoning_analysis': reasoning_analysis
        }, f, indent=2, default=str)
    print(f"💾 Analysis results saved: {analysis_file}")

def process_legal_text(text: str, parser: LegalDependencyParser) -> Tuple[List[Dict], List[Dict]]:
    """Process single legal text for dependency parsing"""
    svo_relationships = parser.extract_svo_relationships(text)
    reasoning_chains = parser.extract_reasoning_chains(text)
    return svo_relationships, reasoning_chains

def main():
    """Main execution function"""
    print("🚀 Legal Dependency Parsing & Reasoning Chain Analysis")
    print("="*60)
    
    # Initialize parser
    parser = LegalDependencyParser()
    
    # Load data
    data_file = "Text-mining-HTW/data/processed/cleaned/parse_legal_cases.csv"
    df = load_data(data_file)
    
    if df.empty:
        print("❌ No data loaded. Please check the file path.")
        return
    
    # Process cases
    print(f"\n📊 Processing cases for dependency parsing...")
    all_svo_data = []
    all_reasoning_data = []
    
    num_cases = min(50, len(df))  # Process first 50 cases
    
    for i, (_, row) in enumerate(df.head(num_cases).iterrows()):
        if pd.isna(row['opinion_text']):
            continue
            
        svo_relationships, reasoning_chains = process_legal_text(str(row['opinion_text']), parser)
        
        # Add case metadata
        for rel in svo_relationships:
            rel['case_id'] = row['id']
            rel['case_name'] = row.get('case_name_short', 'Unknown')
            rel['court_type'] = row.get('court_type', 'Unknown')
        
        for chain in reasoning_chains:
            chain['case_id'] = row['id']
            chain['case_name'] = row.get('case_name_short', 'Unknown')
            chain['court_type'] = row.get('court_type', 'Unknown')
        
        all_svo_data.extend(svo_relationships)
        all_reasoning_data.extend(reasoning_chains)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} cases...")
    
    print(f"✓ Processing complete!")
    print(f"  - Extracted {len(all_svo_data)} SVO relationships")
    print(f"  - Found {len(all_reasoning_data)} reasoning chains")
    
    # Analyze results
    print(f"\n📈 Analyzing results...")
    svo_analysis = analyze_svo_relationships(all_svo_data)
    reasoning_analysis = analyze_reasoning_chains(all_reasoning_data)
    
    # Print analysis summary
    print(f"\n🔍 ANALYSIS SUMMARY")
    print(f"="*40)
    print(f"SVO Relationships: {svo_analysis.get('total_relationships', 0)}")
    print(f"Legal verbs: {svo_analysis.get('legal_verb_count', 0)}")
    print(f"Legal subjects: {svo_analysis.get('legal_subject_count', 0)}")
    print(f"Reasoning chains: {reasoning_analysis.get('total_chains', 0)}")
    
    if reasoning_analysis.get('reasoning_types'):
        print(f"Reasoning types: {dict(reasoning_analysis['reasoning_types'])}")
    
    # Create visualizations
    if all_svo_data:
        print(f"\n📊 Creating SVO visualizations...")
        visualize_svo_analysis(svo_analysis)
    
    if all_reasoning_data:
        print(f"\n📊 Creating reasoning visualizations...")
        visualize_reasoning_analysis(reasoning_analysis)
        
        # Build and visualize reasoning graph
        reasoning_graph = parser.build_reasoning_graph(all_reasoning_data)
        if len(reasoning_graph.nodes()) > 0:
            print(f"\n🕸️  Creating reasoning graph...")
            visualize_reasoning_graph(reasoning_graph)
    
    # Save results
    print(f"\n💾 Saving results...")
    save_results(all_svo_data, all_reasoning_data, svo_analysis, reasoning_analysis)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()