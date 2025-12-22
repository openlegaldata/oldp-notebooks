import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues_r")


class LegalAnalyticsVisualizer:
    """
    A comprehensive visualization class for legal analytics data.
    Creates presentation-ready static charts including bar charts by state, histograms, and trend analysis.
    """

    def __init__(self, processed_df=None):
        self.df = processed_df
        self.state_abbreviations = {
            'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
            'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
            'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA',
            'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
            'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
            'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH',
            'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY', 'north carolina': 'NC',
            'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA',
            'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD', 'tennessee': 'TN',
            'texas': 'TX', 'utah': 'UT', 'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA',
            'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC'
        }

    def load_processed_data(self, file_path):
        """Load the processed data from the main analytics script."""
        if os.path.exists(file_path):
            self.df = pd.read_csv(file_path)
            print(f"Processed data loaded: {len(self.df)} records")
        else:
            print(f"File not found: {file_path}")

    def extract_state_from_jurisdiction(self, jurisdiction_text):
        """Extract state information from court jurisdiction text."""
        if pd.isna(jurisdiction_text) or not isinstance(jurisdiction_text, str):
            return 'Unknown'

        jurisdiction_lower = jurisdiction_text.lower()
        for state_name, abbrev in self.state_abbreviations.items():
            if state_name in jurisdiction_lower or abbrev.lower() in jurisdiction_lower:
                return abbrev
        return 'Unknown'

    def create_state_case_volume_chart(self, save_path='us_case_volume.png'):
        """Create a static bar chart showing case volume by state."""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return

        print("Creating state case volume chart...")
        self.df['state'] = self.df['court_jurisdiction'].apply(self.extract_state_from_jurisdiction)
        state_counts = self.df['state'].value_counts().nlargest(20)
        state_counts = state_counts[state_counts.index != 'Unknown']

        plt.figure(figsize=(12, 8))
        sns.barplot(x=state_counts.values, y=state_counts.index, palette="Blues_d")
        plt.title('Top 20 States by Legal Case Volume', fontsize=18, fontweight='bold')
        plt.xlabel('Number of Cases', fontsize=12)
        plt.ylabel('State', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"State case volume chart saved to: {save_path}")

    def create_state_outcome_success_rate_chart(self, save_path='us_outcome_rate.png'):
        """Create a static bar chart showing success rates (Affirmed cases) by state."""
        if self.df is None or 'outcome' not in self.df.columns:
            print("No outcome data available.")
            return

        print("Creating state outcome success rate chart...")
        if 'state' not in self.df.columns:
            self.df['state'] = self.df['court_jurisdiction'].apply(self.extract_state_from_jurisdiction)

        state_outcomes = self.df.groupby('state')['outcome'].apply(
            lambda x: (x == 'Affirmed').sum() / len(x) * 100 if len(x) > 0 else 0
        ).reset_index()
        state_outcomes.columns = ['state', 'success_rate']

        state_case_counts = self.df['state'].value_counts()
        valid_states = state_case_counts[state_case_counts >= 10].index
        state_outcomes = state_outcomes[
            (state_outcomes['state'] != 'Unknown') &
            (state_outcomes['state'].isin(valid_states))
            ].nlargest(20, 'success_rate')

        plt.figure(figsize=(12, 8))
        sns.barplot(x='success_rate', y='state', data=state_outcomes, palette="Blues_d")
        plt.title('Case Success Rate by State (% Affirmed)', fontsize=18, fontweight='bold')
        plt.xlabel('Success Rate (%)', fontsize=12)
        plt.ylabel('State', fontsize=12)
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"State success rate chart saved to: {save_path}")

    def create_topic_distribution_chart(self, save_path='topic_distribution_by_state.png'):
        """Create a stacked bar chart showing topic distribution across top states."""
        if self.df is None or 'topic_label' not in self.df.columns:
            print("No topic data available.")
            return

        print("Creating topic distribution chart...")
        if 'state' not in self.df.columns:
            self.df['state'] = self.df['court_jurisdiction'].apply(self.extract_state_from_jurisdiction)

        top_states = self.df['state'].value_counts().nlargest(10).index
        df_top_states = self.df[self.df['state'].isin(top_states)]

        topic_state_counts = pd.crosstab(df_top_states['state'], df_top_states['topic_label'], normalize='index') * 100

        # Plotting
        topic_state_counts.plot(kind='bar', stacked=True, figsize=(15, 10), colormap='Blues')
        plt.title('Distribution of Legal Topics Across Top 10 States', fontsize=18, fontweight='bold')
        plt.ylabel('Percentage of Cases (%)', fontsize=12)
        plt.xlabel('State', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Legal Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Topic distribution chart saved to: {save_path}")

    def create_case_timeline_histogram(self, save_path='case_timeline_histogram.png'):
        """Create a histogram showing case distribution over time."""
        if self.df is None or 'date_filed' not in self.df.columns:
            print("No date data available.")
            return

        print("Creating case timeline histogram...")
        self.df['date_filed'] = pd.to_datetime(self.df['date_filed'], errors='coerce')
        self.df['year'] = self.df['date_filed'].dt.year
        valid_data = self.df.dropna(subset=['year'])

        plt.figure(figsize=(14, 8))
        plt.hist(valid_data['year'], bins=30, color='steelblue', alpha=0.8, edgecolor='black')
        plt.title('Distribution of Legal Cases Over Time', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Cases', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Case timeline histogram saved to: {save_path}")

    def create_outcome_distribution_histogram(self, save_path='outcome_distribution_histogram.png'):
        """Create a histogram showing the distribution of case outcomes."""
        if self.df is None or 'outcome' not in self.df.columns:
            print("No outcome data available.")
            return

        print("Creating outcome distribution histogram...")
        plt.figure(figsize=(12, 8))
        outcome_counts = self.df['outcome'].value_counts()
        bars = plt.bar(outcome_counts.index, outcome_counts.values, color='skyblue', alpha=0.8, edgecolor='black')
        plt.title('Distribution of Legal Case Outcomes', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Outcome Type', fontsize=14)
        plt.ylabel('Number of Cases', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Outcome distribution histogram saved to: {save_path}")

    def create_topic_popularity_chart(self, save_path='topic_popularity_chart.png'):
        """Create a horizontal bar chart showing most popular topics."""
        if self.df is None or 'topic_label' not in self.df.columns:
            print("No topic data available.")
            return

        print("Creating topic popularity chart...")
        plt.figure(figsize=(14, 10))
        topic_counts = self.df['topic_label'].value_counts().head(15)
        bars = plt.barh(range(len(topic_counts)), topic_counts.values, color='cornflowerblue', alpha=0.8)
        plt.yticks(range(len(topic_counts)), [label[:50] + '...' if len(label) > 50 else label
                                              for label in topic_counts.index])
        plt.xlabel('Number of Cases', fontsize=14)
        plt.title('Most Common Legal Topics in Cases', fontsize=18, fontweight='bold', pad=20)
        for i, (bar, value) in enumerate(zip(bars, topic_counts.values)):
            plt.text(value + 0.5, bar.get_y() + bar.get_height() / 2,
                     str(value), va='center', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Topic popularity chart saved to: {save_path}")

    def generate_all_visualizations(self, output_dir='visualizations'):
        """Generate all visualizations at once."""
        os.makedirs(output_dir, exist_ok=True)
        print("\nGenerating all visualizations...")
        print("=" * 50)

        self.create_state_case_volume_chart(f'{output_dir}/us_case_volume.png')
        self.create_state_outcome_success_rate_chart(f'{output_dir}/us_outcome_rate.png')
        self.create_topic_distribution_chart(f'{output_dir}/topic_distribution_by_state.png')
        self.create_case_timeline_histogram(f'{output_dir}/case_timeline_histogram.png')
        self.create_outcome_distribution_histogram(f'{output_dir}/outcome_distribution_histogram.png')
        self.create_topic_popularity_chart(f'{output_dir}/topic_popularity_chart.png')

        print("=" * 50)
        print(f"All visualizations completed! Check the '{output_dir}' folder.")
