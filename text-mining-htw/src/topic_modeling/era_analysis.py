import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EraAnalyzer:
    """
    Analyzes legal topics across different historical eras to identify trends
    and shifts in civil litigation focus.
    """

    def __init__(self, processed_df, eras):
        self.df = processed_df.copy()
        self.eras = eras
        self._prepare_data()

    def _prepare_data(self):
        """Prepares the DataFrame for era-based analysis."""
        self.df['date_filed'] = pd.to_datetime(self.df['date_filed'], errors='coerce')
        self.df['year'] = self.df['date_filed'].dt.year
        self.df.dropna(subset=['year', 'topic_label'], inplace=True)

        def assign_era(year):
            for era_name, (start, end) in self.eras.items():
                if start <= year <= end:
                    return era_name
            return None
        self.df['era'] = self.df['year'].apply(assign_era)

    def analyze_dominant_topics_by_era(self, save_path):
        """Identifies and visualizes the most dominant legal topics in each era."""
        print("Analyzing dominant topics per era...")
        if 'era' not in self.df.columns or self.df['era'].isnull().all():
            print("Could not perform era analysis. Check date ranges and data.")
            return

        top_topics_by_era = self.df.groupby('era')['topic_label'].value_counts().groupby(level=0).nlargest(5).reset_index(level=0, drop=True)
        top_topics_df = top_topics_by_era.reset_index(name='count')

        plt.figure(figsize=(15, 10))
        sns.barplot(data=top_topics_df, x='count', y='topic_label', hue='era', dodge=False, palette='Blues_d')
        plt.title('Top 5 Dominant Legal Topics by Era', fontsize=18, fontweight='bold')
        plt.xlabel('Number of Cases', fontsize=12)
        plt.ylabel('Topic', fontsize=12)
        plt.legend(title='Era', title_fontsize='13', fontsize='11')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Dominant topics chart saved to: {save_path}")

    def analyze_civil_litigation_change(self, keywords, save_path):
        """Tracks the frequency of specific civil litigation topics over time."""
        print(f"Analyzing trends for civil litigation keywords: {keywords}...")
        df_civil = self.df[self.df['topic_label'].str.contains('|'.join(keywords), case=False, na=False)]
        if df_civil.empty:
            print("No topics found matching the civil litigation keywords.")
            return

        total_cases_per_year = self.df['year'].value_counts().sort_index()
        civil_cases_per_year = df_civil['year'].value_counts().sort_index()
        trend_percentage = (civil_cases_per_year / total_cases_per_year * 100).fillna(0)

        plt.figure(figsize=(14, 7))
        trend_percentage.plot(kind='line', marker='o', linestyle='-', color='blue')
        plt.title('Focus on Key Civil Litigation Topics Over Time', fontsize=16, fontweight='bold')
        plt.ylabel('Percentage of All Cases (%)', fontsize=12)
        plt.xlabel('Year', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Civil litigation trend chart saved to: {save_path}")

    def run_era_analysis(self, output_dir):
        """Runs the full suite of era analysis and saves the visualizations."""
        self.analyze_dominant_topics_by_era(
            save_path=os.path.join(output_dir, 'dominant_topics_by_era.png')
        )
        self.analyze_civil_litigation_change(
            keywords=['consumer', 'civil rights', 'employment', 'contract'],
            save_path=os.path.join(output_dir, 'civil_litigation_trends.png')
        )
