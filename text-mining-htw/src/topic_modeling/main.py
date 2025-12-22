import os
import pandas as pd
from analysis import AdvancedLegalAnalytics
from visualization import LegalAnalyticsVisualizer
from era_analysis import EraAnalyzer
from outcome_prediction import OutcomePredictor


def main():
    """
    Main orchestrator to run the entire legal analytics pipeline,
    from topic modeling and trend analysis to final visualization and advanced analytics.
    """
    # --- Configuration ---
    CONFIG = {
        "input_file_path": 'data/processed/cleaned/parse_legal_cases.csv',
        "output_dir": 'output',
        "sample_size": None, # Using a sample for faster execution
        "min_topic_size": 15,
        "topics_to_track": ["Environmental", "Social", "Governance", "Finance"],
        "positive_outcome_label": "Affirmed",
        "negative_outcome_label": "Reversed/Vacated",
        "eras": {
            "Early 2000s": (2000, 2008),
            "Post-Recession": (2009, 2016),
            "Modern Era": (2017, 2024)
        }
    }

    print("Starting the advanced legal text analytics pipeline")
    print("=" * 60)

    # --- 1. Main Analysis Phase ---
    analyzer = AdvancedLegalAnalytics()
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    processed_data_path = os.path.join(CONFIG["output_dir"], 'analyzed_legal_data.csv')

    # Load data
    if os.path.exists(CONFIG["input_file_path"]):
        df = pd.read_csv(CONFIG["input_file_path"], low_memory=False)
    else:
        print(f"File not found: '{CONFIG['input_file_path']}'. Creating sample data.")
        df = analyzer.create_sample_data(n_samples=2000)

    df_subset = df.head(CONFIG["sample_size"]).copy() if CONFIG["sample_size"] else df.copy()

    documents = analyzer.prepare_documents(df_subset, text_column='opinion_text', date_column='date_filed')
    if len(documents) < 50:
        print("Error: Not enough documents for robust analysis after filtering.")
        return

    analyzer.create_outcome_variable()
    analyzer.fit_topic_model(documents, min_topic_size=CONFIG["min_topic_size"])
    analyzer.map_topics_to_judicial_classification()
    analyzer.map_to_sustainability_topics()

    analyzer.analyze_topic_trends(time_column='date_filed', jurisdiction_column='court_jurisdiction', output_dir=CONFIG["output_dir"])
    analyzer.model_outcomes(positive_outcome=CONFIG["positive_outcome_label"])
    analyzer.processed_df.to_csv(processed_data_path, index=False)
    print(f"\nProcessed data saved to: {processed_data_path}")

    # --- 2. Visualization Phase ---
    print("\n\nStarting the visualization phase...")
    print("=" * 60)
    df_for_viz = analyzer.processed_df.copy()

    if 'sustainability_category' in df_for_viz.columns:
        df_for_viz['topic_label'] = df_for_viz.apply(
            lambda row: row['sustainability_category'] if row['sustainability_category'] != 'Non-ESG' else row['judicial_category'],
            axis=1
        )

    visualizer = LegalAnalyticsVisualizer(processed_df=df_for_viz)
    visualizer.generate_all_visualizations(output_dir=CONFIG["output_dir"])

    # --- 3. Era Analysis Phase ---
    print("\n\nStarting the historical era analysis...")
    print("=" * 60)
    era_analyzer = EraAnalyzer(processed_df=df_for_viz, eras=CONFIG["eras"])
    era_analyzer.run_era_analysis(output_dir=CONFIG["output_dir"])

    # --- 4. Outcome Prediction Analysis ---
    print("\n\nStarting the outcome prediction analysis...")
    print("=" * 60)
    outcome_predictor = OutcomePredictor(processed_df=analyzer.processed_df)
    outcome_predictor.run_prediction_analysis(
        output_dir=CONFIG["output_dir"],
        target_outcome=CONFIG["negative_outcome_label"]
    )

    print("\n\nFull analysis and visualization pipeline complete.")


if __name__ == "__main__":
    main()
