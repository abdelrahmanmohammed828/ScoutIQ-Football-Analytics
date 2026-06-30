# ⚽ ScoutIQ — Find the Next Mo Salah, Before the Market Does

ScoutIQ is a machine learning-powered scouting platform that identifies statistically undervalued U23 football talent in European leagues, by comparing player performance profiles against elite benchmarks (Mohamed Salah, Kevin De Bruyne) and quantifying the financial opportunity in signing them early.

**[Live App →](https://scoutiq-football-analytics-sjwf5wsfrvy5qegi9sau5g.streamlit.app/)**

---

## The Problem

Roma signed Mohamed Salah for €15M in 2016. Liverpool bought him a year later for £36.9M — a 146% increase, for the same underlying statistical profile. The market is slow to price emerging talent correctly, and that gap is where transfer value is won or lost.

ScoutIQ exists to find players who already match an elite statistical profile, before the market catches up.

## What It Does

- Analyzes **264 U23 players** across the Eredivisie, Primeira Liga, and EFL Championship (2023/24 season)
- Benchmarks each player against two elite reference profiles: Mohamed Salah (AS Roma, 2015/16) and Kevin De Bruyne (VfL Wolfsburg, 2014/15)
- Scores statistical similarity using a **hybrid model (60% Cosine + 40% Euclidean distance)** across 7 performance features (goals/90, assists/90, shots/90, shots on target/90, shot accuracy, finishing quality)
- Layers in a **financial decision framework**: market value (Transfermarkt), arbitrage potential vs. benchmark valuation, and a Value Efficiency Score (VES)
- Produces a final weighted **Decision Score** (50% Similarity + 30% VES + 20% Performance) and a recommendation tier: STRONG BUY / BUY / MONITOR / PASS
- Surfaces everything through an interactive Streamlit dashboard: scout engine with filters, individual player profiles with radar comparisons, and market intelligence views (arbitrage rankings, efficiency frontier)

## Methodology

Built following the **CRISP-DM framework** across 7 phases:

1. **Data collection** — FBref.com (performance stats, 2023/24 season) + Transfermarkt (market valuations, Aug/Sep 2024)
2. **Preprocessing** — feature engineering of per-90 statistics, cleaning, standardization
3. **Dimensionality reduction** — PCA (80.0% variance explained)
4. **Clustering** — K-Means (K=4) to group players by statistical profile
5. **Similarity scoring** — hybrid Cosine + Euclidean distance against benchmark vectors
6. **Financial layer** — Value Efficiency Score, arbitrage calculation (benchmark transfer value vs. current market value), cost-per-goal
7. **Decision framework** — weighted composite score mapped to actionable recommendation tiers

This project was built to address four gaps commonly noted in sports analytics literature: moving from pure prediction to actionable decisions, explainability (feature-level breakdowns and natural-language rationale per player), integration of financial context alongside performance stats, and clear decision rules rather than raw scores alone.

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn (StandardScaler, PCA, K-Means)
- **Streamlit** — interactive multi-page dashboard
- **Plotly** — radar charts, bar charts, efficiency frontier scatter plots

## Project Structure

```
├── scoutiq_app_final.py        # Main Streamlit application
├── scoutiq_master.csv          # Player profiles, scores, and recommendations
├── scoutiq_complete.csv        # Full feature set across all analyzed players
├── scoutiq_salah_financial.csv # Salah-benchmark financial comparisons
├── scoutiq_kdb_financial.csv   # De Bruyne-benchmark financial comparisons
├── scoutiq_logo.png
└── requirements.txt
```

## Running Locally

```bash
git clone https://github.com/abdelrahmanmohammed828/ScoutIQ-Football-Analytics.git
cd ScoutIQ-Football-Analytics
pip install -r requirements.txt
streamlit run scoutiq_app_final.py
```

## Results

- Identified up to **€14.2M** in transfer arbitrage potential per signing on the Salah profile, and **€17.2M** on the De Bruyne profile
- 34 of the 264 analyzed players have full Transfermarkt valuations integrated for financial scoring

## About

Built by **Abdelrahman M. Elhosary** as part of the Master of Business Analytics program (ABW508 Analytics Lab) at Universiti Sains Malaysia, under Dr. Khaw Khai Wah.

## Disclaimer

This is an academic/portfolio project. Player valuations and recommendations are illustrative outputs of a statistical model and are not intended as real-world transfer advice.
