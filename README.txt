# ⚽ Expected Goals (xG) Model & Player Scouting Engine

An end-to-end Football Analytics and Player Scouting web application built with **Python**, **Streamlit**, **Scikit-Learn**, and **mplsoccer**, powered by open event data from **StatsBomb** (FIFA World Cup 2022).

## 🚀 Key Features

1. **📊 Match Shot Map & xG Scoreboard**:
   - Analyzes team shot selection and calculates custom Expected Goals (xG) per match.
   - Plots inverted pitch shot maps (Home vs Away) powered by `mplsoccer`.
   
2. **🔮 Real-Time xG Calculator**:
   - Interactive simulator predicting goal probability based on shot distance, angle, header, and play type.
   - Real-time animated pitch visualization showing exact shot coordinates.

3. **🔍 Player Scouting & Cosine Similarity Engine**:
   - Machine learning similarity model finding top player profiles using **Cosine Similarity**.
   - Interactive Radar Comparison Charts comparing target and recommended players.

## 🛠️ Tech Stack & Libraries

- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn (Random Forest Classifier - ROC-AUC: 0.8375), XGBoost
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: mplsoccer, Matplotlib
- **Web Framework**: Streamlit
- **Data Source**: StatsBomb Open Data

## 📂 Project Structure

```text
├── 📓 01_extracao_dados_e_modelo_xg.ipynb   # xG Model Training Notebook
├── 📓 02_scouting_player_similarity.ipynb  # Cosine Similarity Notebook
├── 📦 xg_model.pkl                         # Trained ML Model
├── 📊 player_stats.csv                      # Processed Player Dataset
├── 🛠️ helpers.py                            # Custom Calculation Functions
├── 🚀 app.py                                # Streamlit Web UI
└── 📄 requirements.txt                     # Dependencies