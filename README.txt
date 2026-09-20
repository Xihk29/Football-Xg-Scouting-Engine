# ⚽ Expected Goals (xG) Model & Player Scouting Engine

An end-to-end Football Analytics and Player Scouting web application built with **Python**, **Streamlit**, **Scikit-Learn**, and **mplsoccer**, powered by open event data from **StatsBomb** (FIFA World Cup 2022).

---

## 🖼️ Application Showcase

### 1. 📊 Match Scoreboard Comparison (Actual Goals vs xG)
![Match Scoreboard](assets/scoreboard.png)

### 2. ⚽ Inverted Pitch Match Shot Map
![Match Shot Map](assets/shot_map.png)

### 3. 🔮 Real-Time xG Simulator & Interactive Pitch Position
![xG Simulator](assets/simulator.png)

### 4. 🔍 Player Scouting & Visual Radar Comparison Engine
![Player Scouting Engine](assets/radar_chart.png)

---

## 🚀 Key Features

* **📊 Match Scoreboard & Shot Map**: Compares actual match goals with team total Expected Goals (xG) and plots inverted team shot maps (Home vs Away) powered by `mplsoccer`.
* **🔮 Real-Time xG Calculator**: Interactive simulator predicting goal probability based on shot distance, angle, header, and play type with dynamic mini-pitch visualization.
* **🔍 Player Scouting & Cosine Similarity Engine**: Machine learning similarity model finding top player profiles using **Cosine Similarity** and interactive **Radar Comparison Charts**.

---

## 🛠️ Tech Stack & Libraries

* **Language**: Python 3.10+
* **Machine Learning**: Scikit-Learn (Random Forest Classifier - ROC-AUC: 0.8375), XGBoost
* **Data Manipulation**: Pandas, NumPy
* **Visualizations**: mplsoccer, Matplotlib
* **Web Framework**: Streamlit
* **Data Source**: StatsBomb Open Data

---

## 📂 Project Structure

├── 📓 01_extracao_dados_e_modelo_xg.ipynb   # xG Model Training Notebook
├── 📓 02_scouting_player_similarity.ipynb  # Cosine Similarity Notebook
├── 📦 xg_model.pkl                         # Trained ML Model
├── 📊 player_stats.csv                      # Processed Player Dataset
├── 🛠️ helpers.py                            # Custom Calculation Functions
├── 🚀 app.py                                # Streamlit Web UI
├── 📄 requirements.txt                     # Cloud Dependencies
└── 📁 assets/                              # Screenshots for Documentation
    ├── scoreboard.png
    ├── shot_map.png
    ├── simulator.png
    └── radar_chart.png
