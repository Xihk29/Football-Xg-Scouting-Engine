# app.py: Streamlit User Interface layout

import pandas as pd
import streamlit as st

# Import all custom helper functions from helpers.py
from helpers import *

# Page setup
st.set_page_config(
    page_title="Football Analytics & Scouting", page_icon="⚽", layout="wide"
)
st.title("⚽ Football Analytics & Scouting Engine")

# Sidebar Setup
with st.sidebar:
  st.header("📌 Project Details")
  st.write("**ML Model:** Random Forest Classifier")
  st.write("**Model Score (ROC-AUC):** 0.8375")
  st.write("**Data Provider:** StatsBomb Open Data")
  st.write("**Tournament:** FIFA World Cup 2022")
  st.divider()
  st.caption("Built with Python, Streamlit, Scikit-Learn & mplsoccer")

# Load artifacts
xg_model = load_xg_model()
player_stats = load_player_stats()
matches = load_matches()

# Tabs Navigation
tab1, tab2, tab3 = st.tabs(
    ["📊 Match Shot Map", "🔮 Real-Time xG Calculator", "🔍 Player Scouting Engine"]
)

# =============================================================================
# TAB 1: Match Shot Map & Expected Goals Scoreboard
# =============================================================================
with tab1:
  st.header("Match Shot Map & Scoreboard Comparison")
  matches["match_label"] = (
      matches["home_team"]
      + " vs "
      + matches["away_team"]
      + " ("
      + matches["match_date"]
      + ")"
  )
  selected_match_label = st.selectbox(
      "Select World Cup 2022 Match:", matches["match_label"]
  )

  match_row = matches[matches["match_label"] == selected_match_label].iloc[0]

  if st.button("Generate Match Analysis"):
    with st.spinner("Analyzing match events and team xG..."):
      analysis_res = generate_match_shot_map(match_row["match_id"], xg_model)

      # Protection check: Verify shot data exists before unpacking tuple
      if analysis_res and analysis_res[0] is not None:
        fig, h_team, h_goals, h_xg, a_team, a_goals, a_xg = analysis_res

        # Display Scoreboard vs xG Comparison
        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
          st.subheader(f"🔵 {h_team}")
          st.metric("Actual Goals", h_goals)
          st.metric("Total xG Created", f"{h_xg:.2f}")

        with col2:
          st.subheader("VS")

        with col3:
          st.subheader(f"🔴 {a_team}")
          st.metric("Actual Goals", a_goals)
          st.metric("Total xG Created", f"{a_xg:.2f}")

        # Render Pitch Shot Map
        st.pyplot(fig)
      else:
        st.warning(
            "No shot event data recorded for the selected match in StatsBomb."
        )

# =============================================================================
# TAB 2: Real-Time xG Calculator & Field Position
# =============================================================================
with tab2:
  st.header("Real-Time xG Calculator & Field Position")
  st.write(
      "Adjust the shot parameters below to see the position on the pitch and"
      " calculate xG:"
  )

  col_a, col_b = st.columns([1, 1.2])

  with col_a:
    sim_dist = st.slider(
        "Distance to Goal Center (meters)", 1.0, 40.0, 10.0, 0.5
    )
    sim_angle = st.slider("Shot Angle (degrees)", 1.0, 90.0, 34.0, 1.0)
    sim_header = st.checkbox("Header Shot?")
    sim_open_play = st.checkbox("Open Play Shot?", value=True)

    # Calculate xG prediction
    sim_input = pd.DataFrame({
        "distance": [sim_dist],
        "angle_degrees": [sim_angle],
        "is_header": [int(sim_header)],
        "is_open_play": [int(sim_open_play)],
    })
    sim_xg = xg_model.predict_proba(sim_input)[0][1]

    st.subheader("Calculated Probability:")
    st.metric(
        label="Expected Goal (xG)", value=f"{sim_xg:.3f} ({sim_xg*100:.1f}%)"
    )

  with col_b:
    st.subheader("Simulated Pitch Position:")
    # Generate mini pitch visualization
    fig_mini = generate_simulated_pitch(sim_dist, sim_angle, sim_xg)
    st.pyplot(fig_mini)

# =============================================================================
# TAB 3: Player Scouting & Visual Radar Comparison
# =============================================================================
with tab3:
  st.header("Player Scouting & Visual Comparison Engine")

  selected_player = st.selectbox(
      "Select Target Player:", player_stats["player"].sort_values().unique()
  )
  top_n = st.slider("Number of similar players to return:", 3, 10, 5)

  if selected_player:
    results_df = get_similar_players(selected_player, player_stats, top_n)

    col_table, col_radar = st.columns([1.2, 1])

    with col_table:
      st.subheader(f"Top {top_n} Profiles Similar to '{selected_player}':")
      st.dataframe(results_df, use_container_width=True)

      # Interactive Selector to pick WHICH similar player to plot in the Radar Chart
      compared_player = st.selectbox(
          "Select a similar player from the list to compare on Radar:",
          options=results_df["Similar Player"].tolist(),
          index=0,
      )

    with col_radar:
      p1_label = selected_player.split()[0]
      p2_label = compared_player.split()[0]
      st.subheader(f"Radar Comparison: {p1_label} vs {p2_label}")

      # Generate Radar Chart for the interactively chosen compared player
      fig_radar = generate_radar_chart(
          selected_player, compared_player, player_stats
      )
      st.pyplot(fig_radar)