# helpers.py: Helper functions for xG predictions, pitch plotting, radar charts, and scouting similarity

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from statsbombpy import sb

# =============================================================================
# 1. DATA & MODEL LOADERS
# =============================================================================


# -----------------------------------------------------------------------------
# Load Trained Machine Learning xG Model
# -----------------------------------------------------------------------------
def load_xg_model(model_path="xg_model.pkl"):
  with open(model_path, "rb") as f:
    return pickle.load(f)


# -----------------------------------------------------------------------------
# Load Player Performance Statistics CSV
# -----------------------------------------------------------------------------
def load_player_stats(csv_path="player_stats.csv"):
  return pd.read_csv(csv_path)


# -----------------------------------------------------------------------------
# Fetch World Cup 2022 Matches List
# -----------------------------------------------------------------------------
def load_matches(competition_id=43, season_id=106):
  return sb.matches(competition_id=competition_id, season_id=season_id)


# =============================================================================
# 2. MATCH SHOT MAP VISUALIZATION
# =============================================================================


# -----------------------------------------------------------------------------
# Generate Inverted Pitch Shot Map & Team xG Comparison
# -----------------------------------------------------------------------------
def generate_match_shot_map(match_id, xg_model):
  events = sb.events(match_id=match_id)
  shots = events[events["type"] == "Shot"].copy()

  if len(shots) == 0:
    return None, None, None, None, None, None, None

  # Identify home and away teams in match
  teams = shots["team"].unique()
  home_team = teams[0]
  away_team = teams[1] if len(teams) > 1 else teams[0]

  # Calculate features for xG model
  shots["x"] = shots["location"].apply(
      lambda loc: loc[0] if isinstance(loc, list) else np.nan
  )
  shots["y"] = shots["location"].apply(
      lambda loc: loc[1] if isinstance(loc, list) else np.nan
  )
  shots["distance"] = np.sqrt((120 - shots["x"]) ** 2 + (40 - shots["y"]) ** 2)

  dx = 120 - shots["x"]
  dy1 = 36 - shots["y"]
  dy2 = 44 - shots["y"]
  shots["angle_degrees"] = np.degrees(
      np.abs(np.arctan2(dy2, dx) - np.arctan2(dy1, dx))
  )

  shots["is_header"] = (shots["shot_body_part"] == "Head").astype(int)
  shots["is_open_play"] = (shots["shot_type"] == "Open Play").astype(int)
  shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)

  # Predict xG using model
  X_shot = shots[["distance", "angle_degrees", "is_header", "is_open_play"]]
  shots["calculated_xg"] = xg_model.predict_proba(X_shot)[:, 1]

  # Split shots by team
  home_shots = shots[shots["team"] == home_team].copy()
  away_shots = shots[shots["team"] == away_team].copy()

  # Home Team attacks RIGHT (X=120)
  home_shots["x_plot"] = home_shots["x"]
  home_shots["y_plot"] = home_shots["y"]

  # Away Team attacks LEFT (Invert coordinates to X=0)
  away_shots["x_plot"] = 120 - away_shots["x"]
  away_shots["y_plot"] = 80 - away_shots["y"]

  # Draw Pitch
  pitch = Pitch(
      pitch_type="statsbomb", pitch_color="#22312b", line_color="#c7d5cc"
  )
  fig, ax = pitch.draw(figsize=(11, 7))

  # Plot Home Shots (Blue/Gold)
  home_goals = home_shots[home_shots["is_goal"] == 1]
  home_nongoals = home_shots[home_shots["is_goal"] == 0]
  pitch.scatter(
      home_nongoals["x_plot"],
      home_nongoals["y_plot"],
      s=home_nongoals["calculated_xg"] * 600 + 40,
      c="#3498db",
      alpha=0.7,
      edgecolors="white",
      ax=ax,
      label=f"{home_team} Shot",
  )
  pitch.scatter(
      home_goals["x_plot"],
      home_goals["y_plot"],
      s=home_goals["calculated_xg"] * 600 + 80,
      c="#f1c40f",
      marker="*",
      edgecolors="black",
      alpha=0.9,
      ax=ax,
      label=f"{home_team} Goal ★",
  )

  # Plot Away Shots (Red/Green)
  away_goals = away_shots[away_shots["is_goal"] == 1]
  away_nongoals = away_shots[away_shots["is_goal"] == 0]
  pitch.scatter(
      away_nongoals["x_plot"],
      away_nongoals["y_plot"],
      s=away_nongoals["calculated_xg"] * 600 + 40,
      c="#e74c3c",
      alpha=0.7,
      edgecolors="white",
      ax=ax,
      label=f"{away_team} Shot",
  )
  pitch.scatter(
      away_goals["x_plot"],
      away_goals["y_plot"],
      s=away_goals["calculated_xg"] * 600 + 80,
      c="#2ecc71",
      marker="*",
      edgecolors="black",
      alpha=0.9,
      ax=ax,
      label=f"{away_team} Goal ★",
  )

  plt.legend(
      loc="lower center",
      bbox_to_anchor=(0.5, -0.08),
      ncol=4,
      facecolor="#22312b",
      labelcolor="white",
  )
  fig.set_facecolor("#22312b")

  return (
      fig,
      home_team,
      len(home_goals),
      home_shots["calculated_xg"].sum(),
      away_team,
      len(away_goals),
      away_shots["calculated_xg"].sum(),
  )


# =============================================================================
# 3. INTERACTIVE PITCH SIMULATOR
# =============================================================================


# -----------------------------------------------------------------------------
# Generate Dynamic Mini Pitch for Real-Time Shot Simulator
# -----------------------------------------------------------------------------
def generate_simulated_pitch(distance, angle_degrees, xg_val):
  # Calculate X position (Goal line is at X=120; Penalty box line is at X=102 -> 18m)
  x_pos = 120 - distance

  # Calculate lateral Y offset (Multiplier 0.8 allows dot to travel outside the box towards touchlines)
  y_offset = (45 - angle_degrees) * 0.8
  y_pos = np.clip(40 + y_offset, 5, 75)

  # Draw half pitch using mplsoccer
  pitch = Pitch(
      pitch_type="statsbomb",
      pitch_color="#22312b",
      line_color="#c7d5cc",
      half=True,
  )
  fig, ax = pitch.draw(figsize=(6, 4))

  # Select dot color based on chance quality
  color = (
      "#f1c40f"
      if xg_val >= 0.20
      else ("#e67e22" if xg_val >= 0.08 else "#e74c3c")
  )

  marker_size = xg_val * 350 + 60

  # Scatter the simulated shot position
  pitch.scatter(
      x_pos,
      y_pos,
      s=marker_size,
      c=color,
      edgecolors="white",
      linewidth=2,
      ax=ax,
  )

  # Annotate position on pitch
  ax.text(
      x_pos,
      y_pos - 4,
      f"xG: {xg_val*100:.1f}%",
      color="white",
      fontsize=10,
      fontweight="bold",
      ha="center",
      bbox=dict(
          boxstyle="round,pad=0.2",
          facecolor="#22312b",
          edgecolor="white",
          alpha=0.8,
      ),
  )

  fig.set_facecolor("#22312b")
  return fig


# =============================================================================
# 4. SCOUTING ENGINE & RADAR CHARTS
# =============================================================================


# -----------------------------------------------------------------------------
# Query Top N Similar Players Using Cosine Similarity
# -----------------------------------------------------------------------------
def get_similar_players(selected_player, player_stats_df, top_n=5):
  feat_cols = [
      "total_shots",
      "total_goals",
      "total_passes",
      "completed_passes",
      "dribbles",
      "interceptions",
  ]

  features_df = player_stats_df.set_index("player")[feat_cols]
  scaler = StandardScaler()
  scaled_features = scaler.fit_transform(features_df)

  similarity_matrix = cosine_similarity(scaled_features)
  similarity_df = pd.DataFrame(
      similarity_matrix, index=features_df.index, columns=features_df.index
  )

  similar_scores = similarity_df[selected_player].sort_values(ascending=False)[
      1 : top_n + 1
  ]

  return pd.DataFrame({
      "Similar Player": similar_scores.index,
      "Similarity Match (%)": (similar_scores.values * 100).round(2),
  }).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Generate Polar Radar Comparison Chart Between Two Players
# -----------------------------------------------------------------------------
def generate_radar_chart(player1_name, player2_name, player_stats_df):
  feat_cols = [
      "total_shots",
      "total_goals",
      "total_passes",
      "completed_passes",
      "dribbles",
      "interceptions",
  ]
  labels = [
      "Shots",
      "Goals",
      "Passes",
      "Acc Passes",
      "Dribbles",
      "Interceptions",
  ]

  # Extract raw player stats
  p1_stats = (
      player_stats_df[player_stats_df["player"] == player1_name][feat_cols]
      .iloc[0]
      .values
  )
  p2_stats = (
      player_stats_df[player_stats_df["player"] == player2_name][feat_cols]
      .iloc[0]
      .values
  )

  # Normalize between 0 and 1 relative to maximum values in dataset
  max_vals = player_stats_df[feat_cols].max().values
  p1_norm = np.nan_to_num(p1_stats / max_vals, nan=0.0)
  p2_norm = np.nan_to_num(p2_stats / max_vals, nan=0.0)

  # Polar angles
  num_vars = len(labels)
  angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

  p1_norm = np.concatenate((p1_norm, [p1_norm[0]]))
  p2_norm = np.concatenate((p2_norm, [p2_norm[0]]))
  angles += angles[:1]

  # Build polar chart
  fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
  fig.patch.set_facecolor("#22312b")
  ax.set_facecolor("#22312b")

  # Plot Player 1
  p1_short_name = player1_name.split()[0]
  ax.plot(
      angles, p1_norm, color="#f1c40f", linewidth=2, label=p1_short_name
  )
  ax.fill(angles, p1_norm, color="#f1c40f", alpha=0.25)

  # Plot Player 2
  p2_short_name = player2_name.split()[0]
  ax.plot(
      angles, p2_norm, color="#3498db", linewidth=2, label=p2_short_name
  )
  ax.fill(angles, p2_norm, color="#3498db", alpha=0.25)

  ax.set_xticks(angles[:-1])
  ax.set_xticklabels(labels, color="white", size=10)
  ax.tick_params(colors="white")

  plt.legend(
      loc="upper right",
      bbox_to_anchor=(1.3, 1.1),
      facecolor="#22312b",
      labelcolor="white",
  )
  return fig