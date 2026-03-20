import streamlit as st
import pandas as pd

st.set_page_config(page_title="March Madness 2026 Predictor", page_icon="\U0001f3c0", layout="centered")


@st.cache_data
def load_data():
    teams = pd.read_csv("data/teams.csv")
    preds = pd.read_csv("data/predictions.csv")
    return teams, preds


teams_df, preds_df = load_data()

st.title("\U0001f3c0 March Madness 2026 Predictor")
st.caption(
    "LightGBM + CatBoost ensemble \u00b7 CV Brier 0.1619 \u00b7 "
    "[GitHub](https://github.com/hr-27-678/march-madness-2026)"
)

# Gender selection
gender = st.radio("Tournament", ["Men", "Women"], horizontal=True)
gender_prefix = "M" if gender == "Men" else "W"

# Filter teams by gender
gender_teams = teams_df[teams_df["Gender"] == gender_prefix].sort_values("TeamName")
team_names = gender_teams["TeamName"].tolist()
name_to_id = dict(zip(gender_teams["TeamName"], gender_teams["TeamID"]))

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Team A", team_names, index=None, placeholder="Select a team...")
with col2:
    team_b = st.selectbox("Team B", team_names, index=None, placeholder="Select a team...")

if team_a and team_b:
    if team_a == team_b:
        st.warning("Please select two different teams.")
    else:
        id_a = name_to_id[team_a]
        id_b = name_to_id[team_b]

        # Predictions are stored as lower_id vs higher_id
        low_id, high_id = min(id_a, id_b), max(id_a, id_b)
        game_id = f"2026_{low_id}_{high_id}"

        row = preds_df[preds_df["ID"] == game_id]

        if row.empty:
            st.error(f"No prediction found for {team_a} vs {team_b}.")
        else:
            # Pred = P(lower_id wins)
            p_low_wins = row["Pred"].values[0]

            if id_a == low_id:
                p_a = p_low_wins
            else:
                p_a = 1.0 - p_low_wins

            p_b = 1.0 - p_a

            st.divider()

            # Display results
            left, right = st.columns(2)
            with left:
                st.metric(team_a, f"{p_a:.1%}")
            with right:
                st.metric(team_b, f"{p_b:.1%}")

            # Progress bar as visual
            st.progress(p_a)

            # Color-coded verdict
            if p_a > 0.5:
                winner, prob = team_a, p_a
            else:
                winner, prob = team_b, p_b

            if prob > 0.75:
                confidence = "High confidence"
            elif prob > 0.6:
                confidence = "Moderate confidence"
            else:
                confidence = "Close matchup"

            st.success(f"**{winner}** favored \u00b7 {confidence} ({prob:.1%})")

st.divider()
st.caption(
    "Model: 0.5 \u00d7 LightGBM (3 seeds) + 0.5 \u00d7 CatBoost \u00b7 "
    "19 features \u00b7 Temperature scaling T=0.90 \u00b7 "
    "Trained on 2003\u20132025 NCAA tournament data"
)
