from flask import Flask, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.utils
import json
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from flask.wrappers import Response

# Application Configuration
APP_HOST = "0.0.0.0"  # Allow external connections in container
APP_PORT = 5505
DEBUG_MODE = False

# F1 Points System (1st to 10th place)
F1_POINTS_SYSTEM = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 10

# Monte Carlo Simulation Settings
MONTE_CARLO_SIMULATIONS = 10000
POSITION_WEIGHT_EXPONENT = 1.5
RANDOMNESS_STANDARD_DEVIATION = 0.3
MINIMUM_WEIGHT = 0.01

# Update Schedule Settings
RACE_WEEKEND_CHECK_MINUTES = 10
REGULAR_UPDATE_HOURS = 6
POST_RACE_DATA_DELAY_HOURS = 4

# Session Duration Map (in hours)
SESSION_DURATIONS = {
    "Practice 1": 1.5,
    "Practice 2": 1.5,
    "Practice 3": 1,
    "Sprint Qualifying": 1,
    "Sprint": 1,
    "Qualifying": 1,
    "Race": 2,
}

# Chart Configuration
TOP_DRIVERS_DISPLAY_COUNT = 10
CHART_LINE_WIDTH = 1.5
CHART_MARKER_SIZE = 6

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Setup FastF1 cache
cache_dir = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Initialize background scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Global data storage
seasons_data: Dict[int, Dict[str, Any]] = {}
current_year = datetime.now().year

# Initialize championship data for current year
championship_data: Dict[str, Any] = {
    "last_update": None,
    "driver_standings": [],
    "championship_odds": {},
    "race_positions": {},
    "elimination_races": {},
    "races_completed": 0,
    "total_races": 0,
    "year": current_year,
}


def get_next_race_date() -> Optional[pd.Timestamp]:
    """Get the date of the next race from the F1 schedule"""
    try:
        current_year = datetime.now().year
        schedule = fastf1.get_event_schedule(current_year)
        race_schedule = schedule[schedule["EventFormat"] != "testing"]
        now = pd.Timestamp.now(tz="UTC")

        for idx, event in race_schedule.iterrows():
            race_date = pd.Timestamp(event["Session5DateUtc"])
            if pd.isna(race_date):
                continue

            if race_date.tz is None:
                race_date = race_date.tz_localize("UTC")

            if race_date > now:
                return race_date + timedelta(hours=POST_RACE_DATA_DELAY_HOURS)

        return None
    except Exception as e:
        logger.error(f"Error getting next race date: {e}")
        return None


def get_current_f1_sessions() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Get all F1 sessions happening now or within the next 24 hours"""
    try:
        current_year = datetime.now().year
        schedule = fastf1.get_event_schedule(current_year)
        now = pd.Timestamp.now(tz="UTC")
        active_sessions = []
        upcoming_sessions = []

        for idx, event in schedule.iterrows():
            if event["EventFormat"] == "testing":
                continue

            for session_num in range(1, 6):
                session_key = f"Session{session_num}DateUtc"
                session_name_key = f"Session{session_num}"

                if session_key not in event or pd.isna(event[session_key]):
                    continue

                session_time = pd.Timestamp(event[session_key])
                if session_time.tz is None:
                    session_time = session_time.tz_localize("UTC")

                session_name = event[session_name_key]
                duration = SESSION_DURATIONS.get(session_name, 1.5)
                session_end = session_time + timedelta(hours=duration)

                if session_time <= now <= session_end:
                    active_sessions.append(
                        {
                            "event_name": event["EventName"],
                            "session_name": session_name,
                            "start_time": session_time,
                            "end_time": session_end,
                            "round": event["RoundNumber"],
                        }
                    )
                elif session_time > now and session_time <= now + timedelta(hours=24):
                    upcoming_sessions.append(
                        {
                            "event_name": event["EventName"],
                            "session_name": session_name,
                            "start_time": session_time,
                            "end_time": session_end,
                            "round": event["RoundNumber"],
                        }
                    )

        return active_sessions, upcoming_sessions
    except Exception as e:
        logger.error(f"Error getting F1 sessions: {e}")
        return [], []


def is_f1_session_active() -> bool:
    """Check if any F1 session is currently active"""
    active_sessions, _ = get_current_f1_sessions()
    return len(active_sessions) > 0


def should_update_data() -> bool:
    """Check if we should update data based on races completed"""
    global championship_data

    try:
        current_year = datetime.now().year
        # Force refresh of schedule by using force_ergast parameter
        schedule = fastf1.get_event_schedule(current_year, backend='ergast')
        now = pd.Timestamp.now(tz="UTC")
        races_completed = 0

        for idx, event in schedule.iterrows():
            if event["EventFormat"] == "testing":
                continue

            race_date = pd.Timestamp(event["Session5DateUtc"])
            if pd.isna(race_date):
                continue

            if race_date.tz is None:
                race_date = race_date.tz_localize("UTC")

            # Count race as complete if it happened (matches fetch_f1_data logic)
            if race_date < now:
                races_completed += 1

        logger.info(f"Race count check: stored={championship_data['races_completed']}, actual={races_completed}")

        if championship_data["races_completed"] < races_completed:
            logger.info(
                f"New race detected! {races_completed} races complete (was {championship_data['races_completed']})"
            )
            return True

        return False
    except Exception as e:
        logger.error(f"Error checking for updates: {e}")
        return True


def fetch_f1_data(year=None):
    """Fetch F1 season data from FastF1 API for a specific year"""
    global championship_data, seasons_data

    if year is None:
        year = datetime.now().year

    # For current year, check if we need to update
    if year == datetime.now().year:
        if (
            championship_data["last_update"] is not None
            and championship_data.get("year") == year
        ):
            if not should_update_data():
                logger.info("No new races detected, using cached data")
                return championship_data

    if year < datetime.now().year and year in seasons_data:
        if seasons_data[year].get("season_complete", False):
            logger.info(f"Using cached complete data for {year} season")
            return seasons_data[year]

    try:
        logger.info(f"Fetching F1 data for {year}...")
        schedule = fastf1.get_event_schedule(year)
        total_races = len(schedule[schedule["EventFormat"] != "testing"])
        driver_standings = {}
        race_positions = {}
        races_completed = 0

        for idx, event in schedule.iterrows():
            if event["EventFormat"] == "testing":
                continue

            race_name = event["EventName"]
            race_date = pd.Timestamp(event["Session5DateUtc"])

            if pd.isna(race_date):
                continue

            if race_date.tz is None:
                race_date = race_date.tz_localize("UTC")

            if race_date > pd.Timestamp.now(tz="UTC"):
                continue

            try:
                session = fastf1.get_session(year, event["RoundNumber"], "R")
                session.load(laps=False, telemetry=False, weather=False, messages=False)
                results = session.results

                if results.empty:
                    continue

                races_completed += 1
                race_positions[race_name] = {}

                for _, driver in results.iterrows():
                    driver_abbr = driver["Abbreviation"]
                    position = driver["Position"]
                    points = driver["Points"]

                    if pd.notna(driver_abbr) and pd.notna(position):
                        if driver_abbr not in driver_standings:
                            driver_standings[driver_abbr] = {
                                "name": driver["FullName"],
                                "team": driver["TeamName"],
                                "points": 0,
                                "positions": [],
                            }

                        driver_standings[driver_abbr]["points"] += (
                            points if pd.notna(points) else 0
                        )
                        driver_standings[driver_abbr]["positions"].append(int(position))
                        race_positions[race_name][driver_abbr] = int(position)
            except Exception as e:
                logger.warning(f"Could not load data for {race_name}: {e}")
                continue

        standings_list = [
            {
                "abbreviation": abbr,
                "name": data["name"],
                "team": data["team"],
                "points": data["points"],
                "average_position": (
                    np.mean(data["positions"]) if data["positions"] else 20
                ),
            }
            for abbr, data in driver_standings.items()
        ]
        standings_list.sort(key=lambda x: x["points"], reverse=True)

        odds = (
            calculate_championship_odds(standings_list, races_completed, total_races)
            if year == datetime.now().year
            else {}
        )
        elimination_races = calculate_elimination_races(
            standings_list, race_positions, races_completed, total_races
        )

        for driver in standings_list:
            driver["elimination_race"] = elimination_races.get(driver["abbreviation"])

        season_data = {
            "last_update": datetime.now().isoformat(),
            "driver_standings": standings_list,
            "championship_odds": odds,
            "race_positions": race_positions,
            "elimination_races": elimination_races,
            "races_completed": races_completed,
            "total_races": total_races,
            "year": year,
            "season_complete": races_completed == total_races,
        }

        if year == datetime.now().year:
            championship_data = season_data

        seasons_data[year] = season_data
        logger.info(
            f"Data updated successfully for {year}. {races_completed}/{total_races} races completed"
        )
        return season_data
    except Exception as e:
        logger.error(f"Error fetching F1 data: {e}")


def calculate_elimination_races(
    standings: List[Dict[str, Any]],
    race_positions: Dict[str, Dict[str, int]],
    races_completed: int,
    total_races: int,
) -> Dict[str, Optional[int]]:
    """Calculate at which race each driver was mathematically eliminated from championship contention"""
    if not standings or not race_positions:
        return {}

    elimination_races: Dict[str, Optional[int]] = {
        driver["abbreviation"]: None for driver in standings
    }
    race_names = list(race_positions.keys())

    for race_idx, race_name in enumerate(race_names):
        remaining_races_after_this = total_races - (race_idx + 1)
        max_possible_remaining_points = remaining_races_after_this * 25

        race_standings = {}
        for driver in standings:
            driver_abbr = driver["abbreviation"]
            points_so_far = 0

            for completed_race in race_names[: race_idx + 1]:
                if driver_abbr in race_positions[completed_race]:
                    position = race_positions[completed_race][driver_abbr]
                    if position <= 10:
                        points_so_far += F1_POINTS_SYSTEM[position - 1]

            race_standings[driver_abbr] = points_so_far

        if race_standings:
            leader_points = max(race_standings.values())

            for driver_abbr, points in race_standings.items():
                if elimination_races[driver_abbr] is not None:
                    continue

                max_possible_points = points + max_possible_remaining_points
                if max_possible_points < leader_points:
                    elimination_races[driver_abbr] = race_idx + 1

    return elimination_races


def calculate_championship_odds(
    standings: List[Dict[str, Any]], races_completed: int, total_races: int
) -> Dict[str, float]:
    """Calculate championship winning probability for each driver using Monte Carlo simulation"""
    if not standings or races_completed == 0:
        return {}

    remaining_races = total_races - races_completed
    return (
        _calculate_final_season_odds(standings)
        if remaining_races == 0
        else _run_monte_carlo_simulation(standings, remaining_races)
    )


def _calculate_final_season_odds(standings: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate odds when season is complete"""
    winner_points = standings[0]["points"]
    return {
        driver["abbreviation"]: 100.0 if driver["points"] == winner_points else 0.0
        for driver in standings
    }


def _run_monte_carlo_simulation(
    standings: List[Dict[str, Any]], remaining_races: int
) -> Dict[str, float]:
    """Run Monte Carlo simulation to predict championship odds"""
    wins = {driver["abbreviation"]: 0 for driver in standings}

    for _ in range(MONTE_CARLO_SIMULATIONS):
        sim_points = {driver["abbreviation"]: driver["points"] for driver in standings}

        for _ in range(remaining_races):
            race_result = _simulate_single_race(standings)
            _award_race_points(sim_points, race_result)

        winner = max(sim_points.items(), key=lambda x: x[1])[0]
        wins[winner] += 1

    return {
        driver: (count / MONTE_CARLO_SIMULATIONS) * 100
        for driver, count in wins.items()
    }


def _simulate_single_race(standings: List[Dict[str, Any]]) -> List[str]:
    """Simulate a single race and return driver finishing order"""
    drivers = [driver["abbreviation"] for driver in standings]
    weights = _calculate_driver_weights(standings)
    weights = weights + np.random.normal(0, RANDOMNESS_STANDARD_DEVIATION, len(weights))
    weights = np.maximum(weights, MINIMUM_WEIGHT)
    weights = weights / weights.sum()
    return np.random.choice(
        drivers, size=len(drivers), replace=False, p=weights
    ).tolist()


def _calculate_driver_weights(standings: List[Dict[str, Any]]) -> np.ndarray:
    """Calculate driver performance weights for race simulation"""
    leader_points = max(1, standings[0]["points"])
    weights = [
        (1.0 / (driver["average_position"] ** POSITION_WEIGHT_EXPONENT))
        * ((driver["points"] / leader_points) ** 0.5)
        for driver in standings
    ]
    return np.array(weights)


def _award_race_points(sim_points: Dict[str, int], race_result: List[str]) -> None:
    """Award points to drivers based on race finishing positions"""
    for position, driver_abbr in enumerate(race_result[:10]):
        sim_points[driver_abbr] += F1_POINTS_SYSTEM[position]


@app.route("/")
def index() -> str:
    """Serve the main dashboard HTML page with championship odds and driver progression"""
    return render_template("index.html")


@app.route("/api/data")
@app.route("/api/data/<int:year>")
def get_data(year: Optional[int] = None) -> Union[Response, Tuple[Response, int]]:
    """API endpoint for fetching championship data for a specific year"""
    global seasons_data

    if year is None:
        year = datetime.now().year

    if year not in seasons_data:
        season_data = fetch_f1_data(year)
        if season_data is None:
            return jsonify({"error": f"No data available for {year}"}), 404

    data = (
        championship_data.copy()
        if year == datetime.now().year
        else seasons_data.get(year, {}).copy()
    )

    if year == datetime.now().year:
        next_race = get_next_race_date()
        data["next_race_date"] = next_race.isoformat() if next_race else None
        active_sessions, upcoming_sessions = get_current_f1_sessions()
        data["active_sessions"] = active_sessions
        data["upcoming_sessions"] = upcoming_sessions[:3]
        data["is_session_active"] = len(active_sessions) > 0

    data["available_years"] = list(range(2020, datetime.now().year + 1))
    return jsonify(data)


@app.route("/api/charts")
@app.route("/api/charts/<int:year>")
def get_charts(year: Optional[int] = None) -> Response:
    """Generate Plotly charts for data visualization"""
    global seasons_data

    if year is None:
        year = datetime.now().year

    data = (
        championship_data if year == datetime.now().year else seasons_data.get(year, {})
    )

    if not data or not data.get("driver_standings"):
        return jsonify({"odds_chart": None, "progression_chart": None})

    standings = data["driver_standings"]
    odds = data.get("championship_odds", {})
    odds_chart = None

    if year == datetime.now().year and odds:
        top_drivers = standings[:TOP_DRIVERS_DISPLAY_COUNT]
        drivers = [d["abbreviation"] for d in top_drivers]
        probabilities = [odds.get(d, 0) for d in drivers]

        odds_chart = go.Figure(
            data=[
                go.Bar(
                    x=drivers,
                    y=probabilities,
                    text=[f"{p:.1f}%" for p in probabilities],
                    textposition="auto",
                    marker_color=[
                        (
                            "#1f77b4"
                            if i == 0
                            else (
                                "#ff7f0e"
                                if i == 1
                                else "#2ca02c" if i == 2 else "#d62728"
                            )
                        )
                        for i in range(len(drivers))
                    ],
                    hovertemplate="<b>%{x}</b><br>Win Probability: %{y:.1f}%<extra></extra>",
                )
            ]
        )

        odds_chart.update_layout(
            title=f"{year} F1 Championship Win Probability",
            xaxis_title="Driver",
            yaxis_title="Win Probability (%)",
            template="plotly_dark",
            height=500,
            showlegend=False,
        )

    # Points chart for historical seasons
    elif year < datetime.now().year:
        top_drivers = standings[:TOP_DRIVERS_DISPLAY_COUNT]
        drivers = [d["abbreviation"] for d in top_drivers]
        points = [d["points"] for d in top_drivers]

        odds_chart = go.Figure(
            data=[
                go.Bar(
                    x=drivers,
                    y=points,
                    text=[f"{p} pts" for p in points],
                    textposition="auto",
                    marker_color=[
                        (
                            "#FFD700"
                            if i == 0
                            else (
                                "#C0C0C0"
                                if i == 1
                                else "#CD7F32" if i == 2 else "#4ECDC4"
                            )
                        )
                        for i in range(len(drivers))
                    ],
                    hovertemplate="<b>%{x}</b><br>Points: %{y}<extra></extra>",
                )
            ]
        )

        odds_chart.update_layout(
            title=f"{year} F1 Championship Final Standings",
            xaxis_title="Driver",
            yaxis_title="Points",
            template="plotly_dark",
            height=500,
            showlegend=False,
        )

    # Driver Position Progression Line Chart
    race_positions = data.get("race_positions", {})

    if race_positions:
        races = list(race_positions.keys())

        # Define color scheme for better distinction
        colors = [
            "#FFD700",  # Gold for 1st
            "#C0C0C0",  # Silver for 2nd
            "#CD7F32",  # Bronze for 3rd
            "#FF6B6B",  # Red for 4th
            "#4ECDC4",  # Teal for 5th
            "#95E77E",  # Green
            "#FFA07A",  # Light salmon
            "#87CEEB",  # Sky blue
            "#DDA0DD",  # Plum
            "#F0E68C",  # Khaki
        ]

        # Create traces for top drivers (reduced to 5 by default for clarity)
        traces = []
        for idx, driver in enumerate(standings[:5]):  # Reduced from 10 to 5
            driver_abbr = driver["abbreviation"]
            positions = []

            for race in races:
                if driver_abbr in race_positions[race]:
                    positions.append(race_positions[race][driver_abbr])
                else:
                    positions.append(None)

            if any(p is not None for p in positions):
                # Use chart configuration constants

                traces.append(
                    go.Scatter(
                        x=races,
                        y=positions,
                        mode="lines+markers",
                        name=f"{driver_abbr} (P{idx+1})",
                        line=dict(
                            width=CHART_LINE_WIDTH,
                            color=colors[idx],
                            dash="dot",  # Dotted lines for all
                        ),
                        marker=dict(
                            size=CHART_MARKER_SIZE,
                            color=colors[idx],
                            line=dict(color="white", width=0.5),
                        ),
                        hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Position: P%{y}<extra></extra>",
                        visible=True,
                    )
                )

        # Add remaining drivers (6-10) as hidden traces that can be toggled
        for idx, driver in enumerate(standings[5:10], start=5):
            driver_abbr = driver["abbreviation"]
            positions = []

            for race in races:
                if driver_abbr in race_positions[race]:
                    positions.append(race_positions[race][driver_abbr])
                else:
                    positions.append(None)

            if any(p is not None for p in positions):
                traces.append(
                    go.Scatter(
                        x=races,
                        y=positions,
                        mode="lines+markers",
                        name=f"{driver_abbr} (P{idx+1})",
                        line=dict(
                            width=1.5,
                            color=colors[idx],
                            dash="dot",  # Dotted line for lower positions
                        ),
                        marker=dict(
                            size=5,
                            color=colors[idx],
                            line=dict(color="white", width=0.5),
                        ),
                        hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Position: P%{y}<extra></extra>",
                        visible="legendonly",  # Hidden by default, can be toggled via legend
                    )
                )

        progression_chart = go.Figure(data=traces)

        progression_chart.update_layout(
            title=f"{year} Driver Championship Position by Race",
            xaxis_title="Race",
            yaxis_title="Position",
            yaxis=dict(
                autorange="reversed",
                dtick=1,
                range=[0.5, 20.5],
                gridcolor="rgba(255,255,255,0.1)",
            ),
            xaxis=dict(tickangle=45, gridcolor="rgba(255,255,255,0.1)"),
            template="plotly_dark",
            height=600,
            hovermode="x unified",
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.3)",
                borderwidth=1,
            ),
            margin=dict(r=150),  # More space for legend
        )
    else:
        progression_chart = None

    return jsonify(
        {
            "odds_chart": (
                json.dumps(odds_chart, cls=plotly.utils.PlotlyJSONEncoder)
                if odds_chart
                else None
            ),
            "progression_chart": (
                json.dumps(progression_chart, cls=plotly.utils.PlotlyJSONEncoder)
                if progression_chart
                else None
            ),
        }
    )


def smart_update_check():
    """Smart update logic - only fetch during F1 sessions or after races"""
    try:
        active_sessions, upcoming_sessions = get_current_f1_sessions()

        if active_sessions:
            logger.info(
                f"F1 session active: {active_sessions[0]['session_name']} at {active_sessions[0]['event_name']}"
            )
            fetch_f1_data()
        elif upcoming_sessions:
            next_session = min(upcoming_sessions, key=lambda s: s["start_time"])
            time_until_session = next_session["start_time"] - pd.Timestamp.now(tz="UTC")

            if time_until_session.total_seconds() < 3600:
                logger.info(
                    f"F1 session starting soon: {next_session['session_name']} in {time_until_session}"
                )
                fetch_f1_data()
            else:
                logger.info(
                    f"No immediate F1 activity. Next: {next_session['session_name']} in {time_until_session}"
                )
        else:
            if should_update_data():
                logger.info("Checking for completed races...")
                fetch_f1_data()
            else:
                logger.info("No F1 sessions active and no new races to process")
    except Exception as e:
        logger.error(f"Error in smart update check: {e}")
        fetch_f1_data()


try:
    scheduler.remove_job("fetch_f1_data")
except:
    pass

scheduler.add_job(
    func=smart_update_check,
    trigger="interval",
    minutes=RACE_WEEKEND_CHECK_MINUTES,
    id="smart_f1_update",
    name="Smart F1 data updates",
    replace_existing=True,
)


def preload_historical_data():
    """Pre-load historical season data for better performance"""
    for year in [2020, 2021, 2022, 2023, 2024]:
        if year != datetime.now().year:
            try:
                logger.info(f"Pre-loading historical data for {year}...")
                fetch_f1_data(year)
            except Exception as e:
                logger.warning(f"Could not pre-load data for {year}: {e}")


scheduler.add_job(
    func=preload_historical_data,
    trigger="interval",
    hours=24,
    id="preload_historical_data",
    name="Pre-load historical F1 data",
    replace_existing=True,
)

# Temporarily disabled to allow Flask to start immediately
# fetch_f1_data()
# preload_historical_data()

if __name__ == "__main__":
    app.run(debug=DEBUG_MODE, host=APP_HOST, port=APP_PORT)
