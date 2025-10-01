# F1 Driver Championship Odds

This web application provides live and historical data on the Formula 1 Driver's Championship. It uses a Monte Carlo simulation to calculate the probability of each driver winning the championship and visualizes the data in an easy-to-understand dashboard.

## Features

- **Championship Odds:** Calculates and displays the percentage chance for each driver to win the current season's championship using a Monte Carlo simulation.
- **Live Standings:** Shows the current driver standings with points.
- **Driver Progression:** A line chart that visualizes each driver's race-by-race finishing positions throughout the season.
- **Historical Data:** View standings and race progression for past F1 seasons (from 2020 onwards).
- **Smart Updates:** The application automatically updates data more frequently during live race sessions.
- **Responsive Dashboard:** A web interface built with Plotly.js for interactive charts and a clean design.

## Technologies Used

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Data Analysis & Simulation:** pandas, numpy
- **F1 Data Source:** fastf1
- **Charting:** Plotly.js
- **Task Scheduling:** APScheduler
- **Deployment:** Docker, Gunicorn
