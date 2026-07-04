# Traffic Carbon Emission Dashboard

A data science dashboard that analyses traffic conditions and estimates carbon emissions across selected towns and areas within the state of Melaka using processed Waze traffic data.

## Project Overview

This project studies the relationship between traffic congestion, vehicle speed, road delay, traffic intensity, and estimated CO₂ emissions.

The dashboard helps users understand:

- Where traffic congestion is highest
- When peak traffic occurs
- Which areas produce the highest estimated CO₂ load
- Which locations have High or Critical environmental risk
- How different areas in Melaka compare

The locations included in the dataset are:

- Melaka City
- Ayer Keroh
- Alor Gajah
- Bemban
- Durian Tunggal
- Merlimau
- Sungai Udang
- Tanjong Kling

## Main Features

- Traffic overview and KPI cards
- Traffic congestion analysis
- Average speed and delay analysis
- Traffic hotspot identification
- Estimated CO₂ emission analysis
- Environmental risk classification
- Interactive emission heatmaps
- Area comparison and ranking
- Environmental recommendations
- Downloadable traffic and emission reports
- Filters by area, congestion level, risk level, date, and time

## Dashboard Pages

### Overview

Displays overall traffic congestion, estimated CO₂ emissions, traffic records, environmental risk, and the highest-emission area.

### Traffic Analysis

Analyses traffic patterns, average speed, congestion levels, peak traffic periods, and traffic hotspots.

### Emission Prediction

Displays estimated CO₂ load and environmental risk based on traffic conditions such as speed, delay, congestion, and road length.

### Environmental Impact

Identifies pollution hotspots, environmental risk levels, emission trends, and suggested actions.

### Comparison

Compares selected areas in Melaka based on traffic records, average speed, congestion, estimated CO₂ emissions, and environmental risk.

### Reports

Generates summaries, findings, recommendations, and downloadable report data.

## Technologies Used

- Python
- Streamlit
- Pandas
- Plotly
- Folium
- Streamlit Folium
- Waze Traffic Data

## Dataset

The project uses processed Waze traffic data containing features such as:

- Area and road name
- Traffic speed
- Traffic delay
- Congestion level
- Road length
- Road type
- Latitude and longitude
- Date and time
- Traffic intensity
- Estimated CO₂ load
- Environmental risk level



## Dataset Access

The dataset used in this project is not included in this repository because it is associated with a government-related project and is not intended for public distribution.

The repository contains the dashboard source code only. To run the application, an authorized user must place the required processed dataset in the same folder as `app.py`.


## Project Files

```text
TRAFFIC-CARBON-EMISSION/
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
