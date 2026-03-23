# SmartStock

SmartStock is an end-to-end demand forecasting and inventory intelligence project built for a data science portfolio. It is designed to look and feel like a compact business product instead of a one-off notebook. The project forecasts short-term retail demand, converts those forecasts into inventory recommendations, and exposes the results through an interactive Streamlit app.

The core business question is simple and valuable:

> How much inventory should we stock over the next week or month, where are we at risk of stockouts, and which items should we reorder first?

This repository is intentionally structured to demonstrate both data science skill and product thinking:

- time series feature engineering
- supervised forecasting
- baseline comparison
- business-facing evaluation
- inventory decision logic
- app-based storytelling
- test coverage for key logic

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Use Case](#use-case)
- [Who This Project Is For](#who-this-project-is-for)
- [What SmartStock Does](#what-smartstock-does)
- [Why This Is a Strong Portfolio Project](#why-this-is-a-strong-portfolio-project)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Modeling Strategy](#modeling-strategy)
- [Inventory Intelligence Layer](#inventory-intelligence-layer)
- [Evaluation Framework](#evaluation-framework)
- [Streamlit App](#streamlit-app)
- [How to Run the Project](#how-to-run-the-project)
- [Outputs and Artifacts](#outputs-and-artifacts)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [How to Extend This Project](#how-to-extend-this-project)
- [Portfolio Talking Points](#portfolio-talking-points)
- [Future Improvements](#future-improvements)

---

## Project Overview

SmartStock predicts future retail demand at the `store_id + item_id` level and translates those predictions into business actions.

Instead of stopping at a forecast chart, the project answers the operational question that matters after prediction:

- Do we have enough inventory?
- How much should we reorder?
- Which products are most at risk?
- Which items are likely overstocked?

The project currently supports:

- `7-day` cumulative demand forecasts
- `28-day` cumulative demand forecasts
- reorder recommendations
- safety stock estimation
- stockout risk scoring
- overstock vs healthy inventory classification
- interactive exploration through Streamlit

The included demo data is synthetic but structured to resemble a real public retail demand dataset such as an M5-style panel. That decision makes the project runnable out of the box while preserving the workflow you would use on a real retail forecasting problem.

---

## Business Problem

Retail inventory planning is difficult because demand is not constant.

Demand changes because of:

- weekly shopping patterns
- seasonality
- product-level popularity
- promotions
- holidays or special events
- store-specific differences
- pricing changes

If a company underestimates demand:

- shelves go empty
- revenue is lost
- customers may switch brands or stores

If a company overestimates demand:

- cash is trapped in inventory
- storage costs increase
- spoilage or markdown risk rises

Most forecasting projects end with a model score. SmartStock goes one step further by turning model output into inventory decisions.

---

## Use Case

Imagine a retail chain with multiple stores and a catalog of fast-moving products.

An inventory manager wants to answer:

- Which products will need replenishment next week?
- Which stores are exposed to stockout risk?
- Which items are carrying too much stock relative to expected demand?

A typical SmartStock workflow looks like this:

1. Historical daily sales are collected for each store and item.
2. The model learns from lagged demand, rolling averages, calendar effects, price, and promotions.
3. SmartStock predicts expected demand for the next `7` and `28` days.
4. The inventory layer compares forecasted demand to current inventory.
5. The app surfaces recommended reorder quantities and risk flags.

Example business scenario:

- `store_2` carries `item_03`
- expected 7-day demand is `82`
- current inventory is `54`
- safety stock is `14`
- recommended reorder quantity becomes `42`

That recommendation gives the team an actionable decision instead of just a prediction.

---

## Who This Project Is For

This project is framed for three business personas:

- Inventory manager
  Focus: reorder quantities, stockout risk, inventory coverage

- Category planner
  Focus: category-level demand patterns, product prioritization, medium-term planning

- Operations analyst
  Focus: forecast quality, excess inventory, service-level tradeoffs

It is also designed for a technical audience reviewing your portfolio:

- recruiters
- hiring managers
- data science interviewers
- analytics leaders

---

## What SmartStock Does

SmartStock combines four capabilities:

### 1. Demand forecasting

The pipeline predicts cumulative future demand for each item and store combination.

Supported forecast horizons:

- `7` days
- `28` days

### 2. Baseline benchmarking

The project includes simple benchmark forecasts so the machine learning model must prove its value.

Benchmarks include:

- naive-style rolling forecast
- seasonal rolling proxy
- optional classical benchmark in code

### 3. Inventory recommendations

The project converts forecasts into:

- `predicted_horizon_demand`
- `current_inventory`
- `safety_stock`
- `recommended_reorder_qty`
- `stockout_risk`
- `inventory_status`

### 4. Business-facing app

The Streamlit app presents:

- demand overview
- forecast explorer
- inventory planning dashboard
- diagnostics against baseline

---

## Why This Is a Strong Portfolio Project

Many portfolio projects show only one skill at a time. SmartStock shows several.

### Data science depth

- time-based train/validation/test splitting
- leakage-aware feature engineering
- multiple forecasting horizons
- baseline vs model comparison
- business-relevant evaluation metrics

### Engineering quality

- modular package structure under `src/`
- reusable pipeline instead of notebook-only logic
- saved model artifacts
- tests for core functionality
- app separated from training code

### Business maturity

- recommendations, not just predictions
- risk scoring and inventory classification
- stakeholder-friendly storytelling
- portfolio-ready product framing

### Demo value

The project can be explained quickly:

> “I built a retail demand forecasting system that predicts 7-day and 28-day demand, compares it to current inventory, and recommends reorder actions in a Streamlit app.”

That is a much stronger story than:

> “I trained a regression model on sales data.”

---

## Dataset

### Current implementation

This repository currently uses a synthetic retail demand dataset generated by the pipeline itself.

Why use synthetic data here:

- the project is runnable immediately from a blank workspace
- no external download is required
- schema and behavior still resemble a real retail forecasting problem
- it allows the portfolio project to remain fully reproducible

### Included columns

The generated raw dataset includes:

- `date`
- `store_id`
- `item_id`
- `category`
- `price`
- `promo_flag`
- `event_name`
- `target_sales`

### Behavioral patterns simulated

The data generator introduces:

- store-level demand differences
- item-level base demand
- weekly seasonality
- annual seasonality
- holiday spikes
- promotional lift
- price sensitivity
- random noise

### Swapping in a real dataset

The project is designed so you can later replace the synthetic data with a public dataset if desired.

Good future candidates:

- M5 forecasting competition style retail data
- Rossmann-style store demand data
- Favorita grocery sales style data

To swap datasets cleanly, preserve the conceptual schema:

- date key
- store identifier
- item identifier
- numeric demand target
- optional price/promotion/calendar signals

---

## Project Architecture

SmartStock is organized as a small analytics product with clear layers.

```text
Raw data
  -> validation
  -> feature engineering
  -> time-based split
  -> baseline forecasts
  -> machine learning training
  -> forecast generation
  -> inventory recommendation logic
  -> saved artifacts
  -> Streamlit app
```

### Core flow

1. Generate or load retail demand data.
2. Validate schema and keys.
3. Build lag, rolling, calendar, and aggregation features.
4. Create future-demand targets for each forecast horizon.
5. Split data by time into train, validation, and test windows.
6. Train the primary forecasting model.
7. Compare against baselines.
8. Produce latest-horizon forecasts.
9. Convert forecasts into inventory recommendations.
10. Save outputs for app consumption.

---

## Repository Structure

```text
.
├── app/
│   └── streamlit_app.py
├── assets/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── README.md
├── reports/
├── src/
│   └── smartstock/
│       ├── __init__.py
│       ├── baseline.py
│       ├── classical.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── inventory.py
│       ├── metrics.py
│       ├── modeling.py
│       └── pipeline.py
└── tests/
    ├── test_data.py
    ├── test_inventory.py
    └── test_modeling.py
```

### Folder purpose

- `app/`
  Contains the Streamlit user interface.

- `data/raw/`
  Stores raw generated or imported demand data.

- `data/processed/`
  Stores feature-rich tables, forecasts, and recommendation outputs.

- `models/`
  Stores serialized training artifacts.

- `reports/`
  Stores model metrics and reporting outputs.

- `src/smartstock/`
  Houses all reusable project logic.

- `tests/`
  Verifies data integrity, business rules, and modeling behavior.

- `notebooks/`
  Reserved for exploration and presentation experiments.

---

## Methodology

### Problem framing

This project treats retail demand forecasting as a supervised learning problem on panel time series data.

Instead of training one model per individual time series, SmartStock trains horizon-specific global models using engineered features across all store-item combinations.

This gives several advantages:

- shared learning across items and stores
- simpler implementation for a portfolio project
- strong baseline for tabular forecasting
- easier extension to richer features later

### Forecast target design

The model predicts cumulative demand over a future horizon.

That means:

- `target_h7` = total sales over the next `7` days
- `target_h28` = total sales over the next `28` days

This makes the later inventory rule much more intuitive, because the recommendation engine directly compares forecasted horizon demand to available inventory.

---

## Feature Engineering

Feature engineering is one of the strongest parts of the project because it demonstrates understanding of temporal structure and leakage prevention.

### Calendar features

Derived from `date`:

- `day_of_week`
- `week_of_year`
- `month`
- `day_of_month`
- `is_weekend`
- `is_holiday`

These help the model learn recurring demand patterns.

### Lag features

Historical demand snapshots:

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`

These capture short-term momentum and weekly seasonality.

### Rolling statistics

Computed from shifted demand:

- `rolling_mean_7`
- `rolling_mean_14`
- `rolling_mean_28`
- `rolling_std_7`
- `rolling_std_14`
- `rolling_std_28`

These represent local demand level and volatility.

### Cross-sectional aggregation features

The project also includes higher-level context:

- `category_rolling_mean_14`
- `store_rolling_mean_14`

These help the model learn broader trends beyond one specific item.

### Price and promotion features

Direct business drivers:

- `price`
- `promo_flag`
- event-driven effects via `event_name` and `is_holiday`

### Leakage prevention

All lags and rolling windows are shifted so they only use information available before the prediction date.

That matters because leakage is one of the most common mistakes in time series portfolio projects.

---

## Modeling Strategy

SmartStock uses a layered forecasting approach rather than jumping straight to one model.

### Baselines

Baselines are included because a forecasting model should always be compared against simple heuristics.

Implemented baselines:

- rolling-demand naive proxy
- seasonal-style rolling proxy

These provide a sanity check and establish whether the machine learning model is actually useful.

### Primary model

The main production model is:

- `GradientBoostingRegressor`

Why this choice:

- strong performance on tabular feature-based forecasting
- interpretable workflow for a portfolio project
- reliable local execution in this environment
- no heavy GPU or distributed setup required

### Optional classical benchmark

The repo also contains an optional classical benchmark path in [classical.py](/Users/pranavojha/Data%20Science%20/src/smartstock/classical.py).

This is not enabled by default in the bootstrap flow because it is slower, but it exists to show awareness of traditional forecasting methods.

### Horizon-specific training

Separate models are trained for:

- `7-day` demand
- `28-day` demand

This keeps each target well-defined and avoids forcing one model to learn conflicting output behavior across horizons.

---

## Inventory Intelligence Layer

This is the part that makes SmartStock feel like a business tool instead of just a forecasting model.

### Main formula

```text
recommended_reorder_qty = max(0, predicted_horizon_demand + safety_stock - current_inventory)
```

Interpretation:

- start from expected future demand
- add a protective inventory buffer
- subtract current inventory already on hand
- if the result is negative, order nothing

### Safety stock

Safety stock is estimated from:

- recent demand volatility
- lead time
- service-level factor

This keeps the logic transparent and easy to explain in interviews.

### Inventory outputs

The recommendation layer produces:

- `predicted_horizon_demand`
- `current_inventory`
- `safety_stock`
- `recommended_reorder_qty`
- `stockout_risk`
- `inventory_status`
- `estimated_excess_inventory`
- `estimated_stockout_units`
- `estimated_stockout_days`

### Risk logic

The project classifies inventory coverage into:

- `High` stockout risk
- `Medium` stockout risk
- `Low` stockout risk

Inventory status is labeled as:

- `Stockout risk`
- `Healthy`
- `Overstock risk`

This makes the output immediately usable in a dashboard or operations discussion.

---

## Evaluation Framework

### Statistical metrics

The project evaluates forecasting quality with:

- `MAE`
- `RMSE`
- `WAPE`
- safe `MAPE`

Why these matter:

- `MAE` is easy to interpret in demand units
- `RMSE` penalizes larger misses more strongly
- `WAPE` works well for aggregate business-facing forecasting comparisons
- safe `MAPE` handles zero-demand periods more carefully than naive MAPE

### Business evaluation

The project also emphasizes business-facing outputs:

- stockout exposure
- excess inventory
- reorder quantity
- horizon-level operational risk

### Current validation snapshot

From the current generated artifact run:

- 7-day baseline `WAPE`: `0.0719`
- 7-day model `WAPE`: `0.0700`
- 28-day baseline `WAPE`: `0.0614`
- 28-day model `WAPE`: `0.0480`

Interpretation:

- the 7-day horizon is only slightly better than baseline
- the 28-day horizon shows a stronger lift
- this creates a great portfolio conversation about where the model helps most and what you would improve next

---

## Streamlit App

The app lives in [streamlit_app.py](/Users/pranavojha/Data%20Science%20/app/streamlit_app.py).

### App sections

#### Overview

Shows:

- total daily demand over time
- category-level demand distribution
- overall business framing

#### Forecast Explorer

Allows the user to:

- filter by store
- filter by category
- inspect predicted demand by item
- compare risk and reorder actions within a slice

#### Inventory Planner

Shows:

- top reorder priorities
- inventory recommendation table
- risk-focused action views

#### Model Diagnostics

Displays:

- baseline vs model performance
- demand drift over time by category

### Interactive simulator controls

The sidebar lets a user change:

- forecast horizon
- lead time
- service buffer
- current inventory coverage

This is important because it makes the project interactive and decision-oriented instead of static.

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate artifacts

```bash
PYTHONPATH=src python3 -m smartstock.pipeline
```

This creates:

- raw synthetic retail data
- processed feature data
- saved forecasts
- saved inventory recommendations
- model bundle
- metrics report

### 3. Launch the app

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

### 4. Run tests

```bash
PYTHONPATH=src pytest
```

---

## Outputs and Artifacts

Generated outputs include:

- [synthetic_retail_demand.csv](/Users/pranavojha/Data%20Science%20/data/raw/synthetic_retail_demand.csv)
- [retail_demand_features.csv](/Users/pranavojha/Data%20Science%20/data/processed/retail_demand_features.csv)
- [latest_forecasts.csv](/Users/pranavojha/Data%20Science%20/data/processed/latest_forecasts.csv)
- [latest_recommendations.csv](/Users/pranavojha/Data%20Science%20/data/processed/latest_recommendations.csv)
- [smartstock_bundle.joblib](/Users/pranavojha/Data%20Science%20/models/smartstock_bundle.joblib)
- [model_metrics.json](/Users/pranavojha/Data%20Science%20/reports/model_metrics.json)

These artifacts separate training from consumption, which is a good engineering pattern for data products.

---

## Testing

The test suite covers the parts of the project that are most important to trust.

### Data tests

- duplicate key detection for `date + store_id + item_id`
- time split ordering
- no leakage in lag features

### Modeling tests

- baseline forecast generation
- training and prediction for both forecast horizons
- stable metrics with zero-demand periods

### Inventory tests

- reorder quantity never goes negative
- larger demand leads to larger reorder decisions
- low inventory plus high demand raises stockout risk

### Current status

The project currently passes:

- `9` tests

---

## Design Decisions

### Why synthetic data first

Using a synthetic dataset here was a deliberate choice, not a shortcut.

Benefits:

- fully reproducible
- no external data dependency
- immediate demo readiness
- easier onboarding for reviewers

### Why feature-based forecasting

Tree-based supervised forecasting is a very good portfolio choice because it demonstrates:

- feature engineering ability
- business variable integration
- strong baseline modeling practice
- scalable reasoning across many time series

### Why cumulative horizon targets

Predicting cumulative horizon demand makes the inventory recommendation step easier to explain and operationalize.

### Why a Streamlit app

A polished app makes the project:

- easier to demo
- easier to understand
- more memorable in a portfolio

---

## Limitations

This project is intentionally strong for a portfolio, but it is not pretending to be a production supply chain platform.

Current limitations:

- synthetic data rather than a real public retail source
- current inventory is simulated from recent demand rather than loaded from an ERP or warehouse system
- no probabilistic forecasting intervals yet
- no hierarchical reconciliation across item/category/store levels
- no live deployment or monitoring pipeline
- no advanced causal treatment for promotions

These are not weaknesses to hide. They are excellent talking points for “what I would improve next.”

---

## How to Extend This Project

Here are realistic ways to deepen SmartStock.

### Data upgrades

- replace synthetic data with a public retail dataset
- add richer event calendars
- include more realistic pricing and promotion schedules

### Modeling upgrades

- test `LightGBM` or `XGBoost`
- add quantile forecasts
- perform horizon-specific feature tuning
- add feature importance or SHAP analysis
- compare with Prophet or richer SARIMAX baselines

### Business upgrades

- user-entered inventory import
- reorder constraints such as minimum order quantity
- service-level optimization
- scenario planning for promotions
- lead-time variation by supplier

### Product upgrades

- stronger app design and layout polish
- downloadable reports
- category/store scorecards
- deployment on Streamlit Community Cloud

---

## Portfolio Talking Points

If you present this in an interview, these points work well:

- “I framed the project around a real business decision, not just prediction.”
- “I used time-aware feature engineering to avoid leakage.”
- “I compared the ML model against meaningful baselines.”
- “I translated demand forecasts into reorder actions and stockout risk.”
- “I built the project as a reusable package with tests and an interactive app.”
- “I intentionally chose a greenfield structure that can scale to a real public retail dataset.”

Short elevator pitch:

> SmartStock is a demand forecasting and inventory planning app that predicts 7-day and 28-day retail demand, estimates safety stock, and recommends reorder quantities by store and item.

---

## Future Improvements

If I were taking this from strong portfolio project to near-production case study, I would prioritize:

1. Replacing the synthetic dataset with a public retail benchmark.
2. Improving the 7-day model performance with richer temporal and cross-series features.
3. Adding SHAP-based model explainability.
4. Introducing probabilistic forecasts and service-level optimization.
5. Deploying the app publicly with screenshots, demo GIFs, and a polished landing section.

---

## Final Note

SmartStock is meant to show that strong data science work is more than fitting a model. It is about connecting data, prediction, evaluation, and business action in a form that others can actually use.

That is exactly the kind of story a portfolio project should tell.

# Smart_stock
