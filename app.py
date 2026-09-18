"""
Streamlit dashboard: Predicting Health Outcomes in Sunderland

Using Sunderland-only Environmental, Socioeconomic and Deprivation Data
Run with:

    streamlit run app.py

NOTE (supervisor revision):
The national ONS Local-Authority-level approach has been removed from this
project on supervisor guidance. The project now focuses ONLY on Sunderland
itself, using Sunderland-specific datasets (English Indices of Deprivation 2025
at LSOA level, Sunderland Silksworth air quality station, and JSNA / Smart City data),
rather than comparing Sunderland against ~270 other English Local Authorities using national ONS data.
"""

# ================================================================
# IMPORTS
# ================================================================

# Here I imported json because I used it to open and read my
# GeoJSON file. I needed this later when creating the map of
# Sunderland's LSOAs.
import json

# Here I imported os because I used it to work with my file paths.
# I did this so the application could find my data and output folders
# relative to where this app.py file is located.
import os

# Here I imported NumPy for numerical operations.
# It was included as part of the data analysis environment.
import numpy as np

# Here I imported Pandas because I used it to load and work with
# my CSV datasets. Pandas allowed me to store the datasets as
# DataFrames and access the different columns easily.
import pandas as pd

# Here I imported Plotly Express because I used it to create
# interactive charts and maps in the Streamlit dashboard.
import plotly.express as px

# Here I imported Plotly Graph Objects because I needed more control
# over some of my charts. For example, I used it to add the
# perfect-prediction line to my predicted-versus-actual plot.
import plotly.graph_objects as go

# Here I imported Streamlit because I used it to turn my Python code
# into an interactive web dashboard where the user can select
# different pages, models and variables.
import streamlit as st


# ================================================================
# FILE PATHS
# ================================================================

# Here I found the location of the current app.py file.
# I did this instead of writing my personal computer's full file path,
# so that the application can find the other project folders relative
# to this file.
BASE = os.getcwd()

# Here I created the path to my data folder.
# This is where I stored my CSV datasets and GeoJSON file.
DATA_DIR = os.path.join(BASE, "data")

# Here I created the path to my outputs folder.
# This is where I stored my saved model results, predictions,
# feature importance results and EDA images.
OUT_DIR = os.path.join(BASE, "outputs")


# ================================================================
# STREAMLIT PAGE SETUP
# ================================================================

# Here I set up the basic appearance of my Streamlit dashboard.
# I used "wide" because my dashboard contains a map, tables and
# charts, so I wanted to give them more horizontal space.
# I also added a title and a hospital icon so the dashboard
# is easier to identify.
st.set_page_config(
    page_title="Sunderland Health Outcomes",
    layout="wide",
    page_icon="🏥"
)


# ================================================================
# SUNDERLAND DATA LOADER
# ================================================================

# Here I used Streamlit's cache_data.
# I did this because I did not want Streamlit to reload the same
# files every time I interacted with the dashboard.
# Caching stores the result after the first time the function runs,
# which helps make the dashboard faster.
@st.cache_data
def load_sunderland():

    # Here I loaded the Sunderland LSOA dataset.
    # This contains the Health Deprivation & Disability target and
    # the other deprivation variables used as predictors across 185 LSOAs.
    df = pd.read_csv(
        f"{DATA_DIR}/sunderland_lsoa_dataset.csv"
    )

    # Here I opened the Sunderland LSOA GeoJSON file.
    # I needed this to display the LSOA-level results on a map.
    with open(
        f"{DATA_DIR}/lsoa_sunderland_clean.geojson"
    ) as f:

        # Here I converted the GeoJSON into a Python object
        # that Plotly can use for the map.
        geo = json.load(f)

    # Here I returned both the Sunderland dataset and its
    # geographical boundaries.
    return df, geo


# ================================================================
# AIR QUALITY DATA LOADER
# ================================================================

# Here I created a separate function to load my air-quality data.
# I did this because the air-quality data is separate from my main
# modelling dataset and is mainly used for environmental context in
# the dashboard, since it comes from a single fixed monitoring station
# rather than 185 separate LSOA readings.
@st.cache_data
def load_air_quality():

    # Here I loaded the daily air-quality dataset.
    # I used parse_dates on the date column so Pandas treats it
    # as an actual date rather than ordinary text.
    daily = pd.read_csv(
        f"{DATA_DIR}/air_quality_daily.csv",
        parse_dates=["date"]
    )

    # Here I also loaded the monthly air-quality dataset.
    monthly = pd.read_csv(
        f"{DATA_DIR}/air_quality_monthly.csv"
    )

    # Here I returned both versions of the air-quality data.
    return daily, monthly


# ================================================================
# MODEL OUTPUT LOADER
# ================================================================

# Here I created a function to load the results from my Sunderland model.
# I kept the "prefix" argument so that this function can easily be reused
# if further Sunderland Smart City datasets are merged in and re-modelled later.
@st.cache_data
def load_model_outputs(prefix="sunderland"):

    # Here I loaded the model evaluation metrics (R², MAE, RMSE, CV scores).
    metrics = json.load(
        open(f"{OUT_DIR}/{prefix}_metrics.json")
    )

    # Here I loaded the permutation feature-importance results.
    importances = json.load(
        open(f"{OUT_DIR}/{prefix}_importances.json")
    )

    # Here I loaded the predictions made on the held-out test set.
    test_preds = json.load(
        open(f"{OUT_DIR}/{prefix}_test_predictions.json")
    )

    # Here I loaded the predictions for the full dataset.
    full_preds = pd.read_csv(
        f"{OUT_DIR}/{prefix}_predictions_full.csv"
    )

    # Here I loaded the correlation matrix created during EDA.
    corr = pd.read_csv(
        f"{OUT_DIR}/{prefix}_correlation_matrix.csv",
        index_col=0
    )

    return metrics, importances, test_preds, full_preds, corr


# ================================================================
# SUNDERLAND FEATURE LABELS
# ================================================================

# Here I created a dictionary to change technical column names
# into labels that are easier for someone using the dashboard to understand.
SUN_FEATURE_LABELS = {

    "income_score":
        "Income deprivation score",

    "employment_score":
        "Employment deprivation score",

    "education_score":
        "Education, Skills & Training score",

    "crime_score":
        "Crime score",

    "housing_barriers_score":
        "Barriers to Housing & Services score",

    "living_environment_score":
        "Living Environment score",

    "pct_population_0_15":
        "Population aged 0-15 (%)",

    "pct_population_60_plus":
        "Population aged 60+ (%)",
}


# ================================================================
# MODEL COLOURS
# ================================================================

# Here I gave each of my three models a consistent colour.
# I did this so that when someone moves between different charts,
# they can easily recognise the same model.
MODEL_COLORS = {
    "Linear Regression": "#4C72B0",
    "Random Forest": "#55A868",
    "XGBoost": "#C44E52"
}


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title("Sunderland Health Outcomes")

# Short description highlighting Sunderland Smart City & JSNA data only
st.sidebar.markdown(
    "Predicting health outcomes across Sunderland's LSOAs using "
    "Sunderland-only socioeconomic, deprivation and environmental "
    "data.\n\n"
    "**Data Sources:**\n"
    "• English Indices of Deprivation 2025 (185 Sunderland LSOAs)\n"
    "• UK-AIR Sunderland Silksworth monitoring station\n"
    "• Sunderland Data Observatory (JSNA) / Smart City Platform"
)

# Navigation radio menu (National ONS section completely removed)
page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Sunderland analysis (LSOA-level)",
        "Air quality context",
        "Data & methodology"
    ],
)


# ================================================================
# OVERVIEW PAGE
# ================================================================

if page == "Overview":

    st.title("Predicting Health Outcomes in Sunderland")

    st.markdown(
        """
        This dashboard predicts health outcomes across Sunderland's **185 Lower-layer
        Super Output Areas (LSOAs)**, using **Sunderland-only data**: socioeconomic
        and deprivation indicators from the English Indices of Deprivation 2025,
        alongside local air quality data from the Sunderland Silksworth monitoring
        station, drawn from the Sunderland Data Observatory (JSNA) and Smart City Data Platform.

        Following supervisor guidance, national ONS Local-Authority-level data comparing
        Sunderland to other English councils has been entirely removed. Every indicator
        used here describes Sunderland itself, at genuine within-city (LSOA) detail.
        """
    )

    st.subheader("Sunderland LSOA Model")

    st.markdown(
        "**Target:** Health Deprivation & Disability score (IoD2025)\n\n"
        "**Scale:** 185 Lower-layer Super Output Areas (LSOAs) within Sunderland\n\n"
        "**Predictors:** Income, employment, education, crime, "
        "housing barriers, and living environment deprivation, plus population structure.\n\n"
        "This gives real within-Sunderland, ward-level geographic detail and maps, "
        "using strictly Sunderland's own local data."
    )

    st.divider()

    st.subheader("Model performance at a glance")

    # Load Sunderland model metrics
    sun_metrics, *_ = load_model_outputs("sunderland")

    # Identify best model based on test R²
    best_sun = max(
        sun_metrics,
        key=lambda k: sun_metrics[k]["test_r2"]
    )

    # Display best performing model metric
    st.metric(
        "Best Model",
        best_sun,
        f"R² = {sun_metrics[best_sun]['test_r2']:.3f}"
    )

    st.caption(
        "The model achieves a strong out-of-sample R², indicating that "
        "socioeconomic and deprivation conditions are highly predictive "
        "of health outcomes across Sunderland's LSOAs -- consistent with "
        "the wider public health literature on the social determinants of health."
    )


# ================================================================
# SUNDERLAND LSOA PAGE
# ================================================================

elif page == "Sunderland analysis (LSOA-level)":

    st.title("Sunderland: Health Deprivation across LSOAs")

    # Load data and saved outputs
    sun_df, sun_geo = load_sunderland()
    metrics, importances, test_preds, full_preds, corr = load_model_outputs("sunderland")

    # Four main tabs for analysis
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Map",
            "Model comparison",
            "Feature importance",
            "Explore relationships"
        ]
    )

    # ------------------------------------------------------------
    # TAB 1: MAP
    # ------------------------------------------------------------
    with tab1:
        metric_choice = st.selectbox(
            "Map metric",
            [
                "Actual Health Deprivation score",
                "Predicted score",
                "Residual (actual - predicted)",
                "Health Deprivation decile",
                "Income deprivation score",
                "Employment deprivation score",
            ],
        )

        col_map = {
            "Actual Health Deprivation score": "health_deprivation_score",
            "Predicted score": "predicted_health_deprivation_score",
            "Residual (actual - predicted)": "residual",
            "Health Deprivation decile": "health_deprivation_decile",
            "Income deprivation score": "income_score",
            "Employment deprivation score": "employment_score",
        }

        col = col_map[metric_choice]
        map_df = full_preds if col in full_preds.columns else sun_df

        fig = px.choropleth_mapbox(
            map_df,
            geojson=sun_geo,
            locations="lsoa_code",
            featureidkey="properties.lsoa_code",
            color=col,
            color_continuous_scale="RdBu_r" if col == "residual" else "OrRd",
            mapbox_style="carto-positron",
            zoom=10.6,
            center={"lat": 54.88, "lon": -1.43},
            opacity=0.8,
            hover_name="lsoa_name" if "lsoa_name" in map_df.columns else None,
            labels={col: metric_choice},
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Note: 6 of 185 Sunderland LSOAs use boundaries that changed "
            "between the 2011 and 2021 Census and are not shown on the map "
            "(they remain in all tabular analysis and model training)."
        )

    # ------------------------------------------------------------
    # TAB 2: MODEL COMPARISON
    # ------------------------------------------------------------
    with tab2:
        st.subheader("Model comparison")

        met_df = pd.DataFrame(metrics).T
        met_df = met_df.rename(
            columns={
                "cv_r2_mean": "CV R² (mean)",
                "cv_r2_std": "CV R² (std)",
                "test_mae": "Test MAE",
                "test_rmse": "Test RMSE",
                "test_r2": "Test R²",
            }
        )

        st.dataframe(
            met_df.style.format("{:.3f}").highlight_max(
                subset=["Test R²"],
                color="lightgreen"
            ),
            use_container_width=True
        )

        st.subheader("Predicted vs Actual (held-out test set)")

        tp = pd.DataFrame(test_preds)

        model_pick = st.radio(
            "Model",
            list(metrics.keys()),
            horizontal=True,
            key="sun_model_pick"
        )

        fig2 = px.scatter(
            tp,
            x="actual",
            y=model_pick,
            labels={
                "actual": "Actual Health Deprivation score",
                model_pick: "Predicted"
            },
            color_discrete_sequence=[MODEL_COLORS[model_pick]]
        )

        lims = [tp["actual"].min(), tp["actual"].max()]

        fig2.add_trace(
            go.Scatter(
                x=lims,
                y=lims,
                mode="lines",
                line=dict(dash="dash", color="grey"),
                name="Perfect prediction"
            )
        )

        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------------------
    # TAB 3: FEATURE IMPORTANCE
    # ------------------------------------------------------------
    with tab3:
        st.subheader("Which factors matter most?")

        model_pick2 = st.radio(
            "Model",
            list(importances.keys()),
            horizontal=True,
            key="sun_imp_model"
        )

        imp = importances[model_pick2]
        imp_df = pd.DataFrame(
            {
                "feature": list(imp.keys()),
                "importance": list(imp.values())
            }
        )

        imp_df["label"] = imp_df["feature"].map(SUN_FEATURE_LABELS)
        imp_df = imp_df.sort_values("importance", ascending=True)

        fig3 = px.bar(
            imp_df,
            x="importance",
            y="label",
            orientation="h",
            color_discrete_sequence=[MODEL_COLORS[model_pick2]]
        )

        fig3.update_layout(
            height=450,
            xaxis_title="Permutation importance",
            yaxis_title=""
        )

        st.plotly_chart(fig3, use_container_width=True)

    # ------------------------------------------------------------
    # TAB 4: EXPLORE RELATIONSHIPS
    # ------------------------------------------------------------
    with tab4:
        st.subheader("Scatter explorer")

        x_col = st.selectbox(
            "X variable",
            list(SUN_FEATURE_LABELS.keys()),
            format_func=lambda c: SUN_FEATURE_LABELS[c]
        )

        fig4 = px.scatter(
            sun_df,
            x=x_col,
            y="health_deprivation_score",
            hover_name="lsoa_name" if "lsoa_name" in sun_df.columns else None,
            labels={
                x_col: SUN_FEATURE_LABELS[x_col],
                "health_deprivation_score": "Health Deprivation score"
            },
            trendline="ols",
            color_discrete_sequence=["#C44E52"]
        )

        st.plotly_chart(fig4, use_container_width=True)


# ================================================================
# AIR QUALITY PAGE
# ================================================================

elif page == "Air quality context":

    st.title("Air quality context: Sunderland Silksworth monitoring station")

    daily, monthly = load_air_quality()

    st.markdown(
        "Hourly air quality readings from the **UK-AIR Sunderland Silksworth** "
        "monitoring station, aggregated to daily and monthly means. "
        "Because this dataset represents a single fixed monitoring point rather than "
        "185 separate LSOA readings, it is included as local environmental "
        "context rather than as a numeric predictor in the ML models."
    )

    pollutant = st.selectbox(
        "Pollutant",
        ["pm25", "pm10", "no2", "nox", "no", "ozone"],
        format_func=lambda p: {
            "pm25": "PM2.5",
            "pm10": "PM10",
            "no2": "Nitrogen dioxide",
            "nox": "Nitrogen oxides",
            "no": "Nitric oxide",
            "ozone": "Ozone"
        }[p]
    )

    fig = px.line(
        daily,
        x="date",
        y=pollutant,
        title=f"Daily mean {pollutant.upper()} (µg/m³)"
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Mean PM2.5",
        f"{daily['pm25'].mean():.1f} µg/m³",
        help="WHO interim target 3 (annual): 15 µg/m³"
    )

    c2.metric(
        "Mean NO2",
        f"{daily['no2'].mean():.1f} µg/m³",
        help="UK annual objective: 40 µg/m³"
    )

    c3.metric(
        "Days PM2.5 > 15 µg/m³",
        f"{(daily['pm25'] > 15).sum()} / {len(daily)}"
    )

    # Display EDA air quality trend image if available
    eda_img_path = f"{OUT_DIR}/eda_air_quality_trend.png"
    if os.path.exists(eda_img_path):
        st.image(eda_img_path)


# ================================================================
# DATA AND METHODOLOGY PAGE
# ================================================================

else:

    st.title("Data & methodology")

    st.markdown(
        """
        ### Data Sources

        | Dataset | Geography | Use in this project |
        | :--- | :--- | :--- |
        | **English Indices of Deprivation 2025** | LSOA (185 areas within Sunderland) | Target (*Health Deprivation & Disability score*) + predictors (*other deprivation domains*) |
        | **UK-AIR Sunderland Silksworth Station** | Single monitoring point | Local environmental context (time-series trend) |
        | **Sunderland Data Observatory (JSNA) / Smart City** | Ward / LSOA (theme-dependent) | Additional Sunderland-only local context |

        ---

        ### Why Sunderland-only, and why LSOA level?
        On supervisor guidance, this project no longer uses national ONS Local-Authority-level
        data or compares Sunderland against other English areas. Every indicator used here
        describes Sunderland itself.

        The target variable and predictors are modelled at **LSOA level (185 areas within Sunderland)**,
        using the English Indices of Deprivation 2025 -- the one dataset that is genuinely
        resolved at that geography for Sunderland. The target is the *Health Deprivation and Disability score*,
        predicted from the other IMD domains (income, employment, education, crime, housing barriers,
        living environment) plus population structure.

        The Sunderland Silksworth air quality data is shown as local environmental context (a time trend,
        not a model predictor), since it comes from a single fixed monitoring point rather than 185
        separate LSOA readings.

        ---

        ### Modelling Approach
        * **Models compared:** Linear Regression, Random Forest, XGBoost
        * **Split:** 80/20 train/test, 5-fold cross-validation on the training set
        * **Metrics:** MAE, RMSE, R² (test set) + CV R² (mean ± std)
        * **Feature importance:** Permutation importance on the held-out test set (model-agnostic)

        ---

        ### Limitations
        1. **Boundaries:** LSOA boundaries changed slightly between the 2011 and 2021 Census; 6 of
           Sunderland's 185 LSOAs do not have a matching 2011 boundary in the mapping file used, so
           they are excluded from the interactive map (but are included in all tabular analysis and modelling).
        2. **Domain Correlation:** The Health Deprivation & Disability score is itself one of seven domains
           in the overall IMD, so predictor domains are not fully independent of the target by construction --
           results describe established deprivation associations rather than strict causal mechanisms.
        3. **Single Air Station:** Air quality reflects one fixed monitoring location and cannot be spatially
           disaggregated across Sunderland's 185 LSOAs without fabricating non-existent variation.
        """
    )