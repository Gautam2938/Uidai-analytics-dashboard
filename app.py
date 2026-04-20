import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    classification_report
)

DATASET_SCHEMAS = {
    "Enrolment": {
        "required": {"age_0_5", "age_5_17", "age_18_greater"},
    },
    "Demographic": {
        "required": {"demo_age_5_17", "demo_age_17_"},
    },
    "Biometric": {
        "required": {"bio_age_5_17", "bio_age_17_"},
    },
}



# APP CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Unified Demographic Analytics",
    layout="wide"
)

st.title("Unified Demographic Analytics Dashboard")
st.caption("Enrolment • Demographic • Biometric")


# DATA LOADING WITH CACHING
# --------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    return df

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is None:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

df = load_data(uploaded_file)

def detect_dataset_type(df):
    cols = set(df.columns)

    for dtype, schema in DATASET_SCHEMAS.items():
        if schema["required"].issubset(cols):
            return dtype

    return None


auto_detected_type = detect_dataset_type(df)



# DATASET TYPE SELECTION
# --------------------------------------------------
dataset_type = st.selectbox(
    "Select Dataset Category",
    options=list(DATASET_SCHEMAS.keys()),
    index=list(DATASET_SCHEMAS.keys()).index(auto_detected_type)
    if auto_detected_type in DATASET_SCHEMAS
    else 0
)

required_cols = DATASET_SCHEMAS[dataset_type]["required"]

missing_cols = required_cols - set(df.columns)

if missing_cols:
    st.error(
        f"""
        Dataset mismatch detected.

        Selected category: **{dataset_type}**
        Missing required columns:
        {', '.join(missing_cols)}

        Please select the correct dataset category
        or upload a compatible file.
        """
    )
    st.stop()



# STANDARDIZE COLUMNS
# --------------------------------------------------
if dataset_type == "Enrolment":
    age_columns = ["age_0_5", "age_5_17", "age_18_greater"]
    df["total_population"] = df[age_columns].sum(axis=1)

elif dataset_type == "Demographic":
    age_columns = ["demo_age_5_17", "demo_age_17_"]
    df["total_population"] = df[age_columns].sum(axis=1)

elif dataset_type == "Biometric":
    age_columns = ["bio_age_5_17", "bio_age_17_"]
    df["total_population"] = df[age_columns].sum(axis=1)


# GLOBAL FILTERS
# --------------------------------------------------
st.sidebar.header("Filters")

states = st.sidebar.multiselect(
    "Select States",
    options=df["state"].unique(),
    default=df["state"].unique()
)

date_range = st.sidebar.date_input(
    "Date Range",
    [df["date"].min(), df["date"].max()]
)

filtered_df = df[
    (df["state"].isin(states)) &
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1]))
]


# EXECUTIVE SUMMARY
# --------------------------------------------------
st.subheader("Executive Summary")
st.write(f"""
• Dataset type: **{dataset_type}**  
• Records analyzed: **{len(filtered_df):,}**  
• States covered: **{filtered_df['state'].nunique()}**  
• Time span: **{filtered_df['date'].min().date()} → {filtered_df['date'].max().date()}**
""")

# 1. TIME-SERIES FORECASTING
# --------------------------------------------------
st.header("1. Time-Series Forecasting")

ts = (
    filtered_df.groupby("date")["total_population"]
    .sum()
    .reset_index()
)

ts["t"] = np.arange(len(ts))

model = LinearRegression()
model.fit(ts[["t"]], ts["total_population"])
ts["prediction"] = model.predict(ts[["t"]])

mae = mean_absolute_error(ts["total_population"], ts["prediction"])

fig, ax = plt.subplots()
ax.plot(ts["date"], ts["total_population"], label="Actual")
ax.plot(ts["date"], ts["prediction"], label="Forecast")
ax.legend()
ax.grid(True)
ax.set_title("Population Forecast")

st.pyplot(fig)
st.metric("Mean Absolute Error", round(mae, 2))

# 2. ANOMALY DETECTION
# --------------------------------------------------
st.header("2. Anomaly Detection")

state_ts = (
    filtered_df.groupby(["state", "date"])["total_population"]
    .sum()
    .reset_index()
)

iso = IsolationForest(contamination=0.01, random_state=42)
state_ts["anomaly"] = iso.fit_predict(state_ts[["total_population"]])

anomalies = state_ts[state_ts["anomaly"] == -1]

st.subheader("Detected Anomalies")
st.dataframe(anomalies)

# 3. HEATMAP
# --------------------------------------------------
st.header("Age Distribution Heatmap")

heatmap_data = (
    filtered_df.groupby("state")[age_columns]
    .sum()
)

heatmap_norm = heatmap_data.div(
    heatmap_data.sum(axis=1), axis=0
)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    heatmap_norm,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    linewidths=0.5,
    ax=ax
)
ax.set_title("Normalized Age Distribution by State")
st.pyplot(fig)

st.header("District-Level Age Distribution Heatmap")

selected_state = st.selectbox(
    "Select State for District Heatmap",
    options=filtered_df["state"].unique()
)

district_heatmap = (
    filtered_df[filtered_df["state"] == selected_state]
    .groupby("district")[age_columns]
    .sum()
)

district_heatmap_norm = district_heatmap.div(
    district_heatmap.sum(axis=1), axis=0
)

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    district_heatmap_norm,
    cmap="coolwarm",
    linewidths=0.3,
    ax=ax
)

ax.set_title(
    f"Normalized Age Distribution by District — {selected_state}"
)
ax.set_xlabel("Age Group")
ax.set_ylabel("District")

st.pyplot(fig)


# 4. CLUSTERING (ONLY WHEN MEANINGFUL)
# --------------------------------------------------
if dataset_type == "Enrolment":
    st.header("3. Geographic Clustering (District Level)")

    district_features = (
        filtered_df.groupby("district")[age_columns]
        .sum()
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(district_features)

    k = st.slider("Number of clusters", 2, 6, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    district_features["cluster"] = kmeans.fit_predict(X_scaled)

    st.dataframe(district_features.reset_index())

# 5. SUPERVISED LEARNING (ONLY FOR ENROLMENT)
# --------------------------------------------------
if dataset_type == "Enrolment":
    st.header("4. Supervised Learning")

    district_features["child_ratio"] = (
        district_features["age_0_5"] /
        district_features.sum(axis=1)
    )

    threshold = st.slider(
        "Child Population Threshold",
        0.1, 0.5, 0.25
    )

    district_features["high_child"] = (
        district_features["child_ratio"] > threshold
    ).astype(int)

    X = district_features[age_columns]
    y = district_features["high_child"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
