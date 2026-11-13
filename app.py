"""Project 05 — Courier Delivery Planning and Feedback Assistant.

Section 7, step 1: Delivery Overview and Duration Estimate only.
Uses the trained model; does not retrain it.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

st.set_page_config(page_title="Courier Delivery Assistant", page_icon="📦", layout="wide")

@st.cache_data
def load_data():
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    main = pd.read_csv(DATA / "main.csv")
    examples = pd.read_csv(DATA / "example_inputs.csv")
    return meta, main, examples

@st.cache_resource
def load_model():
    # Load only this project's trusted, locally saved model.
    return joblib.load(MODELS / "supervised.joblib")

try:
    meta, main, examples = load_data()
    model = load_model()
except (FileNotFoundError, ValueError, ImportError, OSError) as exc:
    st.error(f"Couldn't load required project files: {exc}")
    st.stop()

st.title("📦 Courier Delivery Planning & Feedback Assistant")
st.caption("Section 7 · Delivery overview & duration estimate · Generated classroom data")
st.header("1. Delivery overview and duration estimate")
st.write("Enter a delivery scenario to estimate elapsed delivery time in **minutes**. "
         "This is a demonstration model, not an arrival-time guarantee.")

st.subheader("Dataset overview")
cols = st.columns(4)
cols[0].metric("Total scenarios", f"{len(main):,}")
for i, split in enumerate(["train", "validation", "test"], start=1):
    cols[i].metric(split.title(), int((main["split"] == split).sum()))

train_data = main.loc[main["split"] == "train"]
st.caption("Duration distribution — training split only; units: minutes")
st.bar_chart(train_data["delivery_minutes"].round(-1).value_counts().sort_index().rename_axis("Minutes").rename("Jobs"))

st.subheader("Try a delivery estimate")
example_names = ["Custom input"] + examples["record_id"].astype(str).tolist()
choice = st.selectbox("Select a supplied example (or enter your own)", example_names)
selected = examples.loc[examples["record_id"] == choice]
example = selected.iloc[0] if not selected.empty else None

ranges = meta["numeric_ranges"]
vehicle_values = meta["category_values"]["vehicle"]
def number_default(name, fallback):
    return float(example[name]) if example is not None else fallback

with st.form("delivery_form"):
    left, right = st.columns(2)
    with left:
        distance = st.number_input("Distance (km)", value=number_default("distance_km", 10.0), step=0.1)
        weight = st.number_input("Parcel weight (kg)", value=number_default("weight_kg", 2.0), step=0.1)
        traffic = st.number_input("Traffic index (1–5)", value=int(number_default("traffic_index", 3)), step=1)
    with right:
        stops = st.number_input("Number of stops (0–5)", value=int(number_default("stops_count", 1)), step=1)
        initial_vehicle = str(example["vehicle"]) if example is not None else vehicle_values[0]
        vehicle = st.selectbox("Vehicle type", vehicle_values, index=vehicle_values.index(initial_vehicle))
    submitted = st.form_submit_button("Estimate delivery duration", type="primary")

if submitted:
    inputs = {"distance_km": distance, "weight_kg": weight, "traffic_index": traffic,
              "stops_count": stops, "vehicle": vehicle}
    errors = []
    for field in meta["numeric_features"]:
        low, high = ranges[field]
        if not low <= inputs[field] <= high:
            errors.append(f"{field} must be between {low} and {high}.")
    if inputs["vehicle"] not in vehicle_values:
        errors.append("Vehicle type is not allowed.")
    if errors:
        for error in errors:
            st.error(error)
    else:
        frame = pd.DataFrame([inputs], columns=meta["numeric_features"] + meta["categorical_features"])
        estimate = float(model.predict(frame)[0])
        st.success("Prediction complete")
        st.metric("Estimated delivery duration", f"{estimate:.1f} minutes")
        st.caption("Estimate based on generated classroom examples; not a guaranteed delivery time.")

st.divider()
st.caption("Uses the saved Decision Tree model. No retraining, maps, live tracking, or API required.")
