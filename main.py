import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Traffic Flow Explorer",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic Flow Explorer")
st.caption("Visualize traffic volume and perform basic exploratory analysis (EDA) with Streamlit.")

# ----------------------------
# Data loading
# ----------------------------
DEFAULT_PATH = "traffic.csv"  # Put traffic.csv in the same folder as this app.py

@st.cache_data
def load_data_from_path(path):
    df = pd.read_csv(path)
    return _prepare_data(df)

@st.cache_data
def _prepare_data(df):
    out = df.copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")
    out["Junction"] = pd.to_numeric(out["Junction"], errors="coerce").astype("Int64")
    out["Vehicles"] = pd.to_numeric(out["Vehicles"], errors="coerce")

    out = out.dropna(subset=["DateTime", "Junction", "Vehicles"]).copy()
    out["Junction"] = out["Junction"].astype(int)

    # Feature engineering for EDA
    out["Date"] = out["DateTime"].dt.date
    out["Hour"] = out["DateTime"].dt.hour
    out["DayOfWeek"] = out["DateTime"].dt.dayofweek  # Monday=0
    out["DayName"] = out["DateTime"].dt.day_name()
    out["Month"] = out["DateTime"].dt.to_period("M").astype(str)
    out["Year"] = out["DateTime"].dt.year

    return out.sort_values("DateTime").reset_index(drop=True)

# Optional uploader
uploaded = st.sidebar.file_uploader("Upload a CSV (optional)", type=["csv"])
st.sidebar.markdown("If no file is uploaded, the app will try to read `traffic.csv` from the current folder.")

try:
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        data = _prepare_data(raw_df)
        source_name = uploaded.name
    else:
        data = load_data_from_path(DEFAULT_PATH)
        source_name = DEFAULT_PATH
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Filters")

all_junctions = sorted(data["Junction"].unique().tolist())
selected_junctions = st.sidebar.multiselect(
    "Select junction(s)",
    options=all_junctions,
    default=all_junctions,
)

if not selected_junctions:
    st.warning("Please select at least one junction.")
    st.stop()

min_date = data["DateTime"].min().date()
max_date = data["DateTime"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

start_date, end_date = date_range


agg_label = st.sidebar.selectbox("Time aggregation", ["Hourly", "Daily", "Weekly", "Monthly"], index=1)
agg_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "MS"}
agg_freq = agg_map[agg_label]

show_smooth = st.sidebar.checkbox("Show rolling average", value=True)
smooth_window = st.sidebar.slider("Rolling window (periods)", 2, 30, 7) if show_smooth else 1
show_raw = st.sidebar.checkbox("Show raw data preview", value=False)

# ----------------------------
# Filter data
# ----------------------------
mask = (
    data["Junction"].isin(selected_junctions)
    & (data["DateTime"].dt.date >= start_date)
    & (data["DateTime"].dt.date <= end_date)
)
filtered = data.loc[mask].copy()

if filtered.empty:
    st.warning("No data in the selected range.")
    st.stop()

# ----------------------------
# Top info + KPI cards
# ----------------------------
st.info(f"Data source: **{source_name}** | Records after filtering: **{len(filtered):,}**")

col1, col2, col3, col4 = st.columns(4)

total_vehicles = int(filtered["Vehicles"].sum())
avg_vehicles = float(filtered["Vehicles"].mean())
peak_row = filtered.loc[filtered["Vehicles"].idxmax()]
busiest_junction = filtered.groupby("Junction")["Vehicles"].sum().idxmax()

col1.metric("Total Vehicles", f"{total_vehicles:,}")  # https://docs.streamlit.io/develop/api-reference/data/st.metric
col2.metric("Avg Vehicles / Record", f"{avg_vehicles:.2f}")
col3.metric(
    "Peak Single Record",
    f"{int(peak_row['Vehicles']):,}",
    help=f"{peak_row['DateTime']} @ Junction {int(peak_row['Junction'])}"
)
col4.metric("Busiest Junction", f"J{int(busiest_junction)}")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend", "🕒 Time Patterns", "🔍 Comparison", "🧾 Data & Summary"])

with tab1:
    st.subheader("Traffic trend over time")

    # Aggregate by time and junction
    ts = (
        filtered.set_index("DateTime")
        .groupby("Junction")["Vehicles"]
        .resample(agg_freq)
        .sum()
        .reset_index()
    )
    trend = ts.pivot(index="DateTime", columns="Junction", values="Vehicles").sort_index()
    trend.columns = [f"Junction {c}" for c in trend.columns]

    st.line_chart(trend, use_container_width=True)

    if show_smooth and len(trend) >= smooth_window:
        st.markdown(f"**Rolling average ({smooth_window} periods)**")
        smooth = trend.rolling(smooth_window, min_periods=1).mean()
        st.line_chart(smooth, use_container_width=True)

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown("**Monthly total by junction**")
        monthly = (
            filtered.assign(MonthStart=filtered["DateTime"].dt.to_period("M").dt.to_timestamp())
            .groupby(["MonthStart", "Junction"])["Vehicles"]
            .sum()
            .reset_index()
            .pivot(index="MonthStart", columns="Junction", values="Vehicles")
        )
        monthly.columns = [f"Junction {c}" for c in monthly.columns]
        st.area_chart(monthly, use_container_width=True)

    with c2:
        st.markdown("**Top 10 busiest timestamps**")
        top10 = (
            filtered[["DateTime", "Junction", "Vehicles"]]
            .sort_values("Vehicles", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        st.dataframe(top10, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Temporal patterns (hour/day/week)")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Average traffic by hour of day**")
        hour_profile = (
            filtered.groupby(["Hour", "Junction"])["Vehicles"]
            .mean()
            .reset_index()
            .pivot(index="Hour", columns="Junction", values="Vehicles")
            .sort_index()
        )
        hour_profile.columns = [f"Junction {c}" for c in hour_profile.columns]
        st.line_chart(hour_profile, use_container_width=True)

    with c2:
        st.markdown("**Average traffic by day of week**")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_profile = (
            filtered.groupby(["DayName", "Junction"])["Vehicles"]
            .mean()
            .reset_index()
        )
        dow_profile["DayName"] = pd.Categorical(dow_profile["DayName"], categories=dow_order, ordered=True)
        dow_profile = (
            dow_profile.sort_values("DayName")
            .pivot(index="DayName", columns="Junction", values="Vehicles")
        )
        dow_profile.columns = [f"Junction {c}" for c in dow_profile.columns]
        st.bar_chart(dow_profile, use_container_width=True)

    st.markdown("**Hour × Day heatmap (average vehicles)**")
    heat_mode = st.radio("Heatmap scope", ["Aggregate selected junctions", "Single junction"], horizontal=True)

    if heat_mode == "Single junction":
        heat_j = st.selectbox("Choose junction for heatmap", selected_junctions, key="heat_junction")
        heat_src = filtered[filtered["Junction"] == heat_j].copy()
        heat_title = f"Junction {heat_j}"
    else:
        heat_src = filtered.copy()
        heat_title = "Selected junctions (combined)"

    heat = heat_src.groupby(["DayName", "Hour"])["Vehicles"].mean().reset_index()
    heat["DayName"] = pd.Categorical(heat["DayName"], categories=dow_order, ordered=True)
    heat_pivot = heat.pivot(index="DayName", columns="Hour", values="Vehicles").sort_index().fillna(0)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(heat_pivot.values, aspect="auto")
    ax.set_title(f"Average Vehicles Heatmap — {heat_title}")
    ax.set_yticks(range(len(heat_pivot.index)))
    ax.set_yticklabels(heat_pivot.index)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Avg Vehicles")
    st.pyplot(fig, use_container_width=True)
#
with tab3:
    st.subheader("Junction comparison and distribution")

    c1, c2 = st.columns(2)
#
    with c1:
        st.markdown("**Distribution by junction (boxplot)**")
        fig_box, ax_box = plt.subplots(figsize=(6, 4))
        groups = [filtered.loc[filtered["Junction"] == j, "Vehicles"].values for j in selected_junctions]
        labels = [f"J{j}" for j in selected_junctions]
        ax_box.boxplot(groups, labels=labels, showfliers=False)
        ax_box.set_xlabel("Junction")
        ax_box.set_ylabel("Vehicles")
        ax_box.set_title("Traffic Distribution by Junction")
        st.pyplot(fig_box, use_container_width=True)
#
    with c2:
        st.markdown("**Histogram (all selected junctions)**")
        bins = st.slider("Histogram bins", min_value=5, max_value=60, value=20, key="hist_bins")
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        ax_hist.hist(filtered["Vehicles"], bins=bins)
        ax_hist.set_xlabel("Vehicles")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title("Distribution of Vehicle Counts")
        st.pyplot(fig_hist, use_container_width=True)
#
    st.markdown("**Junction summary table**")
    summary_tbl = (
        filtered.groupby("Junction")["Vehicles"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
        .rename(columns={
            "count": "Records",
            "mean": "Mean",
            "median": "Median",
            "std": "Std",
            "min": "Min",
            "max": "Max",
        })
    )
    st.dataframe(summary_tbl, use_container_width=True)

with tab4:
    st.subheader("Raw data, quality checks, and quick stats")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Missing values**")
        missing = filtered.isna().sum().rename("Missing Count")
        st.dataframe(missing.to_frame(), use_container_width=True)

    with c2:
        st.markdown("**Date coverage by junction**")
        coverage = filtered.groupby("Junction")["DateTime"].agg(["min", "max", "count"])
        st.dataframe(coverage, use_container_width=True)

    st.markdown("**Descriptive statistics**")
    desc = filtered[["Vehicles"]].describe().T.round(2)
    st.dataframe(desc, use_container_width=True)

    if show_raw:
        st.markdown("**Raw data preview**")
        st.dataframe(filtered.head(500), use_container_width=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv_bytes,
        file_name="traffic_filtered.csv",
        mime="text/csv",
    )