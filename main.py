import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
import os
import io
from pathlib import Path

matplotlib.use("Agg")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Let's Eat",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: 1px solid #2d2d4e;
}
[data-testid="stSidebar"] * {
    color: #e8e8f0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
    padding: 6px 0;
}
[data-testid="stSidebar"] h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem !important;
    color: #f0a500 !important;
    margin-bottom: 0.2rem;
}
[data-testid="stSidebar"] .subtitle {
    font-size: 0.75rem;
    color: #9090b0 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Main area */
.main .block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Page heading */
.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #1a1a2e;
    margin-bottom: 0.15rem;
    line-height: 1.1;
}
.page-sub {
    font-size: 0.85rem;
    color: #7070a0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Cards / panels */
.card {
    background: #ffffff;
    border: 1px solid #e2e2ee;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 12px rgba(26,26,46,0.06);
}
.card h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #1a1a2e;
    margin: 0 0 0.8rem 0;
}

/* Accent button overrides */
.stButton > button {
    background: #1a1a2e !important;
    color: #f0a500 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.5rem !important;
    transition: background 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
    background: #f0a500 !important;
    color: #1a1a2e !important;
    transform: translateY(-1px) !important;
}

/* Download button */
.stDownloadButton > button {
    background: #f0a500 !important;
    color: #1a1a2e !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.3rem !important;
}
.stDownloadButton > button:hover {
    background: #d4920a !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #f7f7fc;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    border: 1px solid #e2e2ee;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif;
    color: #1a1a2e !important;
}

/* Dividers */
hr { border-color: #e2e2ee; }

/* Success / info */
.stSuccess { border-radius: 8px; }
.stInfo    { border-radius: 8px; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #c0c0d8 !important;
    border-radius: 10px !important;
    background: #f9f9fc !important;
}

/* Dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1>Let's Eat</h1>", unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Donation Analytics</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["📈 Demand Prediction", "🗺️ Heatmap Generator"],
        label_visibility="collapsed",
    )
    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS  (identical logic to original, no tkinter)
# ══════════════════════════════════════════════════════════════════════════════

def string_to_float(number):
    if isinstance(number, (float, int)):
        return number
    if "," in str(number):
        return float(str(number).replace(",", ""))
    return float(number)


def convert_to_df(filepath_or_buffer):
    try:
        df = pd.read_csv(filepath_or_buffer)
    except Exception:
        return None, "Improper input – could not read CSV."
    required = ["DATE", "ZIPCODE", "DONATION"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"
    df["Given"] = df["DONATION"].map(string_to_float)
    df["DONATION"] = df["DONATION"].map(string_to_float)
    df = df[["DATE", "ZIPCODE", "Given"]]
    return None, df.groupby(["DATE", "ZIPCODE"])["Given"].apply(np.sum).reset_index()


def best_regression_model(K, Y):
    reshape_X = K.reshape(-1, 1)
    best_r2 = -np.inf
    best_equation = None

    # Linear
    lm = LinearRegression().fit(reshape_X, Y)
    r2 = r2_score(Y, lm.predict(reshape_X))
    if r2 > best_r2:
        best_r2 = r2
        best_equation = (lambda x, m=lm: m.coef_[0] * x + m.intercept_, r2)

    # Quadratic
    qt = PolynomialFeatures(degree=2)
    Xq = qt.fit_transform(reshape_X)
    qm = LinearRegression().fit(Xq, Y)
    r2 = r2_score(Y, qm.predict(Xq))
    if r2 > best_r2:
        best_r2 = r2
        best_equation = (
            lambda x, m=qm: m.coef_[2] * x**2 + m.coef_[1] * x + m.intercept_, r2)

    # Exponential
    try:
        Y_log = np.log(Y)
        em = LinearRegression().fit(reshape_X, Y_log)
        r2 = r2_score(Y, np.exp(em.predict(reshape_X)))
        if r2 > best_r2:
            best_r2 = r2
            best_equation = (
                lambda x, m=em: math.exp(m.intercept_) * math.exp(m.coef_[0] * x), r2)
    except Exception:
        pass

    # Logarithmic
    X_log = np.log(K.astype(float) + 1e-6).reshape(-1, 1)
    X_log[0] = -100
    lgm = LinearRegression().fit(X_log, Y)
    r2 = r2_score(Y, lgm.predict(X_log))
    if r2 > best_r2:
        best_r2 = r2
        best_equation = (
            lambda x, m=lgm: m.coef_[0] * math.log(x) + m.intercept_, r2)

    return best_equation


def run_equation(row, predict_year, min_year):
    predict_year = predict_year - min_year
    given_values = list(row.values)
    eq_tuple = best_regression_model(
        deepcopy(np.array(list(row.keys()))),
        deepcopy(np.array(given_values)),
    )
    equation, score = eq_tuple
    if score > 0.4:
        return max(round(equation(predict_year)), 0)
    else:
        return round(np.mean(np.array(given_values[-2:])))


def convert_key_to_year(item, min_year):
    try:
        return int(item) + min_year
    except Exception:
        return item


def regression_analysis(df, year):
    df = df.set_index("ZIPCODE")
    df.index = np.array(df.index).astype("U5").astype(str)
    df.index.name = "ZIPCODE"
    df = df[["DATE", "Given"]]
    min_year = int(min(df["DATE"]))
    df["DATE"] = df["DATE"] - min_year
    combined_df = None

    for index in list(set(df.index)):
        test_df = df.loc[index]
        if isinstance(test_df, pd.Series):
            test_df = pd.DataFrame(test_df).transpose()
        placeholder_dict = {}
        for j in range(int(max(df["DATE"])) + 1):
            subset = test_df.loc[test_df["DATE"] == j]
            if len(subset) > 1:
                placeholder_dict[j] = [float(subset["Given"].sum())]
            else:
                try:
                    placeholder_dict[j] = [float(subset["Given"].values[0])]
                except Exception:
                    placeholder_dict[j] = [0]
        final_df = pd.DataFrame.from_dict(placeholder_dict)
        final_df.index = [index]
        combined_df = final_df if combined_df is None else pd.concat([combined_df, final_df])

    del df
    combined_df = combined_df.replace(0, 0.1)
    combined_df[f"PREDICTION {year}"] = combined_df.apply(
        run_equation, args=(year, min_year), axis=1
    )
    column_dict = {
        k: convert_key_to_year(k, min_year) for k in combined_df.keys()
    }
    combined_df.rename(columns=column_dict, inplace=True)
    combined_df.replace(0.1, 0, inplace=True)
    return combined_df, f"Prediction for {year}"


def process_year(string):
    string = string.strip()
    try:
        return [int(string)]
    except ValueError:
        pass
    if "-" not in string:
        return None
    parts = string.split("-")
    try:
        lo, hi = int(parts[0]), int(parts[1])
        return list(range(lo, hi + 1))
    except Exception:
        return None


def given_map(number):
    if number == 0:   return "0"
    if number < 10:   return "1"
    if number < 50:   return "2"
    if number < 200:  return "3"
    if number < 1000: return "4"
    return "5"


def given_map3(number):
    if number == 0:    return "0"
    if number < 100:   return "1"
    if number < 500:   return "2"
    if number < 1000:  return "3"
    if number < 2500:  return "4"
    return "5"


def county_convert(county):
    return (county.rsplit(" ", 1)[0]).upper()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – DEMAND PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "📈 Demand Prediction":
    st.markdown('<div class="page-title">Demand Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Forecast donation volumes by zip code</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        pred_year = st.number_input(
            "Prediction Year", min_value=2018, max_value=3000,
            value=2025, step=1, help="Year to forecast donations for."
        )
        uploaded = st.file_uploader(
            "Upload Donation CSV",
            type=["csv"],
            help="CSV must contain columns: DATE, ZIPCODE, DONATION",
        )
        run_btn = st.button("▶  Run Prediction", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="card"><h3>Expected Format</h3>'
            '<code>DATE, ZIPCODE, DONATION</code><br>'
            '<small style="color:#7070a0">One row per donation record. '
            'DATE should be a 4-digit year (e.g. 2022).</small></div>',
            unsafe_allow_html=True,
        )

    with col_right:
        if run_btn:
            if uploaded is None:
                st.error("Please upload a CSV file first.")
            else:
                with st.spinner("Running regression analysis…"):
                    _, df = convert_to_df(uploaded)
                    if isinstance(df, str):
                        st.error(df)
                    else:
                        results, label = regression_analysis(df, pred_year)

                        st.success(f"✓ {label} complete — {len(results):,} zip codes analysed")

                        # ── Summary metrics ──────────────────────────────────
                        pred_col = f"PREDICTION {pred_year}"
                        total_pred = int(results[pred_col].sum())
                        max_zip = results[pred_col].idxmax()
                        max_val = int(results[pred_col].max())

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Predicted Items", f"{total_pred:,}")
                        m2.metric("Highest-Demand ZIP", max_zip)
                        m3.metric("Highest-Demand Count", f"{max_val:,}")

                        st.markdown("---")

                        # ── RAW results table ────────────────────────────────
                        st.markdown("**Full Prediction Table**")
                        st.dataframe(results, use_container_width=True, height=380)

                        # ── FINAL summary (sum per column) ───────────────────
                        st.markdown("**Summary — Total Items per Year**")
                        easy = round(results.sum()).rename("Total Items")
                        st.dataframe(
                            easy.to_frame().T, use_container_width=True
                        )

                        # ── Bar chart of top 20 predicted ZIPs ───────────────
                        st.markdown("**Top 20 ZIP Codes by Predicted Demand**")
                        top20 = results[pred_col].nlargest(20).reset_index()
                        top20.columns = ["ZIP Code", "Predicted Items"]
                        fig_bar, ax_bar = plt.subplots(figsize=(8, 3.5))
                        ax_bar.barh(
                            top20["ZIP Code"].astype(str),
                            top20["Predicted Items"],
                            color="#1a1a2e", edgecolor="none"
                        )
                        ax_bar.invert_yaxis()
                        ax_bar.set_xlabel("Predicted Items", fontsize=9)
                        ax_bar.set_title(
                            f"Top 20 ZIP Codes – Predicted Demand {pred_year}",
                            fontsize=10, fontweight="bold"
                        )
                        ax_bar.spines[["top", "right"]].set_visible(False)
                        ax_bar.tick_params(labelsize=8)
                        plt.tight_layout()
                        st.pyplot(fig_bar)
                        plt.close(fig_bar)

                        # ── Downloads ────────────────────────────────────────
                        st.markdown("---")
                        dl1, dl2 = st.columns(2)
                        raw_csv = results.to_csv().encode()
                        final_csv = easy.to_frame().to_csv().encode()
                        dl1.download_button(
                            "⬇ Download RAW Predictions CSV",
                            data=raw_csv,
                            file_name=f"RAW_Prediction_{pred_year}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                        dl2.download_button(
                            "⬇ Download FINAL Summary CSV",
                            data=final_csv,
                            file_name=f"FINAL_Prediction_{pred_year}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:340px;background:#f7f7fc;border-radius:12px;'
                'border:1px dashed #c0c0d8;color:#9090b0;font-size:0.9rem;">'
                "</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – HEATMAP GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
else:
    # ── Local shapefile paths (same structure as the original tkinter app) ────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ZIP_SHP_PATH = os.path.join(BASE_DIR, "ZIPCODE", "ZIPCODE.shp")
    COUNTY_SHP_PATH = os.path.join(BASE_DIR, "county", "COUNTY_BOUNDARY.shp")
    COUNTY_CSV_PATH = os.path.join(BASE_DIR, "counties.csv")

    st.markdown('<div class="page-title">Heatmap Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Geographic donation density across Maryland</div>', unsafe_allow_html=True)

    missing_files = []

    if not os.path.exists(ZIP_SHP_PATH):
        missing_files.append(f"`zipShapes/oklahoma-zip-code-boundaries.shp`")
    if not os.path.exists(COUNTY_SHP_PATH):
        missing_files.append(f"`county/COUNTY_BOUNDARY.shp`")
    if not os.path.exists(COUNTY_CSV_PATH):
        missing_files.append(f"`counties.csv`")
    if missing_files:
        st.warning(
            "The following local files were not found next to the app"
            "county heatmaps may be unavailable:\n\n" + "\n\n".join(missing_files)
        )

    col_left, col_right = st.columns([1, 2.2], gap="large")

    with col_left:
        year_input = st.text_input(
            "Year or Range",
            value="2022",
            help="Single year (e.g. 2022) or range (e.g. 2019-2022)",
        )
        uploaded_map = st.file_uploader(
            "Upload Donation CSV",
            type=["csv"],
            key="map_upload",
            help="CSV must contain columns: DATE, ZIPCODE, DONATION",
        )
        gen_btn = st.button("🗺  Generate Heatmaps", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="card"><h3>Expected Format</h3>'
            '<code>DATE, ZIPCODE, DONATION</code><br>'
            '<small style="color:#7070a0">Shapefiles are read automatically from the '
            'app directory (<code>zipShapes/</code>, <code>county/</code>, '
            '<code>counties.csv</code>).</small></div>',
            unsafe_allow_html=True,
        )

    with col_right:
        if gen_btn:
            year_list = process_year(year_input)
            if year_list is None:
                st.error("Enter a valid year or range, e.g. 2022 or 2019-2022.")
            elif uploaded_map is None:
                st.error("Please upload the donation CSV.")
            elif not os.path.exists(ZIP_SHP_PATH):
                st.error(
                    f"ZIP code shapefile not found at `{ZIP_SHP_PATH}`. "
                    "Place the shapefile folder next to the app and restart."
                )
            else:
                with st.spinner("Building heatmaps…"):

                    # ── Load donation data ───────────────────────────────────
                    df = pd.read_csv(uploaded_map, index_col=False)
                    if isinstance(df, str):
                        st.error(df)
                        st.stop()

                    df = df[df["DATE"].isin(year_list)].copy()
                    df.drop("DATE", axis=1, inplace=True)
                    df["ZIPCODE"] = df["ZIPCODE"].astype(int).astype(str)
                    print(df.keys())
                    # Sum donations per ZIP across all selected years
                    df = df.groupby("ZIPCODE")["DONATION"].sum().reset_index()

                    try:
                        # ── Load ZIP shapefile from local disk ───────────────
                        zip_shapes = gpd.read_file(ZIP_SHP_PATH)

                        # Detect the ZIP column name in the shapefile
                        zcol = next(
                            (c for c in zip_shapes.columns
                             if "zcta" in c.lower() or "zip" in c.lower()),
                            zip_shapes.columns[0],
                        )
                        zip_shapes[zcol] = zip_shapes[zcol].astype(str)

                        # Fill in ZIPs present in shapefile but absent from donations with 0
                        non_code = zip_shapes[zcol][~zip_shapes[zcol].isin(df["ZIPCODE"])].values
                        filled = pd.DataFrame(
                            list(zip(non_code, np.zeros(len(non_code)))),
                            columns=["ZIPCODE", "DONATION"],
                        )
                        df = pd.concat([df, filled], axis=0, ignore_index=True)


                        # Apply the categorical bucket mapping (mirrors given_map2 from Colab)
                        def given_map(number):
                            if number == 0:
                                return "0"
                            elif number < 10:
                                return "1"
                            elif number < 50:
                                return "2"
                            elif number < 200:
                                return "3"
                            elif number < 1000:
                                return "4"
                            return "5"


                        df["Adjusted"] = df["DONATION"].map(given_map)

                        # Left-merge shapefile with donation data (matches Colab's how="left")
                        heatmap_df = pd.merge(
                            zip_shapes, df,
                            left_on=zcol, right_on="ZIPCODE",
                            how="left"
                        )
                        # Fill any unmatched ZIPs with category "0"
                        heatmap_df["Adjusted"] = heatmap_df["Adjusted"].fillna("0")

                        fig1, ax1 = plt.subplots(figsize=(9, 4))
                        heatmap_df.plot(
                            legend=True,
                            column="Adjusted",
                            legend_kwds={
                                "loc": "lower left",
                                "bbox_to_anchor": (0, 0.1),
                                "markerscale": 1.29,
                                "title_fontsize": "medium",
                                "fontsize": "small",
                            },
                            cmap="plasma",
                            ax=ax1,
                        )
                        leg = ax1.get_legend()
                        leg.set_title("Items Donated")
                        for i, t in enumerate(leg.get_texts()):
                            t.set_text(["0", "<10", "<50", "<200", "<1000", ">1000"][i])
                        ax1.axis("off")
                        ax1.set_title(
                            f"Let's Eat Donation Heatmap (ZIP) — {year_input}",
                            fontweight="bold"
                        )
                        plt.tight_layout()

                        st.markdown("**ZIP Code Heatmap**")
                        st.pyplot(fig1)

                        buf1 = io.BytesIO()
                        fig1.savefig(buf1, format="png", dpi=150, bbox_inches="tight")
                        buf1.seek(0)
                        plt.close(fig1)

                        st.download_button(
                            "Download ZIP Heatmap (PNG)",
                            data=buf1,
                            file_name=f"Heatmap_Zipcode_{year_input}.png",
                            mime="image/png",
                        )

                    except Exception as e:
                        st.error(f"Error generating heatmap: {e}")
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:420px;background:#f7f7fc;border-radius:12px;'
                'border:1px dashed #c0c0d8;color:#9090b0;font-size:0.9rem;">'
                "</div>",
                unsafe_allow_html=True,
            )