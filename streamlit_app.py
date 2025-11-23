import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import re

st.set_page_config(layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def sanitize_key(s: str) -> str:
    return re.sub(r"\W+", "_", s)

def fit_distribution(data, dist_name):
    if dist_name == "Normal (Gaussian)":
        params = stats.norm.fit(data)
        return stats.norm, params
    if dist_name == "Exponential":
        params = stats.expon.fit(data)
        return stats.expon, params
    if dist_name == "Uniform":
        params = stats.uniform.fit(data)
        return stats.uniform, params
    if dist_name == "Log-Normal":
        params = stats.lognorm.fit(data)
        return stats.lognorm, params
    if dist_name == "Gamma":
        params = stats.gamma.fit(data)
        return stats.gamma, params
    if dist_name == "Beta":
        params = stats.beta.fit(data)
        return stats.beta, params
    if dist_name == "Chi-Square":
        params = stats.chi2.fit(data)
        return stats.chi2, params
    if dist_name == "Weibull":
        params = stats.weibull_min.fit(data)
        return stats.weibull_min, params
    if dist_name == "Pareto":
        params = stats.pareto.fit(data)
        return stats.pareto, params
    if dist_name == "Poisson":
        data_int = np.round(data).astype(int)
        mu = float(np.mean(data_int))
        return stats.poisson, (mu,)
    if dist_name == "Binomial":
        data_int = np.round(data).astype(int)
        n = int(max(1, np.max(data_int)))
        p = float(np.mean(data_int) / n) if n > 0 else 0.0
        return stats.binom, (n, p)
    return None, None

PARAM_LABELS = {
    "Normal (Gaussian)": ["μ (mean)", "σ (std dev)"],
    "Exponential": ["loc", "scale"],
    "Uniform": ["loc", "scale"],
    "Log-Normal": ["shape (σ)", "loc", "scale"],
    "Gamma": ["shape (a)", "loc", "scale"],
    "Beta": ["a (α)", "b (β)", "loc", "scale"],
    "Chi-Square": ["df", "loc", "scale"],
    "Weibull": ["shape (c)", "loc", "scale"],
    "Pareto": ["shape (b)", "loc", "scale"],
    "Poisson": ["μ (mean)"],
    "Binomial": ["n (trials)", "p (probability)"],
}

def plot_distribution_over_hist(ax, dist_obj, params, data, curve_color="red"):
    if dist_obj in [stats.poisson, stats.binom]:
        data_min = int(np.min(np.round(data).astype(int)))
        data_max = int(np.max(np.round(data).astype(int)))
        x = np.arange(data_min, data_max + 1)
        try:
            pmf = dist_obj.pmf(x, *params)
            ax.plot(x, pmf, linestyle='-', linewidth=2, color=curve_color, marker='o', label="Adjusted fit")
            ax.legend()
        except Exception:
            pass
        return
    x = np.linspace(np.min(data), np.max(data), 300)
    try:
        pdf = dist_obj.pdf(x, *params)
        ax.plot(x, pdf, linestyle='-', linewidth=2, color=curve_color, label="Adjusted fit")
        ax.legend()
    except Exception:
        pass

DIST_LIST = [
    "Normal (Gaussian)", "Exponential", "Uniform", "Poisson", "Binomial",
    "Log-Normal", "Gamma", "Beta", "Chi-Square", "Weibull", "Pareto"
]

# ---------------------------
# Tabs
# ---------------------------
tab_manual, tab_upload = st.tabs(["Manual Input", "Upload a File"])

# ---------------------------
# Manual Input
# ---------------------------
with tab_manual:
    data_col, hist_col, sliders_col = st.columns([0.5, 1.0, 0.5])

    # --- Left: Data editor ---
    with data_col:
        st.subheader("Values")
        col_name = st.text_input("Column name:", "67", key="manual_colname")
        df = pd.DataFrame({col_name: [0.0, 0.0]})
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="manual_data_editor")
        data = edited_df[col_name].dropna().values if not edited_df.empty else np.array([])
        st.write(f"N = {len(data)}")

    if len(data) > 0:
        bins = min(30, max(5, int(len(data)/5)))

        # --- Middle: Histogram ---
        with hist_col:
            fig_placeholder = st.empty()
            dist_obj, fitted_params = fit_distribution(data, "Normal (Gaussian)")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(data, bins=bins, density=True, color="#87CEEB", alpha=0.5, edgecolor='black')
            ax.set_xlabel(col_name)
            ax.set_ylabel("Density")
            plot_distribution_over_hist(ax, dist_obj, fitted_params, data, curve_color="#FF0000")
            fig_placeholder.pyplot(fig)

            # Average error
            hist_vals, bin_edges = np.histogram(data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            fitted_vals = dist_obj.pdf(bin_centers, *fitted_params)
            avg_error = np.mean(np.abs(hist_vals - fitted_vals))
            st.markdown(f"**average error: {avg_error:.4f}**")

        # --- Right: Sliders, colors, distribution selector ---
        with sliders_col:
            st.subheader("Adjust Curve")
            col_band, col_curve = st.columns(2)
            band_color = col_band.color_picker("Bars color", "#87CEEB", key="manual_band_color")
            curve_color = col_curve.color_picker("Curve color", "#FF0000", key="manual_curve_color")

            dist_name = st.selectbox("Distribution:", DIST_LIST, key="manual_dist")
            dist_obj, fitted_params = fit_distribution(data, dist_name)

            labels = PARAM_LABELS.get(dist_name, [f"param_{i+1}" for i in range(len(fitted_params))])
            modified_params = []
            for i, (val, label) in enumerate(zip(fitted_params, labels)):
                key = f"manual_{dist_name}_{label}_{i}"
                min_val = float(val) * 0.5
                max_val = float(val) * 1.5 if val != 0 else 1.0
                step = max(abs(val) * 0.01, 0.001)
                new_val = st.slider(label, min_val, max_val, float(val), step=step, key=key)
                modified_params.append(new_val)

        # --- Update histogram + curve ---
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(data, bins=bins, density=True, color=band_color, alpha=0.5, edgecolor='black')
        ax.set_xlabel(col_name)
        ax.set_ylabel("Density")
        plot_distribution_over_hist(ax, dist_obj, tuple(modified_params), data, curve_color=curve_color)
        fig_placeholder.pyplot(fig)
        
# ---------------------------
# Upload CSV
# ---------------------------
with tab_upload:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="upload_file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        default_col_name = str(df.columns[0])

        # Top-level columns: Data editor | Histogram | Sliders
        data_col, hist_col, sliders_col = st.columns([0.5, 1.0, 0.5])

        # --- Left: Data editor + Column name input ---
        with data_col:
            st.subheader("Values")
            col_name_input = st.text_input("Column name:", default_col_name, key="upload_colname")
            df = df.rename(columns={df.columns[0]: col_name_input})
            safe_col = sanitize_key(col_name_input)

            edited_df = st.data_editor(
                df[[col_name_input]],
                num_rows="dynamic",
                use_container_width=True,
                key=f"upload_{safe_col}_data_editor"
            )
            data = edited_df[col_name_input].dropna().values if not edited_df.empty else np.array([])
            st.write(f"N = {len(data)}")

        if len(data) > 0:
            bins = min(30, max(5, int(len(data)/5)))

            # --- Middle: Histogram ---
            with hist_col:
                fig_placeholder = st.empty()
                dist_obj, fitted_params = fit_distribution(data, "Normal (Gaussian)")
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(data, bins=bins, density=True, color="#87CEEB", alpha=0.5, edgecolor='black')
                ax.set_xlabel(col_name_input)
                ax.set_ylabel("Density")
                plot_distribution_over_hist(ax, dist_obj, fitted_params, data, curve_color="#FF0000")
                fig_placeholder.pyplot(fig)

                # Average error
                hist_vals, bin_edges = np.histogram(data, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                fitted_vals = dist_obj.pdf(bin_centers, *fitted_params)
                avg_error = np.mean(np.abs(hist_vals - fitted_vals))
                st.markdown(f"**Fit quality (average error): {avg_error:.4f}**")

            # --- Right: Sliders, colors, distribution ---
            with sliders_col:
                st.subheader("Adjust Curve")
                col_band, col_curve = st.columns(2)
                band_color = col_band.color_picker("Bars color", "#87CEEB", key=f"upload_{safe_col}_band_color")
                curve_color = col_curve.color_picker("Curve color", "#FF0000", key=f"upload_{safe_col}_curve_color")

                dist_name = st.selectbox("Distribution:", DIST_LIST, key=f"upload_{safe_col}_dist")
                dist_obj, fitted_params = fit_distribution(data, dist_name)

                labels = PARAM_LABELS.get(dist_name, [f"param_{i+1}" for i in range(len(fitted_params))])
                modified_params = []
                for i, (val, label) in enumerate(zip(fitted_params, labels)):
                    key = f"upload_{safe_col}_{dist_name}_{label}_{i}"
                    min_val = float(val) * 0.5
                    max_val = float(val) * 1.5 if val != 0 else 1.0
                    step = max(abs(val) * 0.01, 0.001)
                    new_val = st.slider(label, min_val, max_val, float(val), step=step, key=key)
                    modified_params.append(new_val)

            # --- Update histogram + curve with modified parameters ---
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(data, bins=bins, density=True, color=band_color, alpha=0.5, edgecolor='black')
            ax.set_xlabel(col_name_input)
            ax.set_ylabel("Density")
            plot_distribution_over_hist(ax, dist_obj, tuple(modified_params), data, curve_color=curve_color)
            fig_placeholder.pyplot(fig)
