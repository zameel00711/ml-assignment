"""Assignment 2: UCI Bike Sharing Demand Prediction Pipeline, 5-Fold CV, Analytics & Simulator."""

import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from modules.pdf_generator import build_a4_pdf_report
from utils.ui_theme import apply_plot_theme

DATA_PATH = "data/hour.csv"
PLOTS_DIR = "evaluation_plots"

FEATURE_COLS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]


@st.cache_data
def load_bike_dataset() -> pd.DataFrame:
    """Loads and formats the UCI Bike Sharing hourly dataset."""
    if not os.path.exists(DATA_PATH):
        import io
        import urllib.request
        import zipfile

        url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
        os.makedirs("data", exist_ok=True)
        with urllib.request.urlopen(url, timeout=20) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            z.extract("hour.csv", "data")
            z.extract("day.csv", "data")

    df = pd.read_csv(DATA_PATH)
    return df


def prepare_data_split(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
) -> dict[str, Any]:
    """Splits features and target into train/test sets and fits scaler."""
    X = df[FEATURE_COLS].copy()
    y = df["cnt"].to_numpy()

    rng = np.random.default_rng(random_state)
    n_samples = len(df)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    split_idx = int(n_samples * (1.0 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train_df = X.iloc[train_idx]
    X_test_df = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    return {
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "test_df": df.iloc[test_idx].copy(),
    }


def get_candidate_models() -> dict[str, Any]:
    """Returns candidate regression models for comparison."""
    return {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=14, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=100, max_depth=14, random_state=42, n_jobs=-1
        ),
        "Ridge Linear Regression": Ridge(alpha=10.0),
        "K-Nearest Neighbors (KNN)": KNeighborsRegressor(n_neighbors=7, weights="distance"),
    }


def execute_five_fold_cv(
    models: dict[str, Any],
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Runs 5-Fold Cross-Validation on training set and evaluates on held-out test set."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_rows = []
    trained_artifacts = {}

    for name, model in models.items():
        cv_r2_scores = []
        cv_rmse_scores = []
        cv_mae_scores = []

        for train_fold_idx, val_fold_idx in kf.split(X_train_scaled, y_train):
            X_tr, y_tr = X_train_scaled[train_fold_idx], y_train[train_fold_idx]
            X_val, y_val = X_train_scaled[val_fold_idx], y_train[val_fold_idx]

            fold_model = type(model)(**model.get_params())
            fold_model.fit(X_tr, y_tr)
            val_preds = fold_model.predict(X_val)

            val_r2 = r2_score(y_val, val_preds)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            val_mae = float(mean_absolute_error(y_val, val_preds))

            cv_r2_scores.append(val_r2)
            cv_rmse_scores.append(val_rmse)
            cv_mae_scores.append(val_mae)

        t0 = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        fit_time = round(time.perf_counter() - t0, 3)

        test_preds = model.predict(X_test_scaled)
        test_r2 = float(r2_score(y_test, test_preds))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
        test_mae = float(mean_absolute_error(y_test, test_preds))
        test_mape = float(np.mean(np.abs((y_test - test_preds) / np.maximum(y_test, 1)))) * 100

        benchmark_rows.append(
            {
                "Model": name,
                "CV R2 Mean": round(float(np.mean(cv_r2_scores)), 4),
                "CV R2 Std": round(float(np.std(cv_r2_scores)), 4),
                "CV RMSE Mean": round(float(np.mean(cv_rmse_scores)), 2),
                "CV RMSE Std": round(float(np.std(cv_rmse_scores)), 2),
                "CV MAE Mean": round(float(np.mean(cv_mae_scores)), 2),
                "Test R2": round(test_r2, 4),
                "Test RMSE": round(test_rmse, 2),
                "Test MAE": round(test_mae, 2),
                "Test MAPE (%)": round(test_mape, 1),
                "Fit Time (s)": fit_time,
            }
        )

        trained_artifacts[name] = {
            "model": model,
            "test_preds": test_preds,
            "cv_r2_scores": cv_r2_scores,
            "cv_rmse_scores": cv_rmse_scores,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        }

    bench_df = pd.DataFrame(benchmark_rows).sort_values("CV R2 Mean", ascending=False).reset_index(drop=True)
    best_model_name = bench_df.iloc[0]["Model"]
    return bench_df, trained_artifacts, best_model_name


def save_and_generate_plots(
    df: pd.DataFrame,
    bench_df: pd.DataFrame,
    best_model_name: str,
    best_model: Any,
    test_preds: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
) -> dict[str, str]:
    """Generates all 6 evaluation plots and saves them to evaluation_plots/ directory."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    apply_plot_theme()
    paths = {}

    # 1. Model Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {m: ("#2563EB" if m == best_model_name else "#94A3B8") for m in bench_df["Model"]}

    sns.barplot(data=bench_df, x="CV R2 Mean", y="Model", hue="Model", palette=colors, legend=False, ax=ax1)
    ax1.set_title("5-Fold Cross-Validation R2 Score", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Mean R2")
    for i, v in enumerate(bench_df["CV R2 Mean"]):
        ax1.text(v - 0.08, i, f"{v:.4f}", color="white", fontweight="bold", va="center")

    sns.barplot(data=bench_df, x="CV RMSE Mean", y="Model", hue="Model", palette=colors, legend=False, ax=ax2)
    ax2.set_title("5-Fold Cross-Validation RMSE (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Mean RMSE")
    for i, v in enumerate(bench_df["CV RMSE Mean"]):
        ax2.text(v + 2, i, f"{v:.1f}", color="#1E293B", va="center")

    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["model_comparison"] = p1

    # 2. Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(y_test, test_preds, alpha=0.3, color="#2563EB", edgecolors="none", s=18)
    max_val = max(float(y_test.max()), float(test_preds.max()))
    ax.plot([0, max_val], [0, max_val], "--", color="#DC2626", linewidth=2, label="Identity Line (y = x)")
    r2_val = r2_score(y_test, test_preds)
    rmse_val = np.sqrt(mean_squared_error(y_test, test_preds))
    ax.set_title(f"Actual vs Predicted Demand ({best_model_name})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual Hourly Demand (cnt)")
    ax.set_ylabel("Predicted Demand")
    ax.text(
        0.05,
        0.88,
        f"R2 = {r2_val:.4f}\nRMSE = {rmse_val:.1f}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
        fontsize=9,
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["actual_vs_predicted"] = p2

    # 3. Residuals Diagnostics
    residuals = y_test - test_preds
    fig, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_r1.scatter(test_preds, residuals, alpha=0.3, color="#0284C7", s=16)
    ax_r1.axhline(0, color="#DC2626", linestyle="--", linewidth=1.5)
    ax_r1.set_title("Residuals vs Predicted Values", fontsize=11, fontweight="bold")
    ax_r1.set_xlabel("Predicted Values")
    ax_r1.set_ylabel("Residual (Actual - Predicted)")

    sns.histplot(residuals, kde=True, color="#2563EB", ax=ax_r2, bins=35)
    ax_r2.axvline(0, color="#DC2626", linestyle="--", linewidth=1.5)
    ax_r2.set_title("Residual Error Distribution", fontsize=11, fontweight="bold")
    ax_r2.set_xlabel("Residual Error")
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "residuals_distribution.png")
    fig.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["residuals_distribution"] = p3

    # 4. Feature Importance
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": fi}).sort_values(
            "Importance", ascending=True
        )
        sns.barplot(data=fi_df, x="Importance", y="Feature", hue="Feature", palette="Blues_d", legend=False, ax=ax)
        ax.set_title(f"Feature Importance ({best_model_name})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance Weight")
    else:
        ax.text(0.5, 0.5, "Linear model does not provide tree importances", ha="center")
    plt.tight_layout()
    p4 = os.path.join(PLOTS_DIR, "feature_importance.png")
    fig.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["feature_importance"] = p4

    # 5. Diurnal Hourly Demand Profile
    eval_df = test_df.copy()
    eval_df["pred"] = test_preds
    hourly_agg = eval_df.groupby("hr")[["cnt", "pred"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hourly_agg["hr"], hourly_agg["cnt"], marker="o", color="#0284C7", linewidth=2, label="Actual Average")
    ax.plot(hourly_agg["hr"], hourly_agg["pred"], marker="s", linestyle="--", color="#2563EB", linewidth=2, label="Predicted Average")
    ax.axvspan(7.5, 9.5, color="#FEF08A", alpha=0.35, label="Morning Peak (8-9 AM)")
    ax.axvspan(16.5, 18.5, color="#FDE047", alpha=0.35, label="Evening Peak (5-6 PM)")
    ax.set_title("Hourly Demand Curve: Actual vs Predicted", fontsize=11, fontweight="bold")
    ax.set_xlabel("Hour of Day (0 to 23)")
    ax.set_ylabel("Average Rentals")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper left")
    plt.tight_layout()
    p5 = os.path.join(PLOTS_DIR, "hourly_demand_trend.png")
    fig.savefig(p5, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["hourly_demand_trend"] = p5

    # 6. Correlation Matrix
    corr_cols = ["cnt", "temp", "atemp", "hum", "windspeed", "hr", "season", "workingday"]
    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p6 = os.path.join(PLOTS_DIR, "correlation_matrix.png")
    fig.savefig(p6, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["correlation_matrix"] = p6

    return paths


def return_to_home() -> None:
    """Callback to return to the home screen."""
    st.session_state["nav_selection"] = "Home"


def render_bike_sharing_app() -> None:
    """Renders the Bike Sharing Demand Prediction application."""
    col_back, _ = st.columns([1, 5])
    with col_back:
        st.button("<- Back to Home", on_click=return_to_home, key="btn_back_home_bike")

    st.title("Bike Sharing Demand Prediction")
    st.write(
        "A machine learning regression project using the UCI Bike Sharing Dataset. "
        "The system evaluates multiple regressors with 5-fold cross-validation, saves evaluation plots, "
        "generates an A4 PDF report, and provides an interactive prediction interface."
    )

    df = load_bike_dataset()

    tab_eda, tab_train, tab_plots, tab_pdf, tab_predict = st.tabs(
        [
            "1. Overview & Data Exploration",
            "2. Model Training & 5-Fold CV",
            "3. Evaluation Plots",
            "4. A4 PDF Report",
            "5. Demand Predictor",
        ]
    )

    if "pipeline_ready" not in st.session_state:
        st.session_state["pipeline_ready"] = False

    # ---------------- TAB 1: OVERVIEW & DATA EXPLORATION ----------------
    with tab_eda:
        st.subheader("Problem Overview")
        st.write(
            "Bike-sharing systems provide short-term bicycle rentals across urban station networks. "
            "Because usage varies substantially depending on the time of day, weather conditions, and seasons, "
            "predicting demand is critical for fleet rebalancing and dock management."
        )

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Observations", f"{len(df):,}")
        with col_stat2:
            st.metric("Feature Count", len(FEATURE_COLS))
        with col_stat3:
            st.metric("Average Hourly Rentals", f"{df['cnt'].mean():.1f}")
        with col_stat4:
            st.metric("Maximum Hourly Rentals", f"{df['cnt'].max()}")

        st.divider()
        st.subheader("Data Inspection")
        eda_view = st.radio(
            "Select view:",
            ["Dataset Sample and Statistics", "Hourly Distribution", "Seasonal and Weather Trends"],
            horizontal=True,
        )

        if eda_view == "Dataset Sample and Statistics":
            st.dataframe(df.head(10), use_container_width=True)
            st.write("Summary Statistics for Selected Numerical Columns:")
            st.dataframe(df[["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]].describe().T)
        elif eda_view == "Hourly Distribution":
            apply_plot_theme()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(data=df, x="hr", y="cnt", color="#3B82F6", ax=ax, fliersize=2)
            ax.set_title("Hourly Distribution of Rentals", fontsize=11, fontweight="bold")
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Rental Count")
            st.pyplot(fig)
            plt.close(fig)
        else:
            apply_plot_theme()
            fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(11, 4))
            season_names = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
            df_plot = df.copy()
            df_plot["season_name"] = df_plot["season"].map(season_names)
            sns.barplot(data=df_plot, x="season_name", y="cnt", palette="Blues_d", ax=ax_s)
            ax_s.set_title("Average Demand by Season", fontweight="bold")
            ax_s.set_ylabel("Mean Hourly Rentals")

            weather_names = {1: "Clear", 2: "Mist", 3: "Light Rain/Snow", 4: "Heavy Storm"}
            df_plot["weather_name"] = df_plot["weathersit"].map(weather_names)
            sns.barplot(data=df_plot, x="weather_name", y="cnt", palette="Blues_d", ax=ax_w)
            ax_w.set_title("Average Demand by Weather Condition", fontweight="bold")
            ax_w.set_ylabel("Mean Hourly Rentals")
            st.pyplot(fig)
            plt.close(fig)

    # ---------------- TAB 2: MODEL TRAINING & 5-FOLD CV ----------------
    with tab_train:
        st.subheader("Model Training and Cross-Validation")
        st.write(
            "Trains candidate regression models and performs 5-fold cross-validation on an 80% training "
            "partition, then evaluates each model on the remaining 20% test partition."
        )

        train_col1, train_col2 = st.columns([1, 2])
        with train_col1:
            test_size = st.slider("Test Split Proportion", 0.10, 0.30, 0.20, 0.05)
            random_seed = st.number_input("Random State Seed", value=42, step=1)
            run_btn = st.button("Run Training and 5-Fold Cross-Validation", type="primary", use_container_width=True)

        with train_col2:
            st.write("**Models Included in Benchmark:**")
            st.write("1. Random Forest Regressor (100 trees, max depth 14)")
            st.write("2. Gradient Boosting Regressor (100 estimators, learning rate 0.1)")
            st.write("3. Extra Trees Regressor (100 trees, max depth 14)")
            st.write("4. Ridge Linear Regression (L2 regularization)")
            st.write("5. K-Nearest Neighbors Regressor (k=7, distance-weighted)")

        if run_btn:
            with st.spinner("Training models and running 5-fold cross-validation..."):
                split_data = prepare_data_split(df, test_size=test_size, random_state=random_seed)
                candidate_models = get_candidate_models()
                bench_df, artifacts, best_name = execute_five_fold_cv(
                    models=candidate_models,
                    X_train_scaled=split_data["X_train_scaled"],
                    y_train=split_data["y_train"],
                    X_test_scaled=split_data["X_test_scaled"],
                    y_test=split_data["y_test"],
                )

                best_model = artifacts[best_name]["model"]
                test_preds = artifacts[best_name]["test_preds"]

                plot_paths = save_and_generate_plots(
                    df=df,
                    bench_df=bench_df,
                    best_model_name=best_name,
                    best_model=best_model,
                    test_preds=test_preds,
                    y_test=split_data["y_test"],
                    test_df=split_data["test_df"],
                )

                st.session_state["pipeline_ready"] = True
                st.session_state["bench_df"] = bench_df
                st.session_state["artifacts"] = artifacts
                st.session_state["best_name"] = best_name
                st.session_state["plot_paths"] = plot_paths
                st.session_state["split_data"] = split_data

        if st.session_state.get("pipeline_ready", False):
            bench_df = st.session_state["bench_df"]
            best_name = st.session_state["best_name"]

            st.write(f"**Selected Best Model:** {best_name}")

            m1, m2, m3, m4 = st.columns(4)
            best_row = bench_df.iloc[0]
            with m1:
                st.metric("CV R2 Score (Mean)", f"{best_row['CV R2 Mean']:.4f}")
            with m2:
                st.metric("CV RMSE (Mean)", f"{best_row['CV RMSE Mean']:.2f}")
            with m3:
                st.metric("Test R2 Score", f"{best_row['Test R2']:.4f}")
            with m4:
                st.metric("Test RMSE", f"{best_row['Test RMSE']:.2f}")

            st.subheader("Model Comparison Table")
            st.dataframe(bench_df, use_container_width=True)
        else:
            st.info("Click 'Run Training and 5-Fold Cross-Validation' above to execute the model benchmark.")

    # ---------------- TAB 3: EVALUATION PLOTS ----------------
    with tab_plots:
        st.subheader("Evaluation Visualizations")
        if not st.session_state["pipeline_ready"]:
            st.info("Train the models in Tab 2 to view evaluation plots.")
        else:
            plot_paths = st.session_state["plot_paths"]

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.write("**1. 5-Fold Cross-Validation Comparison**")
                st.image(plot_paths["model_comparison"], use_container_width=True)
                with open(plot_paths["model_comparison"], "rb") as f:
                    st.download_button("Download Plot", f.read(), "model_comparison.png", "image/png", key="dl_p1")

            with p_col2:
                st.write("**2. Actual vs Predicted Values**")
                st.image(plot_paths["actual_vs_predicted"], use_container_width=True)
                with open(plot_paths["actual_vs_predicted"], "rb") as f:
                    st.download_button("Download Plot", f.read(), "actual_vs_predicted.png", "image/png", key="dl_p2")

            p_col3, p_col4 = st.columns(2)
            with p_col3:
                st.write("**3. Residual Error Diagnostics**")
                st.image(plot_paths["residuals_distribution"], use_container_width=True)
                with open(plot_paths["residuals_distribution"], "rb") as f:
                    st.download_button("Download Plot", f.read(), "residuals_distribution.png", "image/png", key="dl_p3")

            with p_col4:
                st.write("**4. Hourly Demand Curve**")
                st.image(plot_paths["hourly_demand_trend"], use_container_width=True)
                with open(plot_paths["hourly_demand_trend"], "rb") as f:
                    st.download_button("Download Plot", f.read(), "hourly_demand_trend.png", "image/png", key="dl_p4")

            p_col5, p_col6 = st.columns(2)
            with p_col5:
                st.write("**5. Feature Importance**")
                st.image(plot_paths["feature_importance"], use_container_width=True)
                with open(plot_paths["feature_importance"], "rb") as f:
                    st.download_button("Download Plot", f.read(), "feature_importance.png", "image/png", key="dl_p5")

            with p_col6:
                st.write("**6. Correlation Heatmap**")
                st.image(plot_paths["correlation_matrix"], use_container_width=True)
                with open(plot_paths["correlation_matrix"], "rb") as f:
                    st.download_button("Download Plot", f.read(), "correlation_matrix.png", "image/png", key="dl_p6")

    # ---------------- TAB 4: A4 PDF REPORT ----------------
    with tab_pdf:
        st.subheader("Generate A4 PDF Evaluation Report")
        st.write(
            "Compile an academic evaluation report containing problem context, feature table, "
            "cross-validation comparison metrics, embedded plots, and operational insights."
        )

        author_name = st.text_input("Author Name", value="Zameel")

        if not st.session_state["pipeline_ready"]:
            st.info("Train the models in Tab 2 to enable report compilation.")
        else:
            bench_df = st.session_state["bench_df"]
            best_name = st.session_state["best_name"]
            plot_paths = st.session_state["plot_paths"]

            best_metrics = {
                "cv_r2_mean": float(bench_df.iloc[0]["CV R2 Mean"]),
                "test_r2": float(bench_df.iloc[0]["Test R2"]),
                "test_rmse": float(bench_df.iloc[0]["Test RMSE"]),
            }

            pdf_bytes = build_a4_pdf_report(
                benchmark_df=bench_df,
                best_model_name=best_name,
                best_metrics=best_metrics,
                feature_importance_df=None,
                plot_paths=plot_paths,
                author_name=author_name,
            )

            st.download_button(
                "Download Evaluation Report (PDF)",
                data=pdf_bytes,
                file_name=f"Bike_Sharing_Assignment_Report_{author_name}.pdf",
                mime="application/pdf",
                type="primary",
            )
            st.caption(f"Report compiled. File size: {len(pdf_bytes) / 1024:.1f} KB.")

    # ---------------- TAB 5: DEMAND PREDICTOR ----------------
    with tab_predict:
        st.subheader("Interactive Demand Predictor")
        st.write(
            "Adjust calendar and weather inputs to compute real-time demand forecasts using the "
            f"selected model ({st.session_state.get('best_name', 'Random Forest Regressor')})."
        )

        sim_col1, sim_col2, sim_col3 = st.columns(3)

        with sim_col1:
            st.write("**Calendar Parameters**")
            preset = st.selectbox(
                "Preset Scenario:",
                [
                    "Custom",
                    "Morning Commute (8 AM, Weekday, Spring)",
                    "Evening Commute (5 PM, Weekday, Summer)",
                    "Weekend Afternoon (2 PM, Summer)",
                    "Cold Night (11 PM, Winter)",
                ],
            )

            if preset == "Morning Commute (8 AM, Weekday, Spring)":
                init_hr, init_work, init_season, init_temp, init_hum, init_weather = 8, 1, 1, 16.0, 55, 1
            elif preset == "Evening Commute (5 PM, Weekday, Summer)":
                init_hr, init_work, init_season, init_temp, init_hum, init_weather = 17, 1, 2, 28.0, 45, 1
            elif preset == "Weekend Afternoon (2 PM, Summer)":
                init_hr, init_work, init_season, init_temp, init_hum, init_weather = 14, 0, 2, 30.0, 40, 1
            elif preset == "Cold Night (11 PM, Winter)":
                init_hr, init_work, init_season, init_temp, init_hum, init_weather = 23, 1, 4, 3.0, 85, 3
            else:
                init_hr, init_work, init_season, init_temp, init_hum, init_weather = 12, 1, 2, 22.0, 50, 1

            in_hr = st.slider("Hour of Day (0 - 23)", 0, 23, init_hr)
            season_labels = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
            in_season = st.selectbox("Season", [1, 2, 3, 4], index=init_season - 1, format_func=lambda s: season_labels[s])
            in_mnth = st.slider("Month (1 - 12)", 1, 12, (in_season - 1) * 3 + 2)
            in_workingday = st.radio("Day Type", [1, 0], index=0 if init_work == 1 else 1, format_func=lambda w: "Working Day" if w == 1 else "Weekend / Holiday")
            in_holiday = st.checkbox("Official Holiday", value=False)
            in_weekday = st.slider("Day of Week (0: Sun to 6: Sat)", 0, 6, 2 if in_workingday else 0)

        with sim_col2:
            st.write("**Weather Parameters**")
            weather_opts = {
                1: "Clear / Few Clouds",
                2: "Mist / Overcast",
                3: "Light Rain / Snow",
                4: "Heavy Rain / Storm",
            }
            in_weather = st.selectbox("Weather Condition", [1, 2, 3, 4], index=init_weather - 1, format_func=lambda w: weather_opts[w])
            in_temp_c = st.slider("Temperature (°C)", -5.0, 42.0, float(init_temp), step=0.5)
            in_atemp_c = st.slider("Feels-Like Temperature (°C)", -5.0, 48.0, float(in_temp_c + 1.0), step=0.5)
            in_hum_pct = st.slider("Relative Humidity (%)", 0, 100, int(init_hum))
            in_wind_kmh = st.slider("Wind Velocity (km/h)", 0.0, 60.0, 12.0, step=1.0)

        norm_temp = in_temp_c / 41.0
        norm_atemp = in_atemp_c / 50.0
        norm_hum = in_hum_pct / 100.0
        norm_wind = in_wind_kmh / 67.0
        yr_val = 1

        input_row = pd.DataFrame(
            [
                {
                    "season": in_season,
                    "yr": yr_val,
                    "mnth": in_mnth,
                    "hr": in_hr,
                    "holiday": int(in_holiday),
                    "weekday": in_weekday,
                    "workingday": in_workingday,
                    "weathersit": in_weather,
                    "temp": norm_temp,
                    "atemp": norm_atemp,
                    "hum": norm_hum,
                    "windspeed": norm_wind,
                }
            ]
        )

        with sim_col3:
            st.write("**Forecast Output**")
            if not st.session_state["pipeline_ready"]:
                st.warning("Train models in Tab 2 to enable real-time prediction.")
            else:
                best_name = st.session_state["best_name"]
                model = st.session_state["artifacts"][best_name]["model"]
                scaler = st.session_state["split_data"]["scaler"]

                input_scaled = scaler.transform(input_row)
                raw_pred = float(model.predict(input_scaled)[0])
                predicted_cnt = max(0, int(round(raw_pred)))

                if in_workingday == 1:
                    casual_ratio = 0.15 if in_hr in [8, 9, 17, 18] else 0.22
                else:
                    casual_ratio = 0.45
                est_casual = int(round(predicted_cnt * casual_ratio))
                est_reg = max(0, predicted_cnt - est_casual)

                st.metric("Forecasted Total Demand", f"{predicted_cnt:,} rentals/hr")
                st.caption(f"Model used: {best_name}")

                if predicted_cnt > 400:
                    tier_name = "Peak Commuter Period"
                    tier_advice = "High demand expected. Prioritize dock replenishment at commercial hubs."
                elif predicted_cnt > 150:
                    tier_name = "Moderate Demand Period"
                    tier_advice = "Standard volume. Regular station balancing is sufficient."
                else:
                    tier_name = "Low / Off-Peak Period"
                    tier_advice = "Low ridership. Suitable time for fleet maintenance and repairs."

                with st.container(border=True):
                    st.write(f"**Status:** {tier_name}")
                    st.write(tier_advice)

                st.write("**Estimated Breakdown:**")
                st.write(f"- Registered Users: {est_reg}")
                st.write(f"- Casual Users: {est_casual}")
