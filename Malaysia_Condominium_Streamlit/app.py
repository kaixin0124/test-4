from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Malaysia Condominium Price Prediction",
    page_icon="🏢",
    layout="wide"
)

FEATURES = ["Bedroom", "Bathroom", "Property Size", "Ad List"]

# ============================================================
# LOAD AND PREPARE DATA
# The cleaning/preparation is kept internal for modelling.
# There is NO separate Data Cleaning page in the app.
# ============================================================
@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "houses.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"houses.csv not found: {data_path}. "
            "Place houses.csv in the same folder as app.py."
        )

    df = pd.read_csv(data_path)
    df_clean = df.copy()

    required = ["price", "Bedroom", "Bathroom", "Property Size", "Ad List"]
    missing = [c for c in required if c not in df_clean.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Same main preparation approach used in the notebook.
    df_clean["price"] = (
        df_clean["price"].astype(str)
        .str.replace("RM", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df_clean["price"] = pd.to_numeric(df_clean["price"], errors="coerce")

    df_clean["Property Size"] = (
        df_clean["Property Size"].astype(str)
        .str.replace("sq.ft.", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df_clean["Property Size"] = pd.to_numeric(
        df_clean["Property Size"], errors="coerce"
    )

    # Use digit extraction, matching the notebook's main cleaning section.
    for col in ["Bedroom", "Bathroom"]:
        df_clean[col] = (
            df_clean[col].astype(str).str.extract(r"(\d+)")[0]
        )
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean["Ad List"] = pd.to_numeric(df_clean["Ad List"], errors="coerce")

    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=["price", "Property Size"])
    df_clean = df_clean[
        (df_clean["price"] > 0) &
        (df_clean["Property Size"] > 0)
    ].copy()

    # Model features: fill missing predictor values with their medians.
    for col in FEATURES:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df, df_clean


@st.cache_resource
def train_models(df_clean):
    X = df_clean[FEATURES].copy()
    y = df_clean["price"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ---------------- Linear Regression ----------------
    # Best parameters from the notebook:
    # fit_intercept=True, positive=False
    scaler_lr = StandardScaler()
    X_train_lr = scaler_lr.fit_transform(X_train)
    X_test_lr = scaler_lr.transform(X_test)

    lr = LinearRegression(
        fit_intercept=True,
        positive=False
    )
    lr.fit(X_train_lr, y_train)
    lr_train_pred = lr.predict(X_train_lr)
    lr_test_pred = lr.predict(X_test_lr)

    # ---------------- KNN ----------------
    # Best parameters from the notebook:
    # n_neighbors=15, p=1, weights='distance'
    scaler_knn = StandardScaler()
    X_train_knn = scaler_knn.fit_transform(X_train)
    X_test_knn = scaler_knn.transform(X_test)

    knn = KNeighborsRegressor(
        n_neighbors=15,
        weights="distance",
        p=1
    )
    knn.fit(X_train_knn, y_train)
    knn_train_pred = knn.predict(X_train_knn)
    knn_test_pred = knn.predict(X_test_knn)

    # ---------------- Random Forest ----------------
    # Best parameters recorded in the notebook:
    # max_depth=10, min_samples_leaf=2,
    # min_samples_split=5, n_estimators=200
    #
    # We use the recorded best parameters directly instead of
    # running the expensive 240-fit GridSearchCV on Streamlit Cloud.
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    )
    rf.fit(X_train, y_train)
    rf_train_pred = rf.predict(X_train)
    rf_test_pred = rf.predict(X_test)

    def score(y_true, pred):
        return {
            "R²": r2_score(y_true, pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, pred)),
            "MAE": mean_absolute_error(y_true, pred)
        }

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,

        "lr": lr,
        "lr_scaler": scaler_lr,
        "lr_train_pred": lr_train_pred,
        "lr_test_pred": lr_test_pred,
        "lr_train_scores": score(y_train, lr_train_pred),
        "lr_test_scores": score(y_test, lr_test_pred),

        "knn": knn,
        "knn_scaler": scaler_knn,
        "knn_train_pred": knn_train_pred,
        "knn_test_pred": knn_test_pred,
        "knn_train_scores": score(y_train, knn_train_pred),
        "knn_test_scores": score(y_test, knn_test_pred),

        "rf": rf,
        "rf_train_pred": rf_train_pred,
        "rf_test_pred": rf_test_pred,
        "rf_train_scores": score(y_train, rf_train_pred),
        "rf_test_scores": score(y_test, rf_test_pred)
    }


try:
    df_original, df_clean = load_data()
except Exception as e:
    st.error("The dataset could not be loaded.")
    st.exception(e)
    st.stop()


# ============================================================
# NAVIGATION
# ============================================================
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Exploratory Analysis",
        "Model Performance",
        "Price Prediction",
        "Dataset"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.title("🏢 Malaysia Condominium Price Prediction")
    st.caption(
        "Streamlit application based on the Malaysia Condominium Prices Jupyter Notebook."
    )

    st.header("Project Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Rows", f"{len(df_original):,}")
    c2.metric("Original Columns", f"{df_original.shape[1]:,}")
    c3.metric("Cleaned Rows", f"{len(df_clean):,}")
    c4.metric("Median Price", f"RM {df_clean['price'].median():,.0f}")

    st.markdown("""
    ### Objective

    This project analyses Malaysian property listing data and develops
    regression models to predict property prices.

    ### Models Used

    - Linear Regression
    - K-Nearest Neighbours (KNN)
    - Random Forest Regression

    ### Selected Prediction Features

    - Bedroom
    - Bathroom
    - Property Size
    - Ad List
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df_clean.head(10), use_container_width=True)


# ============================================================
# EXPLORATORY ANALYSIS
# ============================================================
elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Distribution",
        "Price Boxplot",
        "Correlation Heatmap",
        "Property Type"
    ])

    with tab1:
        st.subheader("Distribution of House Prices")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df_clean["price"], bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of House Prices")
        ax.set_xlabel("House Price (RM)")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader("Boxplot of House Prices")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=df_clean["price"], ax=ax)
        ax.set_title("Boxplot of House Prices")
        ax.set_xlabel("House Price (RM)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("Correlation Heatmap")

        numeric_features = [
            "Bedroom",
            "Bathroom",
            "Property Size",
            "# of Floors",
            "Total Units",
            "Parking Lot"
        ]

        correlation_data = df_clean.copy()

        for col in numeric_features:
            correlation_data[col] = pd.to_numeric(
                correlation_data[col].replace("-", np.nan),
                errors="coerce"
            )

        correlation_data["price"] = pd.to_numeric(
            correlation_data["price"], errors="coerce"
        )

        corr = correlation_data[
            numeric_features + ["price"]
        ].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            cmap="Greens",
            fmt=".2f",
            linewidths=0.5,
            vmin=-0.1,
            vmax=0.8,
            ax=ax
        )
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        st.subheader("Distribution of Property Types")

        if "Property Type" in df_clean.columns:
            counts = df_clean["Property Type"].value_counts()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                y=counts.index.astype(str),
                x=counts.values,
                ax=ax
            )
            ax.set_title("Distribution of Property Types")
            ax.set_xlabel("Number of Properties")
            ax.set_ylabel("Property Type")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Property Type column is not available.")


# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.title("Model Performance")

    with st.spinner("Training models..."):
        models = train_models(df_clean)

    st.subheader("Test Set Performance")

    results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "KNN Regression",
            "Random Forest Regression"
        ],
        "R² Score": [
            models["lr_test_scores"]["R²"],
            models["knn_test_scores"]["R²"],
            models["rf_test_scores"]["R²"]
        ],
        "RMSE": [
            models["lr_test_scores"]["RMSE"],
            models["knn_test_scores"]["RMSE"],
            models["rf_test_scores"]["RMSE"]
        ],
        "MAE": [
            models["lr_test_scores"]["MAE"],
            models["knn_test_scores"]["MAE"],
            models["rf_test_scores"]["MAE"]
        ]
    })

    st.dataframe(
        results.style.format({
            "R² Score": "{:.4f}",
            "RMSE": "RM {:,.2f}",
            "MAE": "RM {:,.2f}"
        }),
        use_container_width=True
    )

    best_model = results.loc[
        results["R² Score"].idxmax(), "Model"
    ]
    st.success(f"Highest test R²: **{best_model}**")

    st.subheader("Training vs Test Performance")

    train_test = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "KNN Regression",
            "Random Forest Regression"
        ],
        "Training R²": [
            models["lr_train_scores"]["R²"],
            models["knn_train_scores"]["R²"],
            models["rf_train_scores"]["R²"]
        ],
        "Test R²": [
            models["lr_test_scores"]["R²"],
            models["knn_test_scores"]["R²"],
            models["rf_test_scores"]["R²"]
        ]
    })

    st.dataframe(
        train_test.style.format({
            "Training R²": "{:.4f}",
            "Test R²": "{:.4f}"
        }),
        use_container_width=True
    )

    selected_model = st.selectbox(
        "Select model for Actual vs Predicted graph",
        [
            "Linear Regression",
            "KNN Regression",
            "Random Forest Regression"
        ]
    )

    if selected_model == "Linear Regression":
        pred = models["lr_test_pred"]
    elif selected_model == "KNN Regression":
        pred = models["knn_test_pred"]
    else:
        pred = models["rf_test_pred"]

    actual = models["y_test"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(actual, pred, alpha=0.5)

    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction"
    )

    ax.set_xlabel("Actual Price (RM)")
    ax.set_ylabel("Predicted Price (RM)")
    ax.set_title(f"Actual vs Predicted Prices ({selected_model})")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if selected_model == "Linear Regression":
        residuals = models["y_test"] - models["lr_test_pred"]

        st.subheader("Linear Regression Residual Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(residuals, bins=50, kde=True, ax=ax)
        ax.axvline(
            residuals.mean(),
            linestyle="--",
            linewidth=2,
            label=f"Mean Residual: RM {residuals.mean():,.2f}"
        )
        ax.set_xlabel("Residual (RM)")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Distribution (Linear Regression)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Residuals vs Predicted Values")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(models["lr_test_pred"], residuals, alpha=0.5)
        ax.axhline(0, linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted Price (RM)")
        ax.set_ylabel("Residual (RM)")
        ax.set_title("Residuals vs Predicted Values")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if selected_model == "Random Forest Regression":
        importance = pd.Series(
            models["rf"].feature_importances_,
            index=FEATURES
        ).sort_values(ascending=True)

        st.subheader("Random Forest Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance.index, importance.values)
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest Feature Importance")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ============================================================
# PRICE PREDICTION
# ============================================================
elif page == "Price Prediction":
    st.title("🏠 Property Price Prediction")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            bedroom = st.number_input(
                "Bedroom",
                min_value=0,
                max_value=20,
                value=3,
                step=1
            )
            bathroom = st.number_input(
                "Bathroom",
                min_value=0,
                max_value=20,
                value=2,
                step=1
            )

        with c2:
            property_size = st.number_input(
                "Property Size (sq.ft.)",
                min_value=100.0,
                max_value=10000.0,
                value=1000.0,
                step=50.0
            )
            ad_list = st.number_input(
                "Ad List",
                min_value=0.0,
                value=float(df_clean["Ad List"].median()),
                step=1.0
            )

        model_name = st.selectbox(
            "Select Model",
            [
                "Linear Regression",
                "KNN Regression",
                "Random Forest Regression"
            ]
        )

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        with st.spinner("Loading model..."):
            models = train_models(df_clean)

        new_data = pd.DataFrame([{
            "Bedroom": bedroom,
            "Bathroom": bathroom,
            "Property Size": property_size,
            "Ad List": ad_list
        }])

        if model_name == "Linear Regression":
            scaled = models["lr_scaler"].transform(new_data)
            prediction = models["lr"].predict(scaled)[0]
        elif model_name == "KNN Regression":
            scaled = models["knn_scaler"].transform(new_data)
            prediction = models["knn"].predict(scaled)[0]
        else:
            prediction = models["rf"].predict(new_data)[0]

        prediction = max(0.0, float(prediction))

        st.success(
            f"Estimated Property Price: **RM {prediction:,.2f}**"
        )

        st.write("Input values:")
        st.dataframe(new_data, use_container_width=True)


# ============================================================
# DATASET
# ============================================================
else:
    st.title("Dataset")

    st.write(
        f"Original dataset: {len(df_original):,} rows × "
        f"{df_original.shape[1]:,} columns."
    )
    st.write(
        f"Prepared modelling dataset: {len(df_clean):,} rows × "
        f"{df_clean.shape[1]:,} columns."
    )

    st.subheader("Dataset Preview")
    st.dataframe(
        df_clean.head(500),
        use_container_width=True
    )

    csv_bytes = df_clean.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prepared Dataset",
        data=csv_bytes,
        file_name="houses_cleaned.csv",
        mime="text/csv"
    )
