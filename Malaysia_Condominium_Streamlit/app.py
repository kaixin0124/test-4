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
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Malaysia Condominium Price Prediction",
    page_icon="🏢",
    layout="wide"
)


# ============================================================
# FEATURES
# Based on the notebook feature-selection section
# ============================================================
NUMERIC_FEATURES = [
    "Bedroom",
    "Bathroom",
    "Property Size",
    "# of Floors",
    "Total Units",
    "Parking Lot",
    "Ad List"
]

CATEGORICAL_FEATURES = [
    "Address",
    "Property Type",
    "Tenure Type",
    "Floor Range",
    "Land Title",
    "Category"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# LOAD AND CLEAN DATA
# Based on the notebook cleaning/preprocessing approach
# ============================================================
@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "houses.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            "houses.csv was not found. Please place houses.csv "
            "in the same GitHub folder as app.py."
        )

    df = pd.read_csv(data_path)
    clean = df.copy()

    required_columns = ["price"] + ALL_FEATURES
    missing_columns = [
        col for col in required_columns
        if col not in clean.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ----------------------------
    # Clean price
    # ----------------------------
    clean["price"] = (
        clean["price"]
        .astype(str)
        .str.replace("RM", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
        .replace(["-", "–", "—", "", "nan", "None"], np.nan)
    )
    clean["price"] = pd.to_numeric(clean["price"], errors="coerce")

    # ----------------------------
    # Clean Bedroom and Bathroom
    # ----------------------------
    for col in ["Bedroom", "Bathroom"]:
        clean[col] = (
            clean[col]
            .astype(str)
            .str.extract(r"(\d+(?:\.\d+)?)")[0]
        )
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    # ----------------------------
    # Clean numerical variables
    # ----------------------------
    numeric_to_clean = [
        "Property Size",
        "Ad List",
        "# of Floors",
        "Total Units",
        "Parking Lot"
    ]

    for col in numeric_to_clean:
        clean[col] = (
            clean[col]
            .astype(str)
            .str.replace("sq.ft.", "", regex=False)
            .str.replace("sq.ft", "", regex=False)
            .str.replace("RM", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace(
                ["-", "–", "—", "", "nan", "None"],
                np.nan
            )
        )
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    # ----------------------------
    # Clean categorical variables
    # ----------------------------
    for col in CATEGORICAL_FEATURES:
        clean[col] = (
            clean[col]
            .astype(str)
            .str.strip()
            .replace(
                ["-", "–", "—", "", "nan", "None"],
                np.nan
            )
            .fillna("Unknown")
        )

    # Same main cleaning logic used in the notebook
    clean = clean.drop_duplicates()

    clean = clean.dropna(
        subset=["price", "Property Size"]
    )

    clean = clean[
        (clean["price"] > 0) &
        (clean["Property Size"] > 0)
    ].copy()

    # Numerical missing values are filled using median.
    for col in NUMERIC_FEATURES:
        if clean[col].notna().any():
            clean[col] = clean[col].fillna(
                clean[col].median()
            )
        else:
            clean[col] = clean[col].fillna(0)

    return df, clean


# ============================================================
# MODEL TRAINING
# Uses the selected numerical + categorical features
# and the notebook's preprocessing idea:
# median imputation + scaling + one-hot encoding.
# ============================================================
@st.cache_resource
def train_models(clean):
    X_raw = clean[ALL_FEATURES].copy()
    y = clean["price"].copy()

    # One-hot encode categorical features
    X = pd.get_dummies(
        X_raw,
        columns=CATEGORICAL_FEATURES,
        drop_first=False
    )

    # Make sure all model inputs are numeric
    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Scaling is needed for KNN and SVM.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),

        "KNN": KNeighborsRegressor(
            n_neighbors=5
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),

        "SVM": SVR(
            kernel="rbf",
            C=100000,
            gamma="scale",
            epsilon=0.1
        )
    }

    predictions = {}
    metrics = {}
    fitted_models = {}

    for name, model in models.items():

        if name in ["KNN", "SVM"]:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        fitted_models[name] = model
        predictions[name] = pred

        mse = mean_squared_error(y_test, pred)

        metrics[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R²": r2_score(y_test, pred)
        }

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        predictions,
        metrics,
        fitted_models
    )


# ============================================================
# SAFE UI HELPERS
# Prevents the previous int(NaN) slider error.
# ============================================================
def safe_numeric_max(series, minimum=1):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return minimum

    maximum = int(np.ceil(values.max()))

    if maximum < minimum:
        maximum = minimum

    return maximum


def safe_numeric_median(series, default=0.0):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return float(default)

    value = float(values.median())

    if not np.isfinite(value):
        return float(default)

    return value


def category_options(series):
    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    values = [v for v in values if v != ""]

    if not values:
        return ["Unknown"]

    return sorted(values)


# ============================================================
# LOAD DATA
# ============================================================
df, clean = load_data()


# ============================================================
# SIDEBAR NAVIGATION BAR
# ============================================================
st.sidebar.title("🏢 Condo Price")
st.sidebar.caption("BMDS2003 Data Science")
st.sidebar.divider()

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

st.sidebar.divider()


# ============================================================
# PROPERTY INPUTS
# These are shown in the sidebar so they can be used
# on the Price Prediction page.
# ============================================================
st.sidebar.subheader("Property Information")

bedroom_max = safe_numeric_max(clean["Bedroom"], 1)
bathroom_max = safe_numeric_max(clean["Bathroom"], 1)

bedroom_default = min(3, bedroom_max)
bathroom_default = min(2, bathroom_max)

bedroom = st.sidebar.slider(
    "Bedroom",
    min_value=1,
    max_value=bedroom_max,
    value=bedroom_default
)

bathroom = st.sidebar.slider(
    "Bathroom",
    min_value=1,
    max_value=bathroom_max,
    value=bathroom_default
)

property_size = st.sidebar.number_input(
    "Property Size (sq.ft.)",
    min_value=1.0,
    value=safe_numeric_median(
        clean["Property Size"],
        900.0
    ),
    step=50.0
)

ad_list = st.sidebar.number_input(
    "Ad List",
    min_value=0.0,
    value=safe_numeric_median(
        clean["Ad List"],
        0.0
    ),
    step=1000.0
)

floors = st.sidebar.number_input(
    "# of Floors",
    min_value=0.0,
    value=safe_numeric_median(
        clean["# of Floors"],
        1.0
    ),
    step=1.0
)

total_units = st.sidebar.number_input(
    "Total Units",
    min_value=0.0,
    value=safe_numeric_median(
        clean["Total Units"],
        100.0
    ),
    step=10.0
)

parking = st.sidebar.number_input(
    "Parking Lot",
    min_value=0.0,
    value=safe_numeric_median(
        clean["Parking Lot"],
        1.0
    ),
    step=1.0
)

address = st.sidebar.selectbox(
    "Address",
    category_options(clean["Address"])
)

property_type = st.sidebar.selectbox(
    "Property Type",
    category_options(clean["Property Type"])
)

land_title = st.sidebar.selectbox(
    "Land Title",
    category_options(clean["Land Title"])
)

tenure_type = st.sidebar.selectbox(
    "Tenure Type",
    category_options(clean["Tenure Type"])
)

floor_range = st.sidebar.selectbox(
    "Floor Range",
    category_options(clean["Floor Range"])
)

category = st.sidebar.selectbox(
    "Category",
    category_options(clean["Category"])
)


# ============================================================
# MAIN TITLE
# ============================================================
st.title("🏠 Malaysia Condominium Price Prediction")
st.write(
    "Estimate a condominium price using property and "
    "building information."
)


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":

    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Original Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Cleaned Properties",
        f"{len(clean):,}"
    )

    col3.metric(
        "Predictor Variables",
        f"{len(ALL_FEATURES)}"
    )

    col4.metric(
        "Target Variable",
        "House Price"
    )

    st.divider()

    st.subheader("Project Objective")

    st.write(
        "This application uses regression machine learning "
        "models to predict Malaysian condominium prices "
        "based on property and building characteristics."
    )

    st.subheader("Predictor Variables")

    st.write(
        ", ".join(ALL_FEATURES)
    )

    st.subheader("Models Used")

    st.write(
        "Linear Regression, K-Nearest Neighbours (KNN), "
        "Random Forest Regression, and Support Vector "
        "Machine (SVM)."
    )


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================
elif page == "Exploratory Analysis":

    st.header("Exploratory Data Analysis")

    # Navigation tabs INSIDE the EDA page
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Price Distribution",
            "Price Boxplot",
            "Correlation Heatmap",
            "Property Type"
        ]
    )

    # --------------------------------------------------------
    # PRICE DISTRIBUTION
    # --------------------------------------------------------
    with tab1:

        st.subheader("Distribution of House Prices")

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.histplot(
            clean["price"],
            bins=30,
            kde=True,
            ax=ax
        )

        ax.set_title(
            "Distribution of House Prices"
        )
        ax.set_xlabel(
            "House Price (RM)"
        )
        ax.set_ylabel(
            "Frequency"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # PRICE BOXPLOT
    # --------------------------------------------------------
    with tab2:

        st.subheader("Boxplot of House Prices")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.boxplot(
            x=clean["price"],
            ax=ax
        )

        ax.set_title(
            "Boxplot of House Prices"
        )
        ax.set_xlabel(
            "House Price (RM)"
        )

        ax.ticklabel_format(
            style="plain",
            axis="x"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.info(
            "The boxplot shows the median, quartiles, "
            "spread, and possible outliers in house prices."
        )

    # --------------------------------------------------------
    # CORRELATION HEATMAP
    # --------------------------------------------------------
    with tab3:

        st.subheader("Correlation Heatmap")

        heatmap_features = [
            "Bedroom",
            "Bathroom",
            "Property Size",
            "# of Floors",
            "Total Units",
            "Parking Lot",
            "Ad List",
            "price"
        ]

        correlation_data = clean[
            heatmap_features
        ].copy()

        correlation_matrix = (
            correlation_data.corr()
        )

        fig, ax = plt.subplots(
            figsize=(11, 8)
        )

        # Dark green -> light green.
        # Fixed range makes the scale consistent.
        sns.heatmap(
            correlation_matrix,
            cmap="Greens",
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.5,
            linecolor="white",
            cbar=True,
            annot=False,
            ax=ax
        )

        # Add readable annotations.
        # Dark text on light cells and white text on dark cells.
        for i in range(
            correlation_matrix.shape[0]
        ):
            for j in range(
                correlation_matrix.shape[1]
            ):
                value = correlation_matrix.iloc[i, j]

                text_color = (
                    "white"
                    if abs(value) >= 0.50
                    else "black"
                )

                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold"
                )

        ax.set_title(
            "Correlation Heatmap",
            fontsize=14
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.write(
            "The heatmap displays the correlation between "
            "the numerical predictor variables and house price."
        )

    # --------------------------------------------------------
    # PROPERTY TYPE
    # --------------------------------------------------------
    with tab4:

        st.subheader(
            "Distribution of Property Types"
        )

        property_counts = (
            clean["Property Type"]
            .value_counts()
            .sort_values()
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        property_counts.plot(
            kind="barh",
            ax=ax
        )

        ax.set_title(
            "Distribution of Property Types"
        )
        ax.set_xlabel(
            "Number of Properties"
        )
        ax.set_ylabel(
            "Property Type"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.write(
            "This chart shows the number of properties "
            "available for each property type."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":

    st.header("Model Performance")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        predictions,
        metrics,
        fitted_models
    ) = train_models(clean)

    performance_df = pd.DataFrame(
        metrics
    ).T

    st.subheader(
        "Performance Comparison"
    )

    display_df = performance_df.copy()

    display_df["MAE"] = display_df[
        "MAE"
    ].map(lambda x: f"RM {x:,.2f}")

    display_df["MSE"] = display_df[
        "MSE"
    ].map(lambda x: f"RM {x:,.2f}")

    display_df["RMSE"] = display_df[
        "RMSE"
    ].map(lambda x: f"RM {x:,.2f}")

    display_df["R²"] = performance_df[
        "R²"
    ].map(lambda x: f"{x:.4f}")

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.divider()

    selected_model = st.selectbox(
        "Select Model",
        list(predictions.keys())
    )

    pred = predictions[selected_model]

    # --------------------------------------------------------
    # Actual vs Predicted
    # Two colours:
    # Red = Actual
    # Blue = Predicted
    # --------------------------------------------------------
    st.subheader(
        f"{selected_model}: Actual vs Predicted"
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    # Actual values
    ax.scatter(
        y_test,
        y_test,
        alpha=0.35,
        s=30,
        color="red",
        label="Actual Price"
    )

    # Predicted values
    ax.scatter(
        y_test,
        pred,
        alpha=0.65,
        s=30,
        color="blue",
        label="Predicted Price"
    )

    min_value = min(
        float(y_test.min()),
        float(np.min(pred))
    )

    max_value = max(
        float(y_test.max()),
        float(np.max(pred))
    )

    # Perfect prediction line
    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        color="black",
        linewidth=2,
        label="Perfect Prediction"
    )

    ax.set_title(
        f"{selected_model}: "
        "Actual vs Predicted House Prices"
    )

    ax.set_xlabel(
        "Actual House Price (RM)"
    )

    ax.set_ylabel(
        "Predicted House Price (RM)"
    )

    ax.legend()

    ax.ticklabel_format(
        style="plain",
        axis="both"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.caption(
        "Red points represent actual prices, while blue "
        "points represent predicted prices."
    )


# ============================================================
# PRICE PREDICTION
# ============================================================
elif page == "Price Prediction":

    st.header("Price Prediction")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        predictions,
        metrics,
        fitted_models
    ) = train_models(clean)

    selected_model = st.selectbox(
        "Select Regression Model",
        [
            "Linear Regression",
            "KNN",
            "Random Forest",
            "SVM"
        ]
    )

    input_data = pd.DataFrame([
        {
            "Bedroom": bedroom,
            "Bathroom": bathroom,
            "Property Size": property_size,
            "# of Floors": floors,
            "Total Units": total_units,
            "Parking Lot": parking,
            "Ad List": ad_list,
            "Address": address,
            "Property Type": property_type,
            "Land Title": land_title,
            "Tenure Type": tenure_type,
            "Floor Range": floor_range,
            "Category": category
        }
    ])

    # Same one-hot encoding structure as training
    X_all_raw = clean[
        ALL_FEATURES
    ].copy()

    X_all = pd.get_dummies(
        X_all_raw,
        columns=CATEGORICAL_FEATURES,
        drop_first=False
    )

    X_input = pd.get_dummies(
        input_data,
        columns=CATEGORICAL_FEATURES,
        drop_first=False
    )

    # Make input columns exactly match training columns
    X_input = X_input.reindex(
        columns=X_all.columns,
        fill_value=0
    ).astype(float)

    model = fitted_models[
        selected_model
    ]

    if selected_model in [
        "KNN",
        "SVM"
    ]:
        predicted_price = model.predict(
            scaler.transform(X_input)
        )[0]
    else:
        predicted_price = model.predict(
            X_input
        )[0]

    if st.button(
        "Predict Price",
        type="primary"
    ):
        st.success(
            f"Estimated Property Price: "
            f"RM {predicted_price:,.2f}"
        )

    st.divider()

    st.subheader(
        "Selected Property Information"
    )

    st.dataframe(
        input_data,
        use_container_width=True
    )


# ============================================================
# DATASET
# ============================================================
else:

    st.header("Dataset")

    col1, col2 = st.columns(2)

    col1.metric(
        "Original Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Cleaned Records",
        f"{len(clean):,}"
    )

    st.divider()

    st.subheader(
        "Cleaned Dataset"
    )

    st.dataframe(
        clean,
        use_container_width=True,
        height=600
    )

    st.download_button(
        "Download Cleaned Dataset",
        clean.to_csv(
            index=False
        ).encode("utf-8"),
        "cleaned_houses.csv",
        "text/csv"
    )
