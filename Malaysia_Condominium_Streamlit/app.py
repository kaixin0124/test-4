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
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Malaysia Condominium Price Prediction",
    page_icon="🏢",
    layout="wide"
)

# ============================================================
# FEATURES USED IN THE MODELS
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
    "Land Title",
    "Tenure Type",
    "Floor Range",
    "Category"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================
@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "houses.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"houses.csv not found: {data_path}. "
            "Please place houses.csv in the same folder as app.py."
        )

    df = pd.read_csv(data_path)
    df_clean = df.copy()

    required_columns = ["price"] + ALL_FEATURES
    missing_columns = [
        col for col in required_columns
        if col not in df_clean.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Clean price
    # --------------------------------------------------------
    df_clean["price"] = (
        df_clean["price"]
        .astype(str)
        .str.replace("RM", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    df_clean["price"] = pd.to_numeric(
        df_clean["price"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Clean Bedroom and Bathroom
    # --------------------------------------------------------
    for col in ["Bedroom", "Bathroom"]:
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.extract(r"(\d+(?:\.\d+)?)")[0]
        )
        df_clean[col] = pd.to_numeric(
            df_clean[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Clean numerical features
    # --------------------------------------------------------
    for col in [
        "Property Size",
        "Ad List",
        "# of Floors",
        "Total Units",
        "Parking Lot"
    ]:
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("RM", "", regex=False)
            .str.strip()
            .replace(
                ["-", "–", "—", "", "nan", "None"],
                np.nan
            )
        )

        df_clean[col] = pd.to_numeric(
            df_clean[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Clean categorical features
    # --------------------------------------------------------
    for col in CATEGORICAL_FEATURES:
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.strip()
            .replace(
                ["-", "–", "—", "", "nan", "None"],
                np.nan
            )
        )

    # --------------------------------------------------------
    # Remove duplicates and invalid records
    # --------------------------------------------------------
    df_clean = df_clean.drop_duplicates()

    df_clean = df_clean.dropna(
        subset=["price", "Property Size"]
    )

    df_clean = df_clean[
        (df_clean["price"] > 0) &
        (df_clean["Property Size"] > 0)
    ].copy()

    # --------------------------------------------------------
    # Fill missing numerical values
    # --------------------------------------------------------
    for col in NUMERIC_FEATURES:
        if df_clean[col].notna().any():
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(
                median_value
            )
        else:
            df_clean[col] = df_clean[col].fillna(0)

    # --------------------------------------------------------
    # Fill missing categorical values
    # --------------------------------------------------------
    for col in CATEGORICAL_FEATURES:
        df_clean[col] = df_clean[col].fillna("Unknown")

    return df, df_clean


# ============================================================
# TRAIN MODELS
# ============================================================
@st.cache_resource
def train_models(df_clean):
    X = df_clean[ALL_FEATURES].copy()
    y = df_clean["price"].copy()

    # Convert categorical variables to dummy variables
    X_encoded = pd.get_dummies(
        X,
        columns=CATEGORICAL_FEATURES,
        drop_first=False
    )

    X_encoded = X_encoded.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.20,
        random_state=42
    )

    # Scaling is required for KNN and SVM
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
            min_samples_leaf=1,
            min_samples_split=2,
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
            model.fit(
                X_train_scaled,
                y_train
            )
            prediction = model.predict(
                X_test_scaled
            )

        else:
            model.fit(
                X_train,
                y_train
            )
            prediction = model.predict(
                X_test
            )

        predictions[name] = prediction
        fitted_models[name] = model

        mse = mean_squared_error(
            y_test,
            prediction
        )

        metrics[name] = {
            "MAE": mean_absolute_error(
                y_test,
                prediction
            ),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R²": r2_score(
                y_test,
                prediction
            )
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
# LOAD DATA
# ============================================================
df, df_clean = load_data()


# ============================================================
# SIDEBAR NAVIGATION
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


# ============================================================
# SIDEBAR PROPERTY INFORMATION
# ============================================================
st.sidebar.divider()
st.sidebar.subheader("Property Information")

# Bedroom
bedroom_max = max(
    1,
    int(df_clean["Bedroom"].max())
)

bedroom = st.sidebar.slider(
    "Bedroom",
    min_value=1,
    max_value=bedroom_max,
    value=min(3, bedroom_max)
)

# Bathroom
bathroom_max = max(
    1,
    int(df_clean["Bathroom"].max())
)

bathroom = st.sidebar.slider(
    "Bathroom",
    min_value=1,
    max_value=bathroom_max,
    value=min(2, bathroom_max)
)

# Property Size
property_size = st.sidebar.number_input(
    "Property Size (sq.ft.)",
    min_value=1.0,
    value=float(
        df_clean["Property Size"].median()
    ),
    step=50.0
)

# Ad List
ad_list = st.sidebar.number_input(
    "Ad List",
    min_value=0.0,
    value=float(
        df_clean["Ad List"].median()
    ),
    step=1000.0
)

# Number of Floors
floors = st.sidebar.number_input(
    "# of Floors",
    min_value=0.0,
    value=float(
        df_clean["# of Floors"].median()
    ),
    step=1.0
)

# Total Units
total_units = st.sidebar.number_input(
    "Total Units",
    min_value=0.0,
    value=float(
        df_clean["Total Units"].median()
    ),
    step=10.0
)

# Parking Lot
parking_lot = st.sidebar.number_input(
    "Parking Lot",
    min_value=0.0,
    value=float(
        df_clean["Parking Lot"].median()
    ),
    step=1.0
)

# Address
address_options = sorted(
    df_clean["Address"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

address = st.sidebar.selectbox(
    "Address",
    address_options
)

# Property Type
property_type_options = sorted(
    df_clean["Property Type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

property_type = st.sidebar.selectbox(
    "Property Type",
    property_type_options
)

# Land Title
land_title_options = sorted(
    df_clean["Land Title"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

land_title = st.sidebar.selectbox(
    "Land Title",
    land_title_options
)

# Tenure Type
tenure_options = sorted(
    df_clean["Tenure Type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

tenure_type = st.sidebar.selectbox(
    "Tenure Type",
    tenure_options
)

# Floor Range
floor_range_options = sorted(
    df_clean["Floor Range"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

floor_range = st.sidebar.selectbox(
    "Floor Range",
    floor_range_options
)

# Category
category_options = sorted(
    df_clean["Category"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

category = st.sidebar.selectbox(
    "Category",
    category_options
)


# ============================================================
# MAIN TITLE
# ============================================================
st.title(
    "🏠 Malaysia Condominium Price Prediction"
)

st.write(
    "Estimate a condominium price using property "
    "and building information."
)


# ============================================================
# OVERVIEW PAGE
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
        f"{len(df_clean):,}"
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

    st.subheader("Models Used")

    st.write(
        "Linear Regression, K-Nearest Neighbours (KNN), "
        "Random Forest Regression, and Support Vector "
        "Machines (SVM)."
    )


# ============================================================
# EXPLORATORY ANALYSIS PAGE
# ============================================================
elif page == "Exploratory Analysis":

    st.header("Exploratory Data Analysis")

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

        st.subheader(
            "Distribution of House Prices"
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.hist(
            df_clean["price"],
            bins=40
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

        st.subheader(
            "House Price Boxplot"
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        sns.boxplot(
            x=df_clean["price"],
            ax=ax
        )

        ax.set_title(
            "House Price Boxplot"
        )

        ax.set_xlabel(
            "House Price (RM)"
        )

        plt.tight_layout()
        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # CORRELATION HEATMAP
    # --------------------------------------------------------
    with tab3:

        st.subheader(
            "Correlation Heatmap"
        )

        numeric_columns = [
            "Bedroom",
            "Bathroom",
            "Property Size",
            "# of Floors",
            "Total Units",
            "Parking Lot",
            "Ad List",
            "price"
        ]

        numeric_df = df_clean[
            numeric_columns
        ].copy()

        correlation = numeric_df.corr(
            numeric_only=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="Greens",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(
            "Correlation Heatmap"
        )

        plt.tight_layout()
        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # PROPERTY TYPE
    # --------------------------------------------------------
    with tab4:

        st.subheader(
            "Distribution of Property Types"
        )

        property_counts = (
            df_clean["Property Type"]
            .value_counts()
            .sort_values(ascending=True)
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
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


# ============================================================
# MODEL PERFORMANCE PAGE
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
    ) = train_models(df_clean)

    metric_df = pd.DataFrame(
        metrics
    ).T

    st.subheader(
        "Performance Comparison"
    )

    st.dataframe(
        metric_df.style.format({
            "MAE": "RM {:,.2f}",
            "MSE": "RM {:,.2f}",
            "RMSE": "RM {:,.2f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True
    )

    st.subheader(
        "Actual vs Predicted House Prices"
    )

    selected_model = st.selectbox(
        "Select Model",
        list(predictions.keys())
    )

    prediction = predictions[
        selected_model
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    # Predicted points
    ax.scatter(
        y_test,
        prediction,
        alpha=0.6,
        label="Predicted Price"
    )

    # Perfect prediction line
    min_value = min(
        float(y_test.min()),
        float(np.min(prediction))
    )

    max_value = max(
        float(y_test.max()),
        float(np.max(prediction))
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
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

    plt.tight_layout()
    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# PRICE PREDICTION PAGE
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
    ) = train_models(df_clean)

    selected_model = st.selectbox(
        "Select Regression Model",
        [
            "Linear Regression",
            "KNN",
            "Random Forest",
            "SVM"
        ]
    )

    # User input
    input_df = pd.DataFrame([{
        "Bedroom": bedroom,
        "Bathroom": bathroom,
        "Property Size": property_size,
        "# of Floors": floors,
        "Total Units": total_units,
        "Parking Lot": parking_lot,
        "Ad List": ad_list,
        "Address": address,
        "Property Type": property_type,
        "Land Title": land_title,
        "Tenure Type": tenure_type,
        "Floor Range": floor_range,
        "Category": category
    }])

    # --------------------------------------------------------
    # Create same dummy-variable structure as training data
    # --------------------------------------------------------
    X_all = df_clean[
        ALL_FEATURES
    ].copy()

    X_all_encoded = pd.get_dummies(
        X_all,
        columns=CATEGORICAL_FEATURES,
        drop_first=False
    )

    input_encoded = pd.get_dummies(
        input_df,
        columns=CATEGORICAL_FEATURES,
        drop_first=False
    )

    input_encoded = input_encoded.reindex(
        columns=X_all_encoded.columns,
        fill_value=0
    )

    input_encoded = input_encoded.astype(float)

    model = fitted_models[
        selected_model
    ]

    # Scale only for KNN and SVM
    if selected_model in ["KNN", "SVM"]:

        prediction = model.predict(
            scaler.transform(
                input_encoded
            )
        )[0]

    else:

        prediction = model.predict(
            input_encoded
        )[0]

    st.subheader(
        "Selected Property Information"
    )

    st.dataframe(
        input_df,
        use_container_width=True
    )

    if st.button(
        "Predict Price",
        type="primary"
    ):

        st.success(
            f"Estimated Property Price: "
            f"RM {prediction:,.2f}"
        )


# ============================================================
# DATASET PAGE
# ============================================================
elif page == "Dataset":

    st.header("Dataset")

    col1, col2 = st.columns(2)

    col1.metric(
        "Original Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Cleaned Records",
        f"{len(df_clean):,}"
    )

    st.subheader(
        "Cleaned Dataset"
    )

    st.dataframe(
        df_clean,
        use_container_width=True,
        height=600
    )

    st.download_button(
        "Download Cleaned Dataset",
        data=df_clean.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="cleaned_houses.csv",
        mime="text/csv"
    )
