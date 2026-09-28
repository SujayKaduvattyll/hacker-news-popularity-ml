import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Hacker News Popularity Analytics",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = Path(__file__).parent / "hacker_news_model_data.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


if not DATA_FILE.exists():
    st.error("hacker_news_model_data.csv was not found.")
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error("Unable to load the Hacker News dataset.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📰 Hacker News Popularity Analytics")

st.write(
    "Explore Hacker News posts, popularity levels, engagement patterns, "
    "and machine learning predictions."
)

st.divider()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Popularity Analysis",
        "Post Explorer",
        "Model Performance",
        "About the Data"
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.header("Hacker News Popularity Overview")

    total_posts = len(df)

    if "popularity_label" in df.columns:
        popularity_counts = (
            df["popularity_label"]
            .value_counts()
            .reindex(["Low", "Medium", "High"])
            .fillna(0)
            .astype(int)
        )
    else:
        popularity_counts = pd.Series(
            [0, 0, 0],
            index=["Low", "Medium", "High"]
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Posts", f"{total_posts:,}")

    with col2:
        st.metric("Low Popularity", f"{popularity_counts['Low']:,}")

    with col3:
        st.metric("Medium Popularity", f"{popularity_counts['Medium']:,}")

    with col4:
        st.metric("High Popularity", f"{popularity_counts['High']:,}")

    st.divider()

    st.subheader("Popularity Distribution")

    distribution = pd.DataFrame(
        {
            "Popularity": popularity_counts.index,
            "Posts": popularity_counts.values
        }
    )

    st.bar_chart(
        distribution.set_index("Popularity")
    )

    st.subheader("Popularity Share")

    share = (
        popularity_counts / popularity_counts.sum() * 100
    ).round(2)

    share_df = pd.DataFrame(
        {
            "Popularity": share.index,
            "Percentage": share.values
        }
    )

    st.dataframe(
        share_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Popularity is divided into three categories: Low, Medium, "
        "and High."
    )


# =========================================================
# POPULARITY ANALYSIS
# =========================================================

elif page == "Popularity Analysis":

    st.header("Popularity Analysis")

    if "popularity_label" not in df.columns:
        st.error("Popularity information is not available.")
        st.stop()

    popularity_counts = (
        df["popularity_label"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .astype(int)
    )

    st.subheader("Posts by Popularity Level")

    st.bar_chart(
        popularity_counts
    )

    st.divider()

    # Score analysis

    if "score" in df.columns:

        st.subheader("Score Distribution")

        score_stats = df["score"].describe()

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Average Score",
                f"{df['score'].mean():.2f}"
            )

        with c2:
            st.metric(
                "Median Score",
                f"{df['score'].median():.2f}"
            )

        with c3:
            st.metric(
                "Minimum Score",
                f"{df['score'].min():.0f}"
            )

        with c4:
            st.metric(
                "Maximum Score",
                f"{df['score'].max():.0f}"
            )

        score_distribution = (
            df.groupby("popularity_label")["score"]
            .mean()
            .reindex(["Low", "Medium", "High"])
        )

        st.subheader("Average Score by Popularity")

        st.bar_chart(score_distribution)

    # Comments analysis

    if "comments_count" in df.columns:

        st.subheader("Comments and Engagement")

        comment_stats = (
            df.groupby("popularity_label")["comments_count"]
            .mean()
            .reindex(["Low", "Medium", "High"])
        )

        st.bar_chart(comment_stats)

        engagement = pd.DataFrame(
            {
                "Popularity": comment_stats.index,
                "Average Comments": comment_stats.values.round(2)
            }
        )

        st.dataframe(
            engagement,
            use_container_width=True,
            hide_index=True
        )

    # Posting time

    if "hour_posted" in df.columns:

        st.subheader("Posts by Hour")

        hourly = df["hour_posted"].value_counts().sort_index()

        st.line_chart(hourly)


# =========================================================
# POST EXPLORER
# =========================================================

elif page == "Post Explorer":

    st.header("🔎 Post Explorer")

    st.write(
        "Search and filter Hacker News posts by popularity and other "
        "available attributes."
    )

    # Popularity filter

    if "popularity_label" in df.columns:

        popularity_options = [
            "All",
            "Low",
            "Medium",
            "High"
        ]

        selected_popularity = st.selectbox(
            "Popularity",
            popularity_options
        )

        filtered_df = df.copy()

        if selected_popularity != "All":
            filtered_df = filtered_df[
                filtered_df["popularity_label"]
                == selected_popularity
            ]

    else:
        filtered_df = df.copy()

    # Post type filter

    if "post_type" in df.columns:

        post_type_options = [
            "All"
        ] + sorted(
            df["post_type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_type = st.selectbox(
            "Post Type",
            post_type_options
        )

        if selected_type != "All":
            filtered_df = filtered_df[
                filtered_df["post_type"].astype(str)
                == selected_type
            ]

    # Search

    search_text = st.text_input(
        "Search post titles",
        placeholder="Enter a keyword..."
    )

    if search_text and "title" in filtered_df.columns:

        filtered_df = filtered_df[
            filtered_df["title"]
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                na=False
            )
        ]

    st.write(
        f"Showing **{len(filtered_df):,}** posts"
    )

    # Display useful columns

    preferred_columns = [
        "title",
        "popularity_label",
        "score",
        "comments_count",
        "post_type",
        "domain",
        "author"
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in filtered_df.columns
    ]

    if display_columns:

        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header("Popularity Prediction Model")

    st.write(
        "Machine learning models were evaluated for predicting whether "
        "a Hacker News post belongs to the Low, Medium, or High "
        "popularity category."
    )

    results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "SVM",
                "Gradient Boosting",
                "Tuned Random Forest",
                "Tuned Gradient Boosting"
            ],
            "Accuracy": [
                0.4500,
                0.3983,
                0.4567,
                0.4017,
                0.4617,
                0.4617,
                0.4533
            ],
            "Macro F1": [
                0.3886,
                0.3908,
                0.4184,
                0.1910,
                0.4184,
                0.4181,
                0.3961
            ]
        }
    )

    st.subheader("Model Comparison")

    st.dataframe(
        results.style.format(
            {
                "Accuracy": "{:.4f}",
                "Macro F1": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Accuracy")

    st.bar_chart(
        results.set_index("Model")[["Accuracy"]]
    )

    st.subheader("Macro F1 Score")

    st.bar_chart(
        results.set_index("Model")[["Macro F1"]]
    )

    st.divider()

    st.subheader("Prediction Performance by Popularity")

    class_performance = pd.DataFrame(
        {
            "Popularity": [
                "Low",
                "Medium",
                "High"
            ],
            "Precision": [
                0.47,
                0.44,
                0.45
            ],
            "Recall": [
                0.73,
                0.22,
                0.34
            ],
            "F1 Score": [
                0.57,
                0.29,
                0.39
            ]
        }
    )

    st.dataframe(
        class_performance.style.format(
            {
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1 Score": "{:.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The prediction model is intended to identify popularity "
        "patterns. Actual post popularity can depend on many factors "
        "that are not represented in the available data."
    )


# =========================================================
# ABOUT THE DATA
# =========================================================

elif page == "About the Data":

    st.header("About the Data")

    st.write(
        "This dashboard uses a dataset of Hacker News posts containing "
        "post information, timing attributes, title characteristics, "
        "topic indicators, and popularity information."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Posts",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Columns",
            f"{len(df.columns):,}"
        )

    with col3:
        st.metric(
            "Duplicate Rows",
            f"{df.duplicated().sum():,}"
        )

    st.divider()

    st.subheader("Available Information")

    categories = {
        "Post Information": [
            "Post type",
            "URL",
            "Domain",
            "Author"
        ],
        "Engagement": [
            "Score",
            "Comments count"
        ],
        "Timing": [
            "Hour posted",
            "Day of week",
            "Month",
            "Weekend indicator"
        ],
        "Title Characteristics": [
            "Title length",
            "Word count",
            "Average word length",
            "Uppercase count",
            "Number count",
            "Special character count"
        ],
        "Topic Indicators": [
            "AI",
            "Python",
            "Open Source",
            "Security",
            "Database",
            "Linux",
            "Startup"
        ]
    }

    for category, items in categories.items():

        st.subheader(category)

        for item in items:
            st.write("• " + item)

    st.divider()

    st.subheader("Data Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Hacker News Popularity Analytics Dashboard"
    )
