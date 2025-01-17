import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset directly from the directory
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Sidebar Configuration
def configure_sidebar(data):
    st.sidebar.header("Filter Options")
    numeric_column = st.sidebar.selectbox(
        "Select Numeric Column", data.select_dtypes(include=np.number).columns, index=0
    )
    categorical_column = st.sidebar.selectbox(
        "Select Categorical Column", data.select_dtypes(include=['object', 'category']).columns, index=0
    )
    return numeric_column, categorical_column

# Dashboard Layout
def main():
    st.set_page_config(page_title="Data Dashboard", layout="wide")
    st.title("✨ Depression Data Dashboard")
    st.markdown(
        """
        Welcome to the **Interactive Data Dashboard**! Explore your Depression data dynamically, generate insights, and visualize data effortlessly. 
        Use the filters on the left to customize your analysis.
        """
    )

    # Load the data directly
    file_path = "train.csv"  # Replace with your dataset's file path
    try:
        data = load_data(file_path)
    except FileNotFoundError:
        st.error("The file 'train.csv' was not found. Please make sure it exists in the current directory.")
        return

    # Display data overview
    st.subheader("📊 Dataset Overview")
    st.dataframe(data.head(), use_container_width=True)
    st.write(f"**Dataset Dimensions:** {data.shape[0]} rows, {data.shape[1]} columns")

    # Show dataset statistics
    if st.checkbox("Show Dataset Statistics"):
        st.subheader("Descriptive Statistics")
        st.write(data.describe())

    # Filter Options
    numeric_col, categorical_col = configure_sidebar(data)

    # Visualization 1: Histogram
    st.subheader(f"Distribution of {numeric_col}")
    hist_fig = px.histogram(
        data, x=numeric_col, nbins=30, color_discrete_sequence=["#636EFA"]
    )
    hist_fig.update_layout(title=f"Histogram of {numeric_col}", template="plotly_white")
    st.plotly_chart(hist_fig, use_container_width=True)

    # Visualization 2: Boxplot
    st.subheader(f"{numeric_col} by {categorical_col}")
    box_fig = px.box(
        data, x=categorical_col, y=numeric_col, color=categorical_col, template="plotly_white"
    )
    box_fig.update_layout(title=f"Boxplot of {numeric_col} grouped by {categorical_col}")
    st.plotly_chart(box_fig, use_container_width=True)

    # Filtered Data Table
    st.subheader("📂 Filtered Data View")
    selected_category = st.selectbox(f"Select {categorical_col}", data[categorical_col].unique())
    filtered_data = data[data[categorical_col] == selected_category]
    st.write(filtered_data)

    # Visualization 3: Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("🔗 Correlation Heatmap")
        corr = data.corr()
        heatmap_fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="viridis",
            title="Correlation Heatmap",
            aspect="auto"
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # Advanced Chart: Scatter Plot with Trendline
    if st.checkbox("Show Advanced Scatter Plot"):
        st.subheader(f"Scatter Plot of {numeric_col} vs Other Numeric Columns")
        scatter_col = st.selectbox(
            "Select another Numeric Column", [col for col in data.select_dtypes(include=np.number).columns if col != numeric_col]
        )
        scatter_fig = px.scatter(
            data,
            x=numeric_col,
            y=scatter_col,
            color=categorical_col,
            trendline="ols",
            template="plotly_white",
            title=f"{numeric_col} vs {scatter_col} Scatter Plot",
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    # Data Download Option
    st.subheader("📥 Download Options")
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv",
    )

# Run the dashboard
if __name__ == "__main__":
    main()
