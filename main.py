import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from IPython.testing.decorators import skipif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from streamlit import session_state


def session_state_init():
    if 'input_data' not in st.session_state:
        st.session_state.input_data = False
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = False
    if 'input_data' not in st.session_state:
        st.session_state.input_data = False
    if 'show_preprocessed_data' not in st.session_state:
        st.session_state.show_preprocessed_data = False
    if 'data_split_selected' not in st.session_state:
        st.session_state.data_split_selected = False
    if 'arr_month' not in st.session_state:
        st.session_state.arr_month = list()
    if 'arr_day' not in st.session_state:
        st.session_state.arr_day = list()
    if 'confirm_split' not in st.session_state:
        st.session_state.confirm_split = False
    if 'generate_bin' not in st.session_state:
        st.session_state.generate_bin = False
    if 'generate_group' not in st.session_state:
        st.session_state.generate_group = False
    if 'cols_pred_num' not in st.session_state:
        st.session_state.cols_pred_num = list()
    if 'cols_pred_cat' not in st.session_state:
        st.session_state.cols_pred_cat = list()


@st.cache_data
def load_data(file_path):
    """
    Loads data from the specified file path.

    Args:
        file_path (str): Path to the data file.

    Returns:
        pandas.DataFrame: Loaded DataFrame.
    """

    try:
        df = pd.read_csv(file_path)
    except:
        try:
            df = pd.read_parquet(file_path)
        except:
            raise ValueError("Unsupported file format")

    return df


def preprocess_data(df, target_col, feature_cols, timestamp_col):
    """
    Preprocesses the data.

    Args:
        df (pandas.DataFrame): The loaded DataFrame.
        target_col (str): Name of the target column.
        feature_cols (list): List of feature column names.
        timestamp_col (str): Name of the timestamp column.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame.
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    try:
        df['month'] = df[timestamp_col].dt.strftime('%Y%m')
        df['day'] = df[timestamp_col].dt.strftime('%Y%m%d')
        df['month'] = df['month'].astype(int)
        df['day'] = df['day'].astype(int)
        arr_month = np.sort(df['month'].unique()).copy().tolist()
        arr_day = np.sort(df['day'].unique()).copy().tolist()
    except:
        arr_month = None
        arr_day = None

    feature_cols = list(set(feature_cols))

    cols_pred_num = df[feature_cols].select_dtypes('number').columns.tolist()
    cols_pred_cat = df[feature_cols].select_dtypes('object').columns.tolist()

    display_col = list(set([target_col] + feature_cols + [timestamp_col, 'month', 'day']))

    df = df[display_col]

    return df, cols_pred_num, cols_pred_cat, arr_month, arr_day

def select_slider_opt(arr_month):
    arr_perc = list(range(1, 100))
    hoot_th, oot_th = st.select_slider("Select hoot and oot threshold", options=arr_month, value=(arr_month[0], arr_month[-1]))
    train_perc = st.select_slider("Train sample in %", options=arr_perc)
    test_perc_init = 100-train_perc
    arr_perc_test = list(range(1, test_perc_init+1))
    test_perc = st.select_slider("Test sample in %", options=arr_perc_test, value=test_perc_init//2)
    valid_perc = 100 - train_perc - test_perc
    st.write(f"Train perc: {train_perc}")
    st.write(f"Test perc: {test_perc}")
    st.write(f"Valid perc: {valid_perc}")

    return hoot_th, oot_th, train_perc, test_perc, valid_perc

def check_valid_df(df, target_col, feature_cols, timestamp_col):
    if df[target_col].nunique() > 2:
        st.write("Non binary target")
        st.session_state.show_preprocessed_data = False
    elif df[target_col].nunique() < 2:
        st.write("Single class target")
        st.session_state.show_preprocessed_data = False
    else:
        # Preprocess data
        df, cols_pred_num, cols_pred_cat, arr_month, arr_day = preprocess_data(df, target_col, feature_cols,
                                                                               timestamp_col)

        if (arr_month is None) or (arr_day is None):
            st.write("Failed to convert date")
            del df
            st.session_state.show_preprocessed_data = False
        else:
            st.session_state.arr_month = arr_month
            st.session_state.arr_day = arr_day
            st.session_state.processed_data = True
            st.session_state.cols_pred_num = cols_pred_num
            st.session_state.cols_pred_cat = cols_pred_cat
            return df

def main():
    session_state_init()

    st.title("Fast Logistic Regression Modeller")

    # 1. Data Input
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "parquet"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state.input_data = True

        # Select columns
    if st.session_state.input_data:
        st.divider()
        st.header("Metadata")
        target_col = st.selectbox("Select target column", df.columns)
        feature_cols = st.multiselect("Select feature columns", df.columns)
        timestamp_col = st.selectbox("Select timestamp column", df.columns)

        if st.button("Confirm Metadata"):
            df = check_valid_df(df, target_col, feature_cols, timestamp_col)
            df.to_parquet("dataframe.parquet")
            if 'df' not in st.session_state:
                st.session_state.df = df.copy()


     # Display preprocessed data
    if st.session_state.processed_data:
        st.write("Preprocessed Data:")
        st.dataframe(df)
        st.session_state.show_preprocessed_data = True

    if st.session_state.show_preprocessed_data:
        st.divider()
        st.header("Data Split")
        st.write("HOOT: Historical Out of Time (holdout before train/test/split) sample")
        st.write("OOT: Out of Time (holdout after train/test/split) sample")
        st.write("Train/Valid/Test: Development sample")

        # arr_month = np.sort(df['month'].unique())
        hoot_th, oot_th, train_perc, test_perc, valid_perc = select_slider_opt(st.session_state.arr_month)

        if 'df' not in st.session_state:
            df = pd.read_parquet('dataframe.parquet')
        else:
            df = st.session_state.df

        df = get_train_mask(df, hoot_th, oot_th, train_perc, test_perc, valid_perc, target_col)
        df_stats = get_sample_dist(df, target_col)
        st.dataframe(df_stats)

        if st.button("Confirm Splitting"):
            del df_stats
            # df.to_parquet('dataframe.parquet')
            st.session_state.df = df
            st.session_state.confirm_split = True

        if st.session_state.confirm_split:
            st.divider()
            st.header("Exploration Data Analysis")
            st.write("Monthly ODR")
            plot_eda_monthly(df, 'month', target_col)
            st.write("Feature statistics")
            st.dataframe(df[feature_cols].describe().transpose())

            # Automatic binning
            st.divider()
            st.header("Automatic Binning and Grouping")
            st.write("Numerical features")
            num_bin = st.text_input("Number of bin", value='5')
            min_bin_sample = st.text_input("Min sample per bin", value='100')
            if st.button("Generate bin"):
                st.session_state.generate_bin = True
                st.write("Placeholder")

        if st.session_state.generate_bin:
            st.write("Categorical features")
            num_group = st.text_input("Number of group", value='5')
            min_group_sample = st.text_input("Min sample per group", value='100')
            if st.button("Generate group"):
                st.session_state.generate_group = True
                st.write("Placeholder")

        if st.session_state.generate_group:
            st.divider()
            st.header("Feature Selection")
            st.write("Placeholder")
            cols_feat = st.multiselect("Select feature columns", st.session_state.cols_pred_num + st.session_state.cols_pred_cat)



def plot_eda_monthly(df, col_time, target_col):
    df_stats = df[[col_time, target_col]].groupby(col_time).agg(['mean', 'count'])[target_col]
    df_stats = df_stats.rename(columns={'count':'data count', 'mean':'target mean'}).reset_index()

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(8, 6))

    x_axis = np.arange(1, len(df_stats['month'])+1)
    x_ticklabels = ['0'] + pd.to_datetime(df_stats['month'], format='%Y%m').dt.strftime('%Y-%m').values.tolist()


    # Plot bar chart (left y-axis) with 'data count'
    ax1.bar(x_axis, df_stats['data count'], color='blue', label='Data Count')
    ax1.set_xticklabels(x_ticklabels, rotation=45)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Data Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Create a second y-axis for the line chart (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(x_axis, df_stats['target mean'], color='red', marker='o', label='Target Mean')
    ax2.set_ylabel('Target Mean', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 100:.2f}%'))

    # Show the plot
    plt.title('Bar and Line Chart with Dual Y-Axis')
    fig.tight_layout()
    st.pyplot(fig)

def get_sample_dist(df, target_col):
    df_stats = df[['data_type', target_col]].groupby('data_type').agg(['count', 'mean'])[target_col]
    return df_stats


def get_train_mask(df, hoot_th, oot_th, train_perc, test_perc, valid_perc, target_col):
    df['data_type'] = 'development'
    if hoot_th is not None:
        df.loc[df['month']<hoot_th, 'data_type'] = 'hoot'
    if oot_th is not None:
        df.loc[df['month']>oot_th, 'data_type'] = 'oot'

    X = df.loc[df['data_type'] == 'development', :].index
    y =df.loc[df['data_type'] == 'development', target_col]

    test_val_tot = test_perc + valid_perc

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-(train_perc/100), stratify=y,
                                                        random_state=42)

    if valid_perc>0:
        X_test, X_valid, Y_test, Y_valid = train_test_split(X_test, y_test, test_size=valid_perc/test_val_tot,
                                                            stratify=y_test, random_state=42)
        df.loc[X_valid, 'data_type'] = 'valid'

    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_test, 'data_type'] = 'test'

    return df


    # Model training and evaluation
    # ... (Add model training and evaluation code here)

if __name__ == "__main__":
    main()