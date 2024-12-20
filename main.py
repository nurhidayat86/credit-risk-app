import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import ast
from datetime import datetime
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import DecisionTreeDiscretiser
from feature_engine.encoding import DecisionTreeEncoder
from feature_engine.encoding import WoEEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


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
    if 'feat_num_bin' not in st.session_state:
        st.session_state.feat_num_bin = dict()
    if 'df_feat_num' not in st.session_state:
        st.session_state.df_feat_num = None
    if 'generated_bin_btn' not in st.session_state:
        st.session_state.generated_bin_btn = False
    if 'df_feat_cat' not in st.session_state:
        st.session_state.df_feat_cat = False
    if 'feat_cat_group' not in st.session_state:
        st.session_state.feat_cat_group = False
    if 'confirm_feature' not in st.session_state:
        st.session_state.confirm_feature = False
    if 'confirm_manual_binning' not in st.session_state:
        st.session_state.confirm_manual_binning = False
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if 'target_col' not in st.session_state:
        st.session_state.target_col = 'False'
    if 'scorecard_df' not in st.session_state:
        st.session_state.scorecard_df = None
    if 'eval_score' not in st.session_state:
        st.session_state.eval_score = None
    if 'sc_table' not in st.session_state:
        st.session_state.sc_table = None



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
    df[cols_pred_num] = df[cols_pred_num].fillna(-9999)
    df[cols_pred_cat] = df[cols_pred_cat].fillna('NA')
    df = df[display_col]

    return df, cols_pred_num, cols_pred_cat, arr_month, arr_day

def select_slider_opt(arr_month):
    arr_perc = list(range(1, 100))
    hoot_th, oot_th = st.select_slider("Select hoot and oot threshold", options=arr_month, value=(arr_month[0], arr_month[-1]))
    train_perc = st.select_slider("Train sample in %", options=arr_perc, value=50)
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
            st.session_state.df = df
            return df

def main():
    session_state_init()

    st.title("Credit Risk Express: Ver. 1e-10")
    st.write("""Simple logistic regression based scorecard modeller\n
    Features:\n
    1. IV & correlation based feature selection\n
    2. L2 hyperparameter tuning\n
    3. SQL producer\n
    Use CSV or PARQUET file format: a column containing timestamp and \n
    binary label (0,1) information must be present in the dataset.""")

    # 1. Data Input
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "parquet"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df['category'] = np.random.choice(['A', 'B', 'C', 'D', 'F', 'G', 'H'], size=len(df))
        df['category2'] = np.random.choice(['AA', 'BA', 'CD'], size=len(df))

        st.session_state.input_data = True

        # Select columns
    if st.session_state.input_data:
        st.divider()
        st.header("Metadata")

        # get binary column
        binary_columns = [col for col in df.columns if df[col].nunique() == 2]

        target_col = st.selectbox("Select target column", binary_columns)
        feature_cols = st.multiselect("Select feature columns", df.columns)
        timestamp_col = st.selectbox("Select timestamp column", df.select_dtypes('object').columns.tolist() + df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist())

        if st.button("Confirm Metadata"):
            df = check_valid_df(df, target_col, feature_cols, timestamp_col)
            df.to_parquet("dataframe.parquet")
            if 'df' not in st.session_state:
                st.session_state.df = df.copy()
                st.session_state.target_col = target_col


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
        num_bin = st.text_input("Number of bin", value='3')
        min_bin_sample = st.text_input("Min sample per bin", value='100')
        st.write("Categorical features")
        num_group = st.text_input("Number of group", value='3')
        min_group_sample = st.text_input("Min sample per group", value='100')

        if st.button("Generate Automatic WoE"):
            df_used = df.loc[df['data_type']=='train'].copy()
            st.session_state.df_feat_num, st.session_state.feat_num_bin = automatic_feature_binning(df_used, st.session_state.cols_pred_num, target_col, int(num_bin), int(min_bin_sample))
            st.session_state.df_feat_num, st.session_state.woe_dict, st.session_state.woe_encoder_num = (
                calculate_woe(df_used, st.session_state.cols_pred_num, target_col, st.session_state.feat_num_bin))
            st.session_state.df_feat_cat, st.session_state.feat_cat_group, st.session_state.dt_encoder = (
                encode_with_decision_tree(df_used, st.session_state.cols_pred_cat, target_col, int(num_group),
                                          int(min_group_sample)))
            st.session_state.df_feat_cat, st.session_state.woe_dict_group, st.session_state.woe_encoder_cat = (
                calculate_woe(df_used, st.session_state.cols_pred_cat, target_col))
            st.session_state.generate_bin = True

    if st.session_state.generate_bin:
        st.subheader("Woe for numerical and categorical feature")
        st.dataframe(st.session_state.df_feat_num)
        st.dataframe(st.session_state.df_feat_cat)

        st.divider()
        st.header("Feature Selection")
        st.write("Information values in train data:")
        df_train = st.session_state.df_feat_num.merge(st.session_state.df_feat_cat, left_index=True, right_index=True, how='left')
        df_train[target_col] = df.loc[df['data_type']=='train', target_col].values
        st.session_state.df_train = df_train.copy()
        iv_dict = calculate_iv(st.session_state.df_train, st.session_state.df_train, target_col)
        st.dataframe(iv_dict)
        initial_corr_col = iv_dict.index.tolist()[:2]
        cols_feat = st.multiselect("Select feature columns used for modelling",
                                   iv_dict.index.tolist(),
                                   default=initial_corr_col)
        corr_selected_feature = df_train[cols_feat].corr()
        corr_fig = display_corr(corr_selected_feature)
        st.subheader("Feature correlation")
        st.pyplot(corr_fig)
        st.write("CONFIRM FEATURES SELECTION, PRESS BUTTON TO PROCEED AND CLEAR MEMORIES")
        st.write("Note: You need to start from the beggining if you decided to change the feature selection")
        if st.button("Confirm Features!"):
            del df_stats, df_train
            st.session_state.df = df
            st.session_state.corr_selected_feature = corr_selected_feature.columns.tolist()
            st.session_state.confirm_feature = True
            st.session_state.target_col = target_col

    if st.session_state.confirm_feature:
        st.divider()
        st.header("Manual WoE Setting")
        option = st.selectbox(
            "Choose feature",
            st.session_state.corr_selected_feature
        )

        st.write("You selected:", option)
        st.write("Fine tune your bin")
        tune_woe(option)
        if st.button("Prepare data for modelling"):
            st.session_state.confirm_manual_binning = True

    if st.session_state.confirm_manual_binning:
        st.divider()

        df, woe_cols = calculate_woe_all(st.session_state.df)
        st.session_state.df = df.copy()
        st.session_state.woe_cols = woe_cols

        st.header("Hyperparameter tuning")
        st.write("Learning Journey")
        if 'valid' in df.columns:
            best_model, best_param, summary_df = tune_logistic_regression(df.loc[df['data_type'].isin(['train', 'valid'])],
                                                              st.session_state.woe_cols, st.session_state.target_col,
                                                              n_folds=5)
        else:
            best_model, best_param, summary_df = tune_logistic_regression(df.loc[df['data_type'] == 'train'],
                                                              st.session_state.woe_cols, st.session_state.target_col,
                                                              n_folds=5)
        st.dataframe(summary_df)
        st.write("Best parameter")
        st.write(best_param)
        st.write(f"""Stats:\n
        Best model Coefficient: {best_model.coef_}\n
        Best model Intercept: {best_model.intercept_}\n
        Feature columns: {st.session_state.woe_cols}""")
        st.divider()
        st.header("Scorecards")
        st.write("Set your scorecard conversion")
        pdo = st.text_input("PDO", value=20)
        base_odd = st.text_input("Base odd (X:1)", value=50)
        base_score = st.text_input("Score at base odd", value=300)

        # Assuming 'log_reg_model' is your trained LogisticRegression model and 'feature_columns' is your list of feature names
        scorecard_df = logistic_to_scorecard(df[st.session_state.woe_cols],
                                             best_model,
                                             st.session_state.woe_cols,
                                             pdo=int(pdo),
                                             base_odd=int(base_odd),
                                             base_score=int(base_score))

        st.subheader("Scorecard table")
        st.dataframe(scorecard_df)
        st.session_state.df = df.copy()

        if st.button("Evaluate Score!"):

            st.session_state.scorecard_df = scorecard_df
            st.session_state.eval_score = True
            df = score_row(df, st.session_state.scorecard_df, st.session_state.woe_cols)
            st.session_state.df = df

    if st.session_state.eval_score:
        st.divider()
        st.header("Evaluation")
        st.write("Prediction sample")
        st.dataframe(st.session_state.df.head())
        df_summary = calculate_auc_roc(st.session_state.df, 'score', st.session_state.target_col, 'data_type')
        st.write("AUC summary")
        st.dataframe(df_summary)

        st.divider()
        st.header("Production script")
        st.write("Final score card")
        sc_table = generate_sc_table()
        st.session_state.sc_table = sc_table
        st.dataframe(sc_table)
        your_table = st.text_input("Table Name", value="my_table")
        customer_id = st.text_input("Customer id", value="customer_id")
        score_date = st.text_input("Score datetime", value="score_date")
        query = generate_sql_query(sc_table, your_table,  customer_id, score_date)
        st.code(query)


def generate_sql_query(df, table_name, customer_id, score_date):
    sql_queries = []

    # Loop through the DataFrame to create CASE WHEN statements
    case_when_num = ""
    for ifeat in st.session_state.cols_feat_num:
        df_used = df.loc[df["Feature"]==ifeat,:]

        case_when_num += f"CASE\n"

        for index, row in df_used.iterrows():
            bin_condition = row['Bin']
            points = row['Points']

            if '-inf' in bin_condition:
                bin_condition = bin_condition.replace('-inf', '-999999999')
            if 'inf' in bin_condition:
                bin_condition = bin_condition.replace('inf', '999999999')
            bin_parts = bin_condition.strip('()[]').split(',')
            lower_bound = bin_parts[0].strip()
            upper_bound = bin_parts[1].strip()

            case_when_num += f"WHEN {ifeat} > {lower_bound} AND {ifeat} <= {upper_bound} THEN {points}\n"

        case_when_num += f"END AS {ifeat}_score,\n"

    for ifeat in st.session_state.cols_feat_cat:
        df_used = df.loc[df["Feature"] == ifeat, :]

        case_when_num += f"CASE\n"

        for index, row in df_used.iterrows():
            bin_condition = row['Bin']
            points = row['Points']

            case_when_num += f"WHEN {ifeat} = '{bin_condition}' THEN {points}\n"

        case_when_num += f"END AS {ifeat}_score,\n"

    # Combine the SQL queries into a final SELECT statement
    sql_queries = f"WITH temp AS (\nSELECT\n{customer_id},\n{datetime.now().strftime('%Y-%m-%d')} AS {score_date},\n" + case_when_num[:-2] + f"\nFROM {table_name}\n)\n\n"

    total_score = "( "
    for ichar in [f"{feature}_score +\n" for feature in df["Feature"].unique()]: total_score += ichar

    final_sql = sql_queries + "SELECT\n*,\n" + total_score[:-2] + ") AS total_score \nFROM temp;"

    return final_sql

def generate_sc_table():
    df_mapper = st.session_state.scorecard_df
    df_mapper['Attribute'] = df_mapper['Attribute'].round(4)
    swap_dict_cat = dict()
    for ikey in st.session_state.feat_cat_group.keys():
        dict_temp = {}
        for jkey in st.session_state.feat_cat_group[ikey].keys():
            dict_temp[jkey] = st.session_state.woe_dict_group[ikey][str(st.session_state.feat_cat_group[ikey][jkey])]
        swap_dict_cat[ikey] = dict_temp

    sc_content = []
    for ifeat in st.session_state.cols_feat_num:
        # st.write(ifeat)
        for jfeat in st.session_state.woe_dict[ifeat].keys():
            jwoe = np.round(st.session_state.woe_dict[ifeat][jfeat], 4)
            contain = [ifeat, jfeat] + df_mapper.loc[(df_mapper['Feature']==ifeat+"_woe") & (df_mapper['Attribute'] == jwoe),["Weight", "Attribute", "Points"]].values.tolist()[0]
            sc_content.append(contain)

    for ifeat in st.session_state.cols_feat_cat:
        for jfeat in swap_dict_cat[ifeat].keys():
            jwoe = np.round(swap_dict_cat[ifeat][jfeat], 4)
            contain = [ifeat, jfeat] + df_mapper.loc[(df_mapper['Feature']==ifeat+"_woe") & (df_mapper['Attribute'] == jwoe),["Weight", "Attribute", "Points"]].values.tolist()[0]
            sc_content.append(contain)

    sc_content = pd.DataFrame(sc_content, columns=["Feature", "Bin", "Weight", "Attribute", "Points"])
    return sc_content

def score_row(df, scorecard_df, feature_columns):
    df['score'] = 0
    for icol in feature_columns:
        mapper = scorecard_df.loc[scorecard_df['Feature']==icol, ["Attribute", "Points"]]
        mapper_dict = pd.Series(mapper['Points'].values, index=mapper['Attribute'])
        df['score'] += df[icol].map(mapper_dict)
    return df

def calculate_auc_roc(df, score_column, target_column, groupby_column):
    auc_scores = {}

    # Group by the 'data_type' column
    grouped = df.groupby(groupby_column)

    def calc_auc(y, x):
        auc = roc_auc_score(y, x)
        if auc < 0.5: auc=1-auc
        return auc

    for group_name, group_data in grouped:
        # Calculate the AUC-ROC score for each group
        auc = calc_auc(group_data[target_column], group_data[score_column])
        auc_scores[group_name] = auc

    return auc_scores

def logistic_to_scorecard(df, model, feature_names, pdo=20, base_odd=50, base_score=300):
    """
    Converts a trained logistic regression model to a credit risk scorecard.

    Args:
        model: Trained scikit-learn LogisticRegression model.
        feature_names: List of feature names corresponding to the model coefficients.
        pdo: Points to Double the Odds (e.g., 20 means a 20-point increase doubles the odds).
        base_odds: The odds at the base score.
        base_score: Score at base_odds:1

    Returns:
        pandas.DataFrame: Scorecard with feature names, attributes, points, and weights.
        float: Base score.
    """

    n_feature = len(feature_names)
    B_0 = model.intercept_[0]
    factor = pdo / np.log(2)
    offset = base_score - (factor*np.log(base_odd))
    constant = offset/n_feature

    scorecard = []

    for icol in range(0,n_feature):
        B_i = model.coef_[0][icol]
        feature = feature_names[icol]
        for Xi in np.sort(df[feature].unique())[::-1]:
            contribution = -(((B_i * Xi) + (B_0/n_feature)) * factor) + constant
            scorecard.append([feature, B_i, Xi, contribution])

    return pd.DataFrame(scorecard, columns=['Feature', 'Weight', 'Attribute', 'Points'])


def tune_logistic_regression(df, feature_columns, target_column, n_folds=5, C1_values=np.logspace(-4, 4, 20)):
    # Split the dataframe into features and target

    X = df[feature_columns]
    y = df[target_column]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Set up the logistic regression model
    log_reg = LogisticRegression(solver='liblinear', max_iter=1000)

    # Define the hyperparameters to tune
    param_grid = {
        'C': C1_values,  # Regularization strength
        'penalty': ['l2'],  # L1 and L2 regularization
        'solver': ['liblinear']  # Solver compatible with L1
    }

    # Set up the stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Set up GridSearchCV with AUC-ROC as the scoring metric
    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)

    # Fit the model using cross-validation
    grid_search.fit(X_scaled, y)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_
    best_param = grid_search.best_params_

    # Create a DataFrame to summarize the grid search results
    results = pd.DataFrame(grid_search.cv_results_)

    # Extract the relevant columns: mean_test_score (AUC), C, and penalty
    summary_df = results[['param_C', 'param_penalty', 'mean_test_score']]

    # Rename columns for clarity
    summary_df.columns = ['C_value', 'Penalty', 'ROC_AUC']

    # Return the best model and the summary dataframe
    return best_model, best_param, summary_df


def calculate_woe(df, features, target_col, thresholds=None):
    # Initialize a dictionary to hold WoE for each feature
    woe_df = pd.DataFrame()

    # Loop over each feature
    if thresholds is not None:
        for feature in features:
            # Get the threshold for the current feature
            threshold = thresholds[feature]
            woe_df[feature] = pd.cut(df[feature], threshold)
            # woe_df[feature] = woe_df[feature].fillna(-9999)
            woe_df[feature] = woe_df[feature].astype('str')

    else:
        for feature in features:
            dict_transformer = st.session_state.feat_cat_group[feature]
            woe_df[feature] = df[feature].map(dict_transformer)
            # woe_df[feature] = woe_df[feature].fillna('-9999')
            woe_df[feature] = woe_df[feature].astype('str')

    woe_encoder = WoEEncoder()
    df_transformed = woe_encoder.fit_transform(woe_df[features], df[target_col])
    del woe_df
    return df_transformed, woe_encoder.encoder_dict_, woe_encoder


def calculate_woe_all(df):
    train_mask = (df.data_type=='train')

    # Initialize a dictionary to hold WoE for each feature
    thresholds = st.session_state.feat_num_bin
    dict_group = st.session_state.feat_cat_group
    features_num = st.session_state.cols_feat_num
    features_cat = st.session_state.cols_feat_cat
    target_col = st.session_state.target_col

    woe_df = pd.DataFrame(index = df.index)
    woe_cols = list()

    # Loop over each feature
    if len(features_num) > 0:
        for feature in features_num:
            # Get the threshold for the current feature
            threshold = thresholds[feature]
            woe_df[feature] = pd.cut(df[feature], threshold, labels=False)
            woe_df[feature] = woe_df[feature].astype('object')

        woe_encoder = WoEEncoder()
        woe_encoder.fit(woe_df.loc[train_mask, features_num], df.loc[train_mask, target_col])
        woe_df_num = woe_encoder.transform(woe_df[features_num])

        for icol in woe_df_num.columns:
            woe_df_num = woe_df_num.rename(columns={icol: icol + "_woe"})

        if set(woe_df_num.columns.tolist()).issubset(set(df.columns.tolist())):
            df = df.drop(woe_df_num, axis='columns')

        df = df.merge(woe_df_num, left_index=True, right_index=True, how='left')

        woe_cols.extend(woe_df_num.columns.tolist())

    woe_df = pd.DataFrame(index = df.index)

    if len(features_cat) > 0:
        for feature in features_cat:
            woe_df[feature] = df[feature].map(dict_group[feature])
            woe_df[feature] = woe_df[feature].astype('object')


        woe_encoder = WoEEncoder()
        woe_encoder.fit(woe_df.loc[train_mask, features_cat], df.loc[train_mask, target_col])
        woe_df_cat = woe_encoder.transform(woe_df[features_cat])

        for icol in woe_df_cat.columns:
            woe_df_cat = woe_df_cat.rename(columns={icol: icol+"_woe"})

        if set(woe_df_cat.columns.tolist()).issubset(set(df.columns.tolist())):
            df = df.drop(woe_df_cat, axis='columns')

        df = df.merge(woe_df_cat, left_index=True, right_index=True, how='left')
        woe_cols.extend(woe_df_cat.columns.tolist())

    return df, woe_cols


def tune_woe(option):
    df_used = st.session_state.df
    try:
        df_used = df_used.loc[df_used.data_type.isin(['train', 'valid'])].copy()
    except:
        df_used = df_used.loc[df_used.data_type=='train'].copy()

    st.session_state.cols_feat_num = df_used[st.session_state.corr_selected_feature].select_dtypes(
        'number').columns.tolist()
    st.session_state.cols_feat_cat = df_used[st.session_state.corr_selected_feature].select_dtypes(
        'object').columns.tolist()

    if option in st.session_state.cols_feat_num:
        bin_new = st.text_input("Number of bin", value=st.session_state.feat_num_bin[option])
        bin_new = fix_bin(bin_new)
        st.session_state.feat_num_bin[option] = bin_new
        grouped, woe_summary = calculate_woe_feature_num(df_used, option, st.session_state.target_col, bin_new)


    elif option in st.session_state.cols_feat_cat:
        bin_new = st.text_input("Number of bin", value=st.session_state.feat_cat_group[option])
        bin_new = ast.literal_eval(bin_new)
        st.session_state.feat_cat_group[option] = bin_new
        grouped, woe_summary = calculate_woe_feature_cat(df_used, option, st.session_state.target_col, bin_new)

    st.dataframe(grouped)
    st.dataframe(woe_summary)


def calculate_woe_feature_cat(df, feature, target, transform_dict):
    # Create bins based on the threshold
    df['binned_feature'] = df[feature].map(transform_dict)


    # Calculate the total number of good (target == 0) and bad (target == 1) cases
    good_total = df[df[target] == 0].shape[0]  # Total good outcomes (target = 0)
    bad_total = df[df[target] == 1].shape[0]  # Total bad outcomes (target = 1)

    # Group by the binned feature and calculate the distribution of good and bad outcomes
    grouped = df.groupby(['data_type', 'binned_feature'])[target].agg(['sum', 'count'])

    # Calculate the number of good and bad outcomes in each bin
    grouped['good'] = (grouped['count'] - grouped['sum']) / good_total  # Distribution of good outcomes
    grouped['bad'] = grouped['sum'] / bad_total  # Distribution of bad outcomes
    grouped['bad rate'] = grouped['sum'] / grouped['count']

    # Calculate WoE for each bin (bin values 0 and 1)
    grouped['WoE'] = np.log(grouped['good'] / grouped['bad'])

    grouped = grouped.reset_index()

    # Print the WoE values
    return grouped[['data_type', 'binned_feature', 'good', 'bad', 'sum', 'count', 'bad rate']], grouped.loc[
        grouped['data_type'] == 'train', ['binned_feature', 'good', 'bad', 'sum', 'count', 'bad rate', 'WoE']]


def calculate_woe_feature_num(df, feature, target, threshold):
    # Create bins based on the threshold
    df['binned_feature'] = pd.cut(df[feature], threshold)
    df['binned_feature'] = df['binned_feature'].astype('object')

    # Calculate the total number of good (target == 0) and bad (target == 1) cases
    good_total = df[df[target] == 0].shape[0]  # Total good outcomes (target = 0)
    bad_total = df[df[target] == 1].shape[0]  # Total bad outcomes (target = 1)

    # Group by the binned feature and calculate the distribution of good and bad outcomes
    grouped = df.groupby(['data_type','binned_feature'])[target].agg(['sum', 'count'])

    # Calculate the number of good and bad outcomes in each bin
    grouped['good'] = (grouped['count'] - grouped['sum']) / good_total  # Distribution of good outcomes
    grouped['bad'] = grouped['sum'] / bad_total  # Distribution of bad outcomes
    grouped['bad rate'] = grouped['sum'] / grouped['count']

    # Calculate WoE for each bin (bin values 0 and 1)
    grouped['WoE'] = np.log(grouped['good'] / grouped['bad'])

    grouped = grouped.reset_index()

    # Print the WoE values
    return grouped[['data_type', 'binned_feature', 'good', 'bad', 'sum', 'count', 'bad rate']], grouped.loc[grouped['data_type']=='train', ['binned_feature', 'good', 'bad', 'sum', 'count', 'bad rate', 'WoE']]

def fix_bin(bin_list):
    txt_list = bin_list.split(",")
    bin_list = list(map(float, txt_list[1:-1]))
    return [-np.inf] + bin_list + [np.inf]


def display_corr(corr):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust font size
    ax = sns.heatmap(corr, annot=True, fmt='.2%', cmap='coolwarm', center=0, cbar=True,
                     xticklabels=corr.columns, yticklabels=corr.columns, annot_kws={'size': 12})

    # Set title
    plt.title('Feature Correlation Heatmap')
    return plt


def calculate_iv(df, col_features, target_col):
    # Example usage:
    # df: DataFrame with your WoE features and a binary target column 'target' where 1 = default and 0 = non-default
    # target_col: the column name of the binary target variable (e.g., 'target')
    # Assuming df contains the WoE features and a binary target column
    # iv_values = calculate_iv(df, 'target')
    # print(iv_values)

    iv_dict = {}

    # Loop through each feature (skip the target column)
    for column in col_features:
        # Skip the target column
        if column == target_col:
            continue

        # Calculate the distribution of good (1) and bad (0) for each bin (WoE feature)
        good_total = df[df[target_col] == 1].shape[0]
        bad_total = df[df[target_col] == 0].shape[0]

        # Group by the WoE values and calculate the distribution of good/bad for each bin
        grouped = df.groupby(column).agg({target_col: ['sum', 'count']})

        # Calculate the distribution of good and bad for each bin
        grouped['good'] = grouped[target_col]['sum'] / good_total
        grouped['bad'] = (grouped[target_col]['count'] - grouped[target_col]['sum']) / bad_total

        # Calculate the WoE for each bin
        grouped['WoE'] = np.log(grouped['good'] / grouped['bad'])

        # Calculate IV for the feature
        grouped['IV'] = (grouped['good'] - grouped['bad']) * grouped['WoE']

        # Sum up the IV for each feature
        iv_dict[column] = grouped['IV'].sum()
        iv_dict = pd.Series(iv_dict, name='Information Values')

    return iv_dict.sort_values(ascending=False)


def encode_with_decision_tree(df, categorical_features, target_variable, max_group, min_data_per_group):
    # Instantiate the encoder
    encoder = DecisionTreeEncoder(variables=categorical_features,
                                  param_grid={'max_leaf_nodes': [max_group],  # np.arange(2,max_bins).tolist()
                                              'min_samples_leaf': [min_data_per_group]},
                                  regression=False,
                                  encoding_method='ordered'
                                  )

    # Fit and transform the data
    df_encoded = encoder.fit_transform(df[categorical_features], df[target_variable])

    # Get the encoder mapping dictionary
    encoder_dict = encoder.encoder_dict_

    # build new encoder
    dict_new = dict.fromkeys(encoder_dict.keys())

    for feature in encoder_dict.keys():
        dict_temp = {}
        arr_ = list()

        for value in encoder_dict[feature]:
            arr_.append(encoder_dict[feature][value])

        arr_.sort()
        arr_ = list(set(arr_))

        for value in encoder_dict[feature]:
            dict_temp[value] = arr_.index(encoder_dict[feature][value])

        dict_new[feature] = dict_temp
        df_encoded[feature] = df[feature].map(dict_temp)

    # Combine the encoded columns with the original data (optional)
    return df_encoded, dict_new, encoder

def plot_eda_monthly(df, col_time, target_col):
    df_stats = df[[col_time, target_col]].groupby(col_time).agg(['mean', 'count'])[target_col]
    df_stats = df_stats.rename(columns={'count':'data count', 'mean':'target mean'}).reset_index()

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(8, 6))

    x_axis = np.arange(1, len(df_stats['month'])+1)
    x_ticklabels = pd.to_datetime(df_stats['month'], format='%Y%m').dt.strftime('%Y-%m').values.tolist()


    # Plot bar chart (left y-axis) with 'data count'
    ax1.bar(x_axis, df_stats['data count'], color='blue', label='Data Count')
    ax1.set_xticks(x_axis)
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

def automatic_feature_binning(df, cols_feat_num, target_col, max_bins, min_data_per_bin):
    """
    Perform automatic feature binning using DecisionTreeDiscretiser from feature-engine.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing numerical features.
    max_bins (int): Maximum number of bins for each feature.
    min_data_per_bin (int): Minimum number of data points required per bin.

    Returns:
    pd.DataFrame: DataFrame with discretized features.
    dict: Dictionary containing the features and their corresponding bin thresholds.
    """

    # Instantiate the DecisionTreeDiscretiser (Feature-engine)
    discretizer = DecisionTreeDiscretiser(
        variables=cols_feat_num,
        # Select numeric columns automatically
        param_grid= {'max_leaf_nodes':[max_bins], #np.arange(2,max_bins).tolist()
                     'min_samples_leaf':[min_data_per_bin]},
        bin_output='boundaries', # Return bin labels
        regression=False,
        precision=4
    )

    # Fit and transform the data
    df_binned = discretizer.fit_transform(df[cols_feat_num], df[target_col])

    # Extract bin thresholds for each feature
    # bin_thresholds = {feature: discretizer.binner_.bins[feature] for feature in discretizer.variables_}
    bin_thresholds = discretizer.binner_dict_
    output_dict = {}

    for feature, thresholds in bin_thresholds.items():
        # Check if thresholds is a dictionary (original format)
        if isinstance(thresholds, dict):
            thresholds_list = [value for key, value in sorted(thresholds.items())]
        elif isinstance(thresholds, list):  # Handle if the value is already a list
            thresholds_list = thresholds
        else:
            raise ValueError("Input data must be in dictionary or list format.")

        output_dict[feature] = thresholds_list
        print(thresholds_list)
    return df_binned, output_dict


    # Model training and evaluation
    # ... (Add model training and evaluation code here)

if __name__ == "__main__":
    main()