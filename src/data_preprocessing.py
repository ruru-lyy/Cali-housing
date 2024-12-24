## DATA CLEANING

# libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

'''The reason why the below functions take only training set is so that our model can adapt to real-world, unseen data which are often uncleaned and contains missing vals'''

def feature_selection(df,target_col):
    '''This function separates the predictors and the target variable'''
    
    if target_col not in df.columns:
        raise ValueError(f"The target column '{target_col}' is not in the DataFrame.")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col].copy()

    return X, y

def add_new_features(df): # Feature Engineering
    """Adds new features that are derived from existing columns."""
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['population_per_household'] = df['population'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    return df


def create_full_pipeline(df, ordinal_encoding=True):
    '''This function creates a pipeline for preprocessing numerical and categorical data'''
    # Extract numerical and categorical columns
    num_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)
    cat_cols = list(df.select_dtypes(include='object').columns)

    # Debugging: Print detected columns
    print("Numerical columns detected:", num_cols)
    print("Categorical columns detected:", cat_cols)

    # Define the preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder() if ordinal_encoding else OneHotEncoder(sparse_output=False))
        ]), cat_cols)
    ])

    print("Pipeline created successfully.")
    return preprocessor

