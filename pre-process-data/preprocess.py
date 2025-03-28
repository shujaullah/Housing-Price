# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# import os
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Set random seed for reproducibility
# np.random.seed(42)

# def missing_values_analysis(df):
#     """Calculate missing values percentage"""
#     missing = df.isnull().sum()
#     missing_percent = (missing / len(df)) * 100
#     missing_values = pd.DataFrame({'Missing Count': missing, 'Missing Percentage': missing_percent})
#     return missing_values[missing_values['Missing Count'] > 0].sort_values('Missing Percentage', ascending=False)

# def handle_missing_values(df):
#     """Handle missing values based on data description"""
#     # Numerical features to impute with median
#     numerical_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
    
#     # Categorical features to impute with 'None'
#     categorical_none = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
#                        'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
#                        'PoolQC', 'Fence', 'MiscFeature']
    
#     # Handle numerical missing values
#     for feature in numerical_features:
#         df[feature] = df[feature].fillna(df[feature].median())
    
#     # Handle categorical missing values
#     for feature in categorical_none:
#         df[feature] = df[feature].fillna('None')
    
#     # Fill remaining categorical variables with mode
#     categorical_mode = df.select_dtypes(include=['object']).columns
#     for feature in categorical_mode:
#         if feature not in categorical_none and feature != 'is_train':
#             df[feature] = df[feature].fillna(df[feature].mode()[0])
    
#     return df

# def analyze_correlations(df, threshold=0.7):
#     """Analyze and visualize correlations between features"""
#     # Calculate correlation matrix
#     correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    
#     # Find highly correlated feature pairs
#     high_correlations = []
#     for i in range(len(correlation_matrix.columns)):
#         for j in range(i):
#             if abs(correlation_matrix.iloc[i, j]) > threshold:
#                 high_correlations.append({
#                     'feature1': correlation_matrix.columns[i],
#                     'feature2': correlation_matrix.columns[j],
#                     'correlation': correlation_matrix.iloc[i, j]
#                 })
    
#     # Sort by absolute correlation value
#     high_correlations = sorted(high_correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
#     # Create correlation plot
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
#     plt.title('Feature Correlation Matrix')
#     plt.tight_layout()
#     plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'correlation_matrix.png'))
#     plt.close()
    
#     return high_correlations

# def handle_collinearity(df, high_correlations, target_column='SalePrice'):
#     """Handle highly correlated features"""
#     features_to_drop = set()
    
#     print("\nHighly correlated feature pairs:")
#     print("Feature 1 | Feature 2 | Correlation")
#     print("-" * 50)
    
#     for corr in high_correlations:
#         print(f"{corr['feature1']} | {corr['feature2']} | {corr['correlation']:.3f}")
        
#         # If one feature is already marked for removal, skip
#         if corr['feature1'] in features_to_drop or corr['feature2'] in features_to_drop:
#             continue
            
#         # If dealing with training data and target column exists
#         if target_column in df.columns:
#             # Calculate correlation with target for both features
#             corr1 = abs(df[corr['feature1']].corr(df[target_column]))
#             corr2 = abs(df[corr['feature2']].corr(df[target_column]))
            
#             # Keep the feature with higher correlation to target
#             if corr1 < corr2:
#                 features_to_drop.add(corr['feature1'])
#             else:
#                 features_to_drop.add(corr['feature2'])
#         else:
#             # If no target column, drop the second feature by default
#             features_to_drop.add(corr['feature2'])
    
#     print(f"\nFeatures to be removed due to high correlation: {features_to_drop}")
    
#     # Drop highly correlated features
#     df = df.drop(columns=list(features_to_drop))
    
#     return df

# def engineer_features(df):
#     """Create new features"""
#     # Total square footage
#     df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
#     # Total bathrooms
#     df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + \
#                           df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    
#     # House age and remodeled
#     df['Age'] = df['YrSold'] - df['YearBuilt']
#     df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
#     df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    
#     # Total porch area
#     df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + \
#                          df['3SsnPorch'] + df['ScreenPorch']
    
#     # Has features
#     df['HasPool'] = (df['PoolArea'] > 0).astype(int)
#     df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
#     df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    
#     # Overall quality score
#     df['QualityScore'] = df['OverallQual'] * df['OverallCond']
    
#     return df

# def encode_categorical_features(df):
#     """Encode categorical features"""
#     # Initialize label encoder
#     le = LabelEncoder()
    
#     # Get categorical columns
#     categorical_features = df.select_dtypes(include=['object']).columns
    
#     # Encode each categorical feature
#     for feature in categorical_features:
#         if feature != 'is_train':
#             df[feature] = le.fit_transform(df[feature].astype(str))
    
#     return df

# def scale_features(df):
#     """Scale numerical features"""
#     # Initialize scaler
#     scaler = StandardScaler()
    
#     # Get numerical columns
#     numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    
#     # Drop 'is_train' if it exists
#     if 'is_train' in numerical_features:
#         numerical_features = numerical_features.drop('is_train')
    
#     # Scale numerical features
#     df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
#     return df

# def main():
#     # Get the current directory
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)
#     data_dir = os.path.join(parent_dir, 'house-prices-advanced-regression-techniques')
    
#     # Load datasets
#     print("Loading datasets...")
#     train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
#     test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))

#     # Create train/test flag
#     train_data['is_train'] = 1
#     test_data['is_train'] = 0

#     # Combine datasets
#     all_data = pd.concat([train_data, test_data], axis=0, sort=False)
#     print(f"Total samples: {len(all_data)}")
#     print(f"Training samples: {len(train_data)}")
#     print(f"Test samples: {len(test_data)}")

#     # Save original SalePrice for training data
#     train_sale_price = train_data['SalePrice'].copy() if 'SalePrice' in train_data.columns else None

#     # Analyze missing values
#     print("\nAnalyzing missing values...")
#     print(missing_values_analysis(all_data))

#     # Handle missing values
#     print("\nHandling missing values...")
#     all_data = handle_missing_values(all_data)

#     # Engineer features
#     print("\nEngineering new features...")
#     all_data = engineer_features(all_data)

#     # Encode categorical features
#     print("\nEncoding categorical features...")
#     all_data = encode_categorical_features(all_data)

#     # Analyze correlations
#     print("\nAnalyzing feature correlations...")
#     high_correlations = analyze_correlations(all_data, threshold=0.7)
    
#     # Handle collinearity
#     print("\nHandling multicollinearity...")
#     all_data = handle_collinearity(all_data, high_correlations)

#     # Scale features
#     print("\nScaling features...")
#     all_data = scale_features(all_data)

#     # Split back into train and test
#     processed_train = all_data[all_data['is_train'] == 1].drop('is_train', axis=1)
#     processed_test = all_data[all_data['is_train'] == 0].drop('is_train', axis=1)

#     # Add back SalePrice to training data
#     if train_sale_price is not None:
#         processed_train['SalePrice'] = train_sale_price

#     # Create output directory if it doesn't exist
#     output_dir = os.path.join(current_dir)
#     os.makedirs(output_dir, exist_ok=True)

#     # Export processed datasets
#     print("\nExporting processed datasets...")
#     processed_train.to_csv(os.path.join(output_dir, 'processed_train.csv'), index=False)
#     processed_test.to_csv(os.path.join(output_dir, 'processed_test.csv'), index=False)

#     print(f"\nProcessed training data shape: {processed_train.shape}")
#     print(f"Processed test data shape: {processed_test.shape}")
#     print("\nPreprocessing completed! Files saved as 'processed_train.csv' and 'processed_test.csv'")
#     print("Correlation matrix visualization saved as 'correlation_matrix.png'")

# if __name__ == "__main__":
#     main() 

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore, skew
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

np.random.seed(42)

# ---- Helper Functions ----
def missing_values_analysis(df):
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_values = pd.DataFrame({'Missing Count': missing, 'Missing Percentage': missing_percent})
    return missing_values[missing_values['Missing Count'] > 0].sort_values('Missing Percentage', ascending=False)

def handle_missing_values(df):
    numerical_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
    categorical_none = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                        'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                        'PoolQC', 'Fence', 'MiscFeature']
    for feature in numerical_features:
        df[feature] = df[feature].fillna(df[feature].median())
    for feature in categorical_none:
        df[feature] = df[feature].fillna('None')
    categorical_mode = df.select_dtypes(include=['object']).columns
    for feature in categorical_mode:
        if feature not in categorical_none and feature != 'is_train':
            df[feature] = df[feature].fillna(df[feature].mode()[0])
    return df

def engineer_features(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['QualityScore'] = df['OverallQual'] * df['OverallCond']
    return df

def log_transform_skewed_features(df, threshold=0.75):
    numeric_feats = df.select_dtypes(include=['int64', 'float64']).columns
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > threshold].index
    for feat in high_skew:
        if feat not in ['SalePrice', 'LogSalePrice']:
            df[feat] = np.log1p(df[feat])
    return df

def encode_categorical_features(df):
    le = LabelEncoder()
    categorical_features = df.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        if feature != 'is_train':
            df[feature] = le.fit_transform(df[feature].astype(str))
    return df

def analyze_correlations(df, threshold=0.7):
    correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    high_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                high_correlations.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': correlation_matrix.iloc[i, j]
                })
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    return high_correlations

def handle_collinearity(df, high_correlations, target_column='SalePrice'):
    features_to_drop = set()
    protected_features = {'TotalSF', 'GrLivArea'}
    for corr in high_correlations:
        if corr['feature1'] in features_to_drop or corr['feature2'] in features_to_drop:
            continue
        if corr['feature1'] in protected_features or corr['feature2'] in protected_features:
            continue
        if target_column in df.columns:
            corr1 = abs(df[corr['feature1']].corr(df[target_column]))
            corr2 = abs(df[corr['feature2']].corr(df[target_column]))
            if corr1 < corr2:
                features_to_drop.add(corr['feature1'])
            else:
                features_to_drop.add(corr['feature2'])
        else:
            features_to_drop.add(corr['feature2'])
    df = df.drop(columns=list(features_to_drop))
    return df

def scale_features(df, save_path=None):
    scaler = StandardScaler()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_features = numerical_features.drop(['SalePrice', 'LogSalePrice'], errors='ignore')
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    if save_path:
        joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))
    return df

def remove_outliers(df, column, threshold=3):
    z_scores = zscore(df[column])
    return df[np.abs(z_scores) < threshold]

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_data = pd.read_csv(os.path.join(current_dir, "train.csv"))
    test_data = pd.read_csv(os.path.join(current_dir, "test.csv"))

    train_data['is_train'] = 1
    test_data['is_train'] = 0
    all_data = pd.concat([train_data, test_data], axis=0, sort=False)
    train_sale_price = train_data['SalePrice'].copy()

    print("\nMissing values:")
    print(missing_values_analysis(all_data))

    all_data = handle_missing_values(all_data)
    all_data = engineer_features(all_data)
    all_data = log_transform_skewed_features(all_data)
    all_data = encode_categorical_features(all_data)
    high_corrs = analyze_correlations(all_data)
    all_data = handle_collinearity(all_data, high_corrs)

    all_data['SalePrice'] = pd.concat([train_sale_price, pd.Series([np.nan]*len(test_data))], ignore_index=True)
    all_data['LogSalePrice'] = np.log1p(all_data['SalePrice'])

    processed_train = all_data[all_data['is_train'] == 1].drop(columns=['is_train'])
    processed_test = all_data[all_data['is_train'] == 0].drop(columns=['is_train', 'SalePrice', 'LogSalePrice'])

    processed_train = remove_outliers(processed_train, 'TotalSF')
    processed_train = remove_outliers(processed_train, 'GrLivArea')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(train_sale_price, kde=True)
    plt.title("Original SalePrice")
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(train_sale_price), kde=True)
    plt.title("Log(SalePrice)")
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'saleprice_distribution.png'))
    plt.close()

    processed_train = scale_features(processed_train, save_path=current_dir)
    processed_test = scale_features(processed_test, save_path=current_dir)

    X = processed_train.drop(columns=['SalePrice', 'LogSalePrice'])
    y = processed_train['LogSalePrice']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    processed_train.to_csv(os.path.join(current_dir, "processed_train.csv"), index=False)
    processed_test.to_csv(os.path.join(current_dir, "processed_test.csv"), index=False)
    X_train.to_csv(os.path.join(current_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(current_dir, "X_val.csv"), index=False)
    y_train.to_csv(os.path.join(current_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(current_dir, "y_val.csv"), index=False)

    print("\n✅ Preprocessing complete.")
    print("Files saved: processed_train.csv, processed_test.csv, X_train.csv, X_val.csv, y_train.csv, y_val.csv, scaler.pkl, correlation_matrix.png, saleprice_distribution.png")

if __name__ == "__main__":
    main()
