import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def explore_data(df: pd.DataFrame) -> None:
    """Perform initial data exploration."""
    print("Top of the dataset:")
    print(df.head())
    print("\nBottom of the dataset:")
    print(df.tail())
    print("\nInformation about the dataset:")
    print(df.info())
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())
    print("\nNumber of duplicate rows in the dataset:")
    print(df.duplicated().sum())

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by dropping duplicates."""
    df.drop_duplicates(keep='first', inplace=True)
    print("\nNumber of duplicate rows after cleaning:")
    print(df.duplicated().sum())
    return df

def describe_data(df: pd.DataFrame) -> None:
    """Provide descriptive statistics for numerical columns."""
    print("\nDescriptive statistics for numerical columns:")
    print(df.describe())

def plot_data(df: pd.DataFrame) -> None:
    """Create and save various plots for data visualization."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    sns.boxplot(x=df['salary'], ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_xlabel('Salary', fontsize=12)
    axes[0, 0].set_title('Box Plot of Salary', fontsize=14)

    axes[0, 1].hist(df['salary'], bins=20, edgecolor='black')
    axes[0, 1].set_xlabel('Salary', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Distribution of Salary', fontsize=14)

    sns.countplot(y='job_category', data=df, ax=axes[0, 2], palette='pastel')
    axes[0, 2].set_xlabel('Count', fontsize=12)
    axes[0, 2].set_ylabel('Job Category', fontsize=12)
    axes[0, 2].set_title('Distribution of Job Categories', fontsize=14)
    axes[0, 2].tick_params(axis='y', labelrotation=45)

    sns.regplot(x='work_year', y='salary', data=df, ax=axes[1, 0], scatter_kws={'color': 'skyblue', 'alpha': 0.5})
    axes[1, 0].set_xlabel('Years of Experience', fontsize=12)
    axes[1, 0].set_ylabel('Salary', fontsize=12)
    axes[1, 0].set_title('Salary vs. Years of Experience', fontsize=14)

    corr_matrix = df[['salary', 'work_year', 'salary_in_usd']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Heatmap', fontsize=14)

    employment_counts = df['employment_type'].value_counts()
    axes[1, 2].pie(employment_counts, labels=employment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    axes[1, 2].set_title('Distribution of Employment Types', fontsize=14)

    sns.violinplot(x='job_category', y='salary', data=df, ax=axes[2, 0], palette='pastel')
    axes[2, 0].set_xlabel('Job Category', fontsize=12)
    axes[2, 0].set_ylabel('Salary', fontsize=12)
    axes[2, 0].set_title('Salary Distribution by Job Category', fontsize=14)
    axes[2, 0].tick_params(axis='x', labelrotation=45)

    sns.regplot(x='salary', y='salary_in_usd', data=df, ax=axes[2, 1], scatter_kws={'color': 'skyblue', 'alpha': 0.5})
    axes[2, 1].set_xlabel('Salary', fontsize=12)
    axes[2, 1].set_ylabel('Salary (USD)', fontsize=12)
    axes[2, 1].set_title('Salary vs. Salary in USD', fontsize=14)

    fig.delaxes(axes[2, 2])
    plt.tight_layout()
    plt.savefig('deep_analysis_plots_with_regression.png', dpi=300, bbox_inches='tight')
    plt.show()

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and handle outliers in the dataset."""
    # Compute the interquartile range (IQR) for the 'salary' column
    Q1 = df['salary'].quantile(0.25)
    Q3 = df['salary'].quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df['salary'] < lower_bound) | (df['salary'] > upper_bound)]
    print("Identified outliers:")
    print(outliers)

    # Plot the box plot of 'salary' with outliers highlighted
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['salary'], color='skyblue')
    plt.scatter(outliers.index, outliers['salary'], color='red', label='Outliers')
    plt.legend()
    plt.xlabel('Salary', fontsize=12)
    plt.title('Box Plot of Salary with Outliers', fontsize=14)
    plt.show()

    # Apply log transformation to the 'salary' column
    df['salary'] = np.log1p(df['salary'])  # Using np.log1p to handle zero values

    # Apply robust scaling to the 'work_year' and 'salary' columns
    scaler = RobustScaler()
    df[['work_year', 'salary']] = scaler.fit_transform(df[['work_year', 'salary']])

    # Clip extreme values in the 'work_year' and 'salary' columns
    clip_threshold = 3  # Adjust as needed
    df['work_year'] = np.clip(df['work_year'], -clip_threshold, clip_threshold)
    df['salary'] = np.clip(df['salary'], -clip_threshold, clip_threshold)

    return df

def build_and_evaluate_models(df: pd.DataFrame) -> None:
    """Build and evaluate regression models."""
    X = df[['work_year']]
    y = df['salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)

    print("\nLinear Regression Model:")
    print("Training Set Performance:")
    print("MSE:", mean_squared_error(y_train, y_train_pred))
    print("R-squared:", r2_score(y_train, y_train_pred))
    print("\nTesting Set Performance:")
    print("MSE:", mean_squared_error(y_test, y_test_pred))
    print("R-squared:", r2_score(y_test, y_test_pred))

    plt.figure(figsize=(10, 5))
    plt.scatter(y_train, y_train_pred, color='blue', label='Training Set')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
    plt.title('Linear Regression: Actual vs Predicted (Training Set)')
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.legend()
    plt.grid(True)
    plt.show()

    # SVR
    svr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    param_grid = {
        'svr__kernel': ['linear', 'rbf'],
        'svr__C': [0.1, 1, 10],
        'svr__epsilon': [0.1, 0.2, 0.5]
    }

    grid_search = GridSearchCV(svr_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_svr_model = grid_search.best_estimator_

    y_train_pred = best_svr_model.predict(X_train)
    y_test_pred = best_svr_model.predict(X_test)

    print("\nSVR Model:")
    print("Training Set Performance:")
    print("MSE:", mean_squared_error(y_train, y_train_pred))
    print("R-squared:", r2_score(y_train, y_train_pred))
    print("\nTesting Set Performance:")
    print("MSE:", mean_squared_error(y_test, y_test_pred))
    print("R-squared:", r2_score(y_test, y_test_pred))

    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_test_pred, color='orange', label='Testing Set')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('SVR: Actual vs Predicted (Testing Set)')
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.legend()
    plt.grid(True)
    plt.show()

    # XGBoost
    xgb_regressor = XGBRegressor(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5]
    }

    grid_search = GridSearchCV(xgb_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_xgb_model = grid_search.best_estimator_

    y_train_pred = best_xgb_model.predict(X_train)
    y_test_pred = best_xgb_model.predict(X_test)

    print("\nXGBoost Model:")
    print("Training Set Performance:")
    print("MSE:", mean_squared_error(y_train, y_train_pred))
    print("R-squared:", r2_score(y_train, y_train_pred))
    print("\nTesting Set Performance:")
    print("MSE:", mean_squared_error(y_test, y_test_pred))
    print("R-squared:", r2_score(y_test, y_test_pred))

    plt.figure(figsize=(10, 5))
    plt.scatter(y_train, y_train_pred, color='blue', label='Training Set')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
    plt.title('XGBoost: Actual vs Predicted (Training Set)')
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Comparison Plot
    models = ['Linear Regression', 'SVR', 'XGBoost']
    train_r2 = [r2_score(y_train, lr_model.predict(X_train)), 
                r2_score(y_train, best_svr_model.predict(X_train)), 
                r2_score(y_train, best_xgb_model.predict(X_train))]
    test_r2 = [r2_score(y_test, lr_model.predict(X_test)), 
               r2_score(y_test, best_svr_model.predict(X_test)), 
               r2_score(y_test, best_xgb_model.predict(X_test))]

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(models))

    plt.bar(index, train_r2, bar_width, label='Training Set', color='blue')
    plt.bar([i + bar_width for i in index], test_r2, bar_width, label='Testing Set', color='orange')

    plt.xlabel('Models')
    plt.ylabel('R-squared')
    plt.title('Comparison of R-squared for Different Models')
    plt.xticks([i + bar_width / 2 for i in index], models)
    plt.legend()
    plt.tight_layout()
    plt.show()
