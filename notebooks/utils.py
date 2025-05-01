import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

def load_dataset(filename: str, folder_path: str = 'data/raw') -> pd.DataFrame:

    file_path = os.path.join(
        os.path.dirname(os.getcwd()),
        folder_path,
        filename
    )

    try:
        if filename.endswith('.xlsx'):
            df=pd.read_excel(file_path)
            print('Excel Dataset Loaded Successfully!')
        elif filename.endswith('.csv'):
            df=pd.read_csv(file_path)
            print('CSV Dataset Loaded Successfully!')
        else:
            raise ValueError("Unsupported file format. Please use .csv or .xlsx files.")

        print(f" Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns.")
        
        return df

    except FileNotFoundError as e:
        print(f" {e}")
    except Exception as e:
        print(f" Unexpected error occurred: {str(e)}")
        


def convert_dtypes(df:pd.DataFrame, columns:list)-> pd.DataFrame:
    for col in columns:
        df[col] = df[col].astype('int32')
    return df



def categorical_univariate_analysis_clean(df: pd.DataFrame, categorical_columns: list, top_n: int = 10):
    """
    Visualizes the distribution of categorical features, handling too many unique values.
    """
    sns.set_theme(style="whitegrid")
    n_cols = 2
    n_rows = math.ceil(len(categorical_columns) / n_cols)
    plt.figure(figsize=(6*n_cols, 5*n_rows))
    plt.suptitle('Categorical Features Distribution (Cleaned)', fontsize=24, fontweight='bold', color='Black', y=1.02, style='italic')

    for idx, col in enumerate(categorical_columns, start=1):
        ax = plt.subplot(n_rows, n_cols, idx)
    
        if df[col].nunique() > top_n:
            top_categories = df[col].value_counts().nlargest(top_n).index
            df[col + '_cleaned'] = df[col].apply(lambda x: x if x in top_categories else 'Other')
            plot_data = df[col + '_cleaned']
        else:
            plot_data = df[col]

        sns.countplot(x=plot_data, palette='pastel', order=plot_data.value_counts().index, ax=ax)
        ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=3.0)
    plt.show()




def numeric_univariate_analysis(df: pd.DataFrame, numerical_columns: list, hist_cols: list):
    """
    Visualizes distribution, checks for skewness, and identifies outliers 
    for numerical features.
    """
    sns.set_theme(style="whitegrid")

    n_cols = 3
    n_rows = math.ceil(len(numerical_columns) / n_cols)

    plt.figure(figsize=(6*n_cols, 5*n_rows))
    plt.suptitle('Numerical Features Distribution', fontsize=24, fontweight='bold', color='Black', y=1.02, style='italic')

    for idx, col in enumerate(numerical_columns, start=1):
        ax = plt.subplot(n_rows, n_cols, idx)
        
        if col in hist_cols:
            sns.histplot(data=df, x=col, bins=50, kde=True, color='skyblue', ax=ax)
            ax.set_ylabel('Count', fontsize=12)
        else:
            sns.countplot(data=df, x=col, palette='Set2', stat='percent', ax=ax)
            ax.set_ylabel('Percentage', fontsize=12)

        ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel(col, fontsize=12)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=3.0)
    plt.show()


def bivariate_plots(df):
    target='Price'
    num_cols = [col for col in df.select_dtypes(exclude='O').columns if col != target]
    n_cols=3
    n_rows=math.ceil(len(num_cols)/n_cols)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 5 * n_rows))
    plt.suptitle('Bivariate Analysis of numerical featuers', fontsize=24, fontweight='bold', color='Black', y=1.01, style='italic')
    for idx,column in enumerate(num_cols):
        ax=plt.subplot(n_rows,n_cols,idx+1)
        sns.scatterplot(data=df,x=column,y=target,alpha=0.5)
        ax.set_title(f'{column} vs {target} ', fontsize=12, fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel(target)
    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=3.0)
    plt.show()  


def bivariate_cat_plots(df):
    target='Price'
    cat_columns=df.select_dtypes(include="O").drop(columns=['Route','Airline_cleaned']).columns
    n_cols=2
    n_rows=math.ceil(len(cat_columns)/n_cols)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6*n_cols, 5 * n_rows))
    plt.suptitle('Bivariate Analysis of numerical & categorical featuers', fontsize=24, fontweight='bold', color='Black', y=1.01, style='italic')
    for idx,column in enumerate(cat_columns):
        ax=plt.subplot(n_rows,n_cols,idx+1)
        order = df.groupby(column)[target].mean().sort_values(ascending=False).index
        sns.barplot(data=df, x=column, y=target, estimator=np.mean, palette='Set2',order=order, ax=ax,ci='sd')
        ax.set_title(f'{column} vs {target} Distribution', fontsize=10, fontweight='bold')
        ax.set_xlabel(column,fontsize=12)
        ax.set_ylabel(target,fontsize=12)
        ax.tick_params(axis='x',rotation=90)
    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=3.0)
    plt.show()  



def plot_correlation_heatmap(df):
    num_cols = df.select_dtypes(exclude='O').columns
    corr = df[num_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, square=True, cbar_kws={"shrink": 0.75})
    plt.title("Correlation Heatmap of Numerical Features", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def assign_time_bin(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'


def detect_outliers_iqr(data, continuous_columns):
    """
    Detects outliers using IQR method for continuous columns.
    
    """
    outlier_summary = []

    for col in continuous_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_count = outliers.shape[0]
        outlier_percentage = (outlier_count / len(data)) * 100

        outlier_summary.append({
            'Column': col,
            'Outlier Count': outlier_count,
            'Outlier Percentage': round(outlier_percentage, 2)
        })

    outlier_df = pd.DataFrame(outlier_summary)
    print(" Outlier Summary (IQR Method):\n")
    return outlier_df

def plot_boxplots(data, columns, n_cols=3, figsize=(20, 20), color_palette='Set3'):
    """
    Plots boxplots for the given numerical columns to check for outliers.
    """

    sns.set(style="whitegrid")
    n_rows = (len(columns) + n_cols - 1) // n_cols

    plt.figure(figsize=figsize, facecolor='white')
    plt.suptitle(" Boxplots for Outlier Detection", fontsize=24, fontweight='bold', color='black', y=1.02)

    for idx, column in enumerate(columns, 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        sns.boxplot(y=data[column], palette=color_palette, ax=ax)
        ax.set_title(f'{column}', fontsize=16)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('')
        ax.grid(True)

    plt.tight_layout(pad=2.0)
    plt.show()
