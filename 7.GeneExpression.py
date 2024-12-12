import pandas as pd
import numpy as np

def calculate_gene_expression(expression_data):
    """
    Calculate gene expression levels from provided data.

    Parameters:
    expression_data (pandas DataFrame): A DataFrame with genes as rows and conditions as columns.

    Returns:
    pandas DataFrame: A DataFrame with normalized gene expression levels.
    """
    # Normalize expression data (log2 transformation)
    log_transformed = np.log2(expression_data + 1)

    # Compute mean expression across all conditions for each gene
    mean_expression = log_transformed.mean(axis=1)

    # Calculate fold change (log2 ratio) relative to a baseline condition
    baseline_condition = log_transformed.iloc[:, 0]
    fold_change = log_transformed.subtract(baseline_condition, axis=0)

    # Combine results into a new DataFrame
    results = pd.DataFrame({
        'Gene': expression_data.index,
        'Mean_Expression': mean_expression,
        'Fold_Change': fold_change.mean(axis=1)
    })

    return results

# Example usage with a synthetic dataset
# Suppose 'data' is a pandas DataFrame containing gene expression data
data = pd.DataFrame({
    'Gene': ['Gene1', 'Gene2', 'Gene3', 'Gene4'],
    'Condition1': [5, 3, 6, 2],
    'Condition2': [8, 5, 7, 3],
    'Condition3': [6, 4, 8, 1]
}).set_index('Gene')

gene_expression_results = calculate_gene_expression(data)
print('Bharath C')
print('1BM22CS068')
print(gene_expression_results)
