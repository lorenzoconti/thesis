import pandas as pd
import numpy as np


def correlations_to_latex(df, output_path):
    """
    Exports the correlations table result to a LaTeX file.

    Parameters:
        df        (DataFrame) : The DataFrame to be exported
        output_path  (string) : The path where to store the file

    Returns:
        None
    """
    with open(output_path, 'w') as file:
        file.write(df.to_latex().replace('±', '$\pm$'))


def regression_results_to_latex(observed, predicted_nn, predicted_nn_gp, output_path):
    pass


