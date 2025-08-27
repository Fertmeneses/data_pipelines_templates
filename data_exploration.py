# data_exploration.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================
# Explore leaderboard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================
# Explore leaderboard

def explore_LB(
    LB_file,target_col,cum_plot=True,
    bins=100,x_lim=None,figsize=(10,5)
    ):
    """
    Study the Leaderboard by making a histogram and (optionally) a cumulative plot
    """
    # Get data:
    LB_data = pd.read_csv(LB_file)
    # Make figure:
    fig, ax1 = plt.subplots(figsize=figsize)
    # Histogram:
    counts, bins, patches = ax1.hist(
        LB_data[target_col],bins=bins,edgecolor="navy",
        alpha=0.8,label='Histogram')
    ax1.set_title(f'Leaderboard {target_col} histogram')
    ax1.set_xlabel(target_col)
    ax1.set_ylabel("Frequency")
    # Prepare handles for legend
    handles, labels = ax1.get_legend_handles_labels()
    # Cumulative (optional):
    if cum_plot:
        ax2 = ax1.twinx()
        cum_counts = counts.cumsum()
        cum_counts = cum_counts / cum_counts[-1]  # normalize to 1
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        line, = ax2.plot(
            bin_centers, cum_counts, color="orange", 
            linewidth=2, label="Cumulative")
        ax2.set_ylabel("Cumulative proportion")
        # Add cumulative to legend
        handles.append(line)
        labels.append("Cumulative")
    # Unified legend
    ax1.legend(handles, labels, loc="upper left")
    # Set x limits if required:
    if x_lim is not None:
        plt.xlim(x_lim)

# ===================
# Load dataset and get basic information:

def load_dataset(
    dataset_file,id_col=None,
    ):
    """
    Load a dataset and identify the id column, if any.
    The output is the dataset. If {id_col} is provided, then the output dataset
    and id column are provided as separate variables.
    """
    # Load original datasets:
    df = pd.read_csv(dataset_file)
    # Print some data examples:
    display(df.head())
    # Separate id column if required:
    if id_col is not None:
        ids = df[id_col]
        df = df.drop(columns=[id_col])
    print('_'*40+'\n\nInformation about dataset:\n')
    df.info()
    return df, ids if id_col is not None else df

# ===================
# Plot target distribution

def plot_target_dist(
    df, target_col, pie_threshold=10, reg_bins=50
    ):
    """
    Plot the target distribution in a dataset.
    --- Inputs ---
    {df} [pandas.DataFrame]: Input dataframe.
    {target_col} [str]: Name of the target column.
    {pie_threshold} [int]: For a classification task, it defines the maximum number
    of classes for which the output will be a pie chart, else it will be a bar plot.
    {reg_bins} [int]: For regression, define the number of bins for the histogram.
    
    --- Output ---
    - If the target column is numeric and has more than {pie_threshold} unique values,
    then plot a histogram.
    - If the target column is categorical (or numeric with few unique values):
        - Pie chart if classes <= {pie_threshold}.
        - Horizontal bar plot if classes > {pie_threshold}.
    """
    # Identify target column:
    target = df[target_col]
    nunique = target.nunique()
    target_type = 'Numeric' if pd.api.types.is_numeric_dtype(target) else 'Object'
    
    # If target is numeric and have few unique values, plot histogram:
    if pd.api.types.is_numeric_dtype(target) and nunique > pie_threshold:
        plt.figure(figsize=(8, 5))
        plt.hist(target, bins=reg_bins, edgecolor="navy", alpha=0.7)
        plt.title(f"Histogram of {target_col} (Regression)")
        plt.xlabel(target_col)
        plt.ylabel("Frequency")
        plt.show()
    # Else, plot either a pie chart of horizontal bar plot:
    else:
        value_counts = target.value_counts().sort_values(ascending=False)
        if nunique <= pie_threshold:
            # Pie chart
            plt.figure(figsize=(6, 6))
            wedges, texts, autotexts = plt.pie(
                value_counts,labels=None,autopct="%1.1f%%",startangle=90)
            plt.title(f"Class distribution of {target_col} ({target_type})")
            plt.legend(wedges, value_counts.index,
                title="Classes",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
            plt.show()
        else:
            # Horizontal bar plot
            plt.figure(figsize=(10, 6))
            value_counts.plot(kind="barh", color="steelblue", edgecolor="black")
            plt.title(f"Class distribution of {target_col} ({target_type})")
            plt.xlabel("Frequency")
            plt.ylabel("Classes")
            plt.gca().invert_yaxis()
            plt.show() 