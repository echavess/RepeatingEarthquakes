import os, glob
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from CWI_reps import decorr_index

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def repeater_classification(waveforms_path, station_name, bandpass_filter, windowing, trace_len):

    if not os.path.exists(waveforms_path):
        print(f"The Waveform path declared: {waveforms_path} does not exists, please check")
    else: 
        waveforms_list = glob.glob(os.path.join(waveforms_path, '*SAC'))

        station_name=station_name

        decorr_index(station_name=station_name, data_list=waveforms_list, 
                    trace_len=trace_len, bandpass_filter=bandpass_filter, 
                    windowing=windowing)


def classification(input_correlation_file, band_pass_filter, station_name, waveform_directory):

    data = pd.read_csv(input_correlation_file, delim_whitespace=True, header=None)

    # Rename the columns for better readability
    data.columns = ['event_id', 'correlation_coefficient', 'lag_time', 'time1', 'time2', 'delta_time', 'amplitude1', 'amplitude2']

    # Remove the first row which contains the header information
    data_cleaned = data.iloc[1:].reset_index(drop=True)

    # Convert columns to appropriate data types
    data_cleaned['event_id'] = data_cleaned['event_id'].astype(int)
    data_cleaned['correlation_coefficient'] = data_cleaned['correlation_coefficient'].astype(float)
    data_cleaned['lag_time'] = data_cleaned['lag_time'].astype(float)
    data_cleaned['time1'] = pd.to_datetime(data_cleaned['time1'])
    data_cleaned['time2'] = pd.to_datetime(data_cleaned['time2'])
    data_cleaned['delta_time'] = data_cleaned['delta_time'].astype(float)
    data_cleaned['amplitude1'] = data_cleaned['amplitude1'].astype(float)
    data_cleaned['amplitude2'] = data_cleaned['amplitude2'].astype(float)

    # Extract correlation coefficients for clustering
    X = data_cleaned[['correlation_coefficient']].values

    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1, metric='euclidean', linkage='ward')
    labels = clustering.fit_predict(X)

    # Add the labels to the DataFrame
    data_cleaned['family'] = labels
    data_cleaned_sorted = data_cleaned.sort_values(by='family').reset_index(drop=True)
    
    # Create the figure with multiple subplots again
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5))

    # Correlation Coefficient Distribution
    sns.histplot(data_cleaned_sorted['correlation_coefficient'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Correlation Coefficient Distribution')
    axes[0, 0].set_xlabel('Correlation Coefficient')
    axes[0, 0].set_ylabel('Frequency')

    # Time Series Plot for Correlation Coefficients
    sns.lineplot(x='time1', y='correlation_coefficient', data=data_cleaned_sorted, ax=axes[0, 1])
    axes[0, 1].set_title('Correlation Coefficient Over Time')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Correlation Coefficient')
    # Rotate x-axis ticks
    for tick in axes[0, 1].get_xticklabels():
        tick.set_rotation(45)

    # Scatter plot of Clustering Results
    sns.scatterplot(x='correlation_coefficient', y='delta_time', hue='family', palette='tab10', data=data_cleaned_sorted, ax=axes[1, 0], lw=0.4)
    axes[1, 0].set_title('Clustering Results')
    axes[1, 0].set_xlabel('Correlation Coefficient')
    axes[1, 0].set_ylabel('Delta Time, s')
    axes[1, 0].legend(ncols=3, fontsize=7)

    # Line plot showing delta_time within the same cluster
    for label in data_cleaned_sorted['family'].unique():
        cluster_data = data_cleaned_sorted[data_cleaned['family'] == label]
        axes[1, 1].plot(cluster_data['time1'], cluster_data['delta_time'], label=f'Cluster {label}')
    axes[1, 1].set_title('Delta Time in Clusters')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Delta Time')
    axes[1, 1].legend(ncols=3, fontsize=7)
        # Rotate x-axis ticks
    for tick in axes[1, 1].get_xticklabels():
        tick.set_rotation(45)

    # Histogram of Lag Time
    sns.histplot(data_cleaned_sorted['lag_time'], kde=True, ax=axes[2, 1])
    axes[2, 1].set_title('Lag Time Distribution')
    axes[2, 1].set_xlabel('Lag Time')
    axes[2, 1].set_ylabel('Frequency')

    family_means = data_cleaned_sorted.groupby('family')['correlation_coefficient'].mean()
    family_distances = pdist(family_means.values.reshape(-1, 1), metric='euclidean')
    family_distance_matrix = squareform(family_distances)
    sns.heatmap(family_distance_matrix, annot=False, cmap='coolwarm', 
                xticklabels=family_means.index, 
                yticklabels=family_means.index, ax=axes[2,0], cbar_kws={'label': 'Euclidean Distance'})
    axes[2,0].set_title('Distance Between Families')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    #plt.show()
    fig.savefig(f"Event_classification_{band_pass_filter}_{station_name}.png", format='png', dpi=700)

    data_cleaned_sorted['time1'] = data_cleaned_sorted['time1'].dt.strftime('%Y-%m-%dT%H:%M')
    data_cleaned_sorted['time2'] = data_cleaned_sorted['time2'].dt.strftime('%Y-%m-%dT%H:%M')

        # Function to generate SAC filenames based on time
    def generate_sac_filename(time):
        return f"VICA.HHZ.{time}*.SAC" 

    for family in data_cleaned_sorted['family'].unique():
        print(f"Family {family}:")
        dir_name = f"Family_{family}"

        if not os.path.exists(dir_name):
            os.system(f"mkdir {dir_name}")
        else:
            print(f"Directory for Family {family} exists...")

        family_data = data_cleaned_sorted[data_cleaned_sorted['family'] == family]
        for index, row in family_data.iterrows():

            print(f"Event ID {row['event_id']} - time1: {row['time1']}, time2: {row['time2']}")
            f1 = generate_sac_filename(row['time1'])
            f2 = generate_sac_filename(row['time2'])
            
            for j, i in enumerate([f1, f2]):
                copy = f"cp {waveform_directory}/{i} {dir_name}/{i}"
                os.system(copy)

            print("Finished moving files to directories...")
            
                
    
def main():
    
    bandpass_filter= [(2, 8)]
    trace_len=[0.1, 6] # time before p, time after p 
    windowing =[2, 0.2]
    station_name='VICA'
    waveforms_path='sibilings'

    repeater_classification(waveforms_path=waveforms_path, station_name=station_name,
                            bandpass_filter=bandpass_filter, windowing=windowing, 
                            trace_len=trace_len)

    classification(input_correlation_file=f"Correlation_distance_results_{station_name}_{2}_{8}.dat", 
                   band_pass_filter=bandpass_filter[0], station_name=station_name, 
                   waveform_directory=waveforms_path)

if __name__ == '__main__':
    main()


