import os
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm as default_tqdm
from sklearn.model_selection import train_test_split
from typing import Tuple

import paths
from feature_extract import read_feature_file

@dataclass
class InstrumentData:
    relevance: float
    num_responses: int

    def __repr__(self):
        relevance_str = f'{self.relevance}'.replace("\n", "")
        num_responses_str = f'{self.num_responses}'.replace("\n", "")

        return f'({relevance_str}, {num_responses_str})'


def aggregate_labels(labels_csv: str) -> Tuple[pd.DataFrame, list]:
    # Load the CSV file
    labels_csv = os.path.join(paths.DATA_DIR, "openmic", "openmic-2018-aggregated-labels.csv")

    df = pd.read_csv(labels_csv)

    # Extract and deduplicate the instruments
    distinct_instruments = df['instrument'].unique().tolist()

    # Sort the list alphabetically for easier reading
    distinct_instruments.sort()

    # Aggregate the data by taking the mean of relevance and sum of num_responses for each sample_key and instrument pair
    aggregated_df = df.groupby(['sample_key', 'instrument']).agg({'relevance': 'mean', 'num_responses': 'sum'}).reset_index()

    # Pivot the aggregated dataframe
    pivot_df = aggregated_df.pivot(index='sample_key', columns='instrument', values=['relevance', 'num_responses'])

    # Initialize the final dataframe with sample_keys
    final_df = pd.DataFrame(index=pivot_df.index)

    # Populate the dataframe with tuples of (relevance, num_responses) for each instrument
    for instrument in distinct_instruments:
        relevance_col = ('relevance', instrument)
        num_responses_col = ('num_responses', instrument)

        datas = list(zip(pivot_df[relevance_col], pivot_df[num_responses_col]))

        final_df[instrument] = [InstrumentData(relevance, num_responses) for relevance, num_responses in datas]


    not_found_files = 0

    for row in final_df.iterrows():
        sample_key = row[0]

        sample_key_first_3_digits = sample_key[:3]
        
        audio_path = os.path.join( paths.DATA_DIR,
                                   "features",
                                   sample_key_first_3_digits,
                                   f'{sample_key}.tfrecord' )

        # print(f"audio: {sample_key} -> {audio_path}")

        if os.path.exists(audio_path):
            final_df.loc[sample_key, 'audio_path'] = audio_path
        
        else:
            final_df.loc[sample_key, 'audio_path'] = None
            print("WARNING: Audio file not found for sample_key:", sample_key)
            not_found_files += 1
        
    
    print("Number of audio files not found:", not_found_files)

    # rename the "sample_key" col to "audio_id"
    final_df.index.name = 'audio_id'

    return final_df, distinct_instruments

def calculate_significance_score( aggregated_labels: pd.Series,
                                  audio_id: str,
                                  instruments: list,
                                  lambda_: float = 0.1 ) -> float:
    """
    Calculate the significance score for this particular audio

    Sum(relevance) + lambda * (number of instruments with non-zero relevance)

        * lambda is a hyperparameter that can be tuned
    
    TODO: Add hyperparameter for num_responses
    """

    base_score = 0

    for instrument in instruments:
        relevance = aggregated_labels.loc[audio_id, instrument].relevance

        # handle nan as 0
        if pd.isna(relevance):
            relevance = 0

        base_score += relevance


    num_non_zero_instruments = 0

    for instrument in instruments:
        relevance = aggregated_labels.loc[audio_id, instrument].relevance

        if (relevance > 0) and ( not pd.isna(relevance) ):
            num_non_zero_instruments += 1


    significance_score = base_score + lambda_*num_non_zero_instruments

    return significance_score

def calculate_binary_labels(aggregated_labels: pd.DataFrame, tqdm=default_tqdm, threshold=0.5, lambda_=0.1):

    pbar = tqdm(total=len(aggregated_labels), desc="Calculating classification labels", unit=" rows")


    instruments = aggregated_labels.columns[:-1].to_list()

    for row in aggregated_labels.iterrows():
        audio_id = row[0]

        # Calculate the significance score for this audio, before we mutate the labels
        significance_score = calculate_significance_score( aggregated_labels,
                                                           audio_id,
                                                           instruments,
                                                           lambda_=lambda_ )

        for instrument in instruments:
            confidence = aggregated_labels.loc[audio_id, instrument].relevance

            # classify as present if confidence > threshold
            if confidence > threshold:
                aggregated_labels.loc[audio_id, instrument] = True
            
            # classify as absent if confidence < threshold, or nan
            else:
                aggregated_labels.loc[audio_id, instrument] = False
        
        # Add the significance score as the last column
        aggregated_labels.loc[audio_id, 'significance_score'] = significance_score
    

        pbar.update(1)

    # sort by significance score
    aggregated_labels = aggregated_labels.sort_values(by='significance_score', ascending=False)

    # make sure the labels are boolean
    aggregated_labels[instruments] = aggregated_labels[instruments].astype(bool)
    
    pbar.close()

    return aggregated_labels


def load_features(aggregated_labels: pd.DataFrame, tqdm=default_tqdm):
    pbar = tqdm( total=len(aggregated_labels), desc="Loading features", unit=" rows" )

    aggregated_labels['features'] = pd.Series(dtype='object')

    for row in aggregated_labels.iterrows():
        audio_id = row[0]

        audio_path = aggregated_labels.loc[audio_id, 'audio_path']
        assert audio_path is not None
            
        aggregated_labels.at[audio_id, 'features'] = read_feature_file(audio_path)

        pbar.update(1)
    
    pbar.close()


def split_data(aggregated_labels: pd.DataFrame, split_ratio=0.8):
    # baseline with sklearn

    # shuffle the data
    aggregated_labels = aggregated_labels.sample(frac=1)

    # split the data
    train_df, test_df = train_test_split(aggregated_labels, test_size=1-split_ratio)

    return train_df, test_df
