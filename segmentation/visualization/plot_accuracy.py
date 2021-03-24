import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

plot = 1

sv_path = r'C\Segmentation_working_area\results\new_acc_metrics\prealigned_vol'
filepath = r'C:\Segmentation_working_area\results\new_acc_metrics\prealigned_vol\prealigned_volume_accuracy_results.pickle'

with open(filepath, 'rb') as file:
    results = pickle.load(file)

    keys = list(results.keys())
    values = list(results.values())

    if plot >= 1:

        # Scatterplot false negatives
        b1 = plt.figure()

        for i in range(len(keys)):
            plt.bar(keys[i], values[i]['fn'], color='b')

        plt.title('False negatives comparison')
        plt.ylabel('False negatives (abs)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()

        # False positives
        for i in range(len(keys)):
            plt.bar(keys[i], values[i]['fp'], color='r')

        plt.title('False positives comparison')
        plt.ylabel('False positives (abs)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()

        # True positives
        for i in range(len(keys)):
            plt.bar(keys[i], values[i]['tp'], color='k')

        plt.title('True positives comparison')
        plt.ylabel('True positives (abs)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()

        # Undersegmentation
        for i in range(len(keys)):
            plt.bar(keys[i], values[i]['us'], color='m')

        plt.title('Undersegmentation comparison')
        plt.ylabel('Undersegmentations (abs)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()

        # Oversegmentation
        for i in range(len(keys)):
            plt.bar(keys[i], values[i]['os'], color='g')

        plt.title('Oversegmentation comparison')
        plt.ylabel('Oversegmentations (abs)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()

    # TODO create a table (pandas dataframe) with all results
    # list of lists for pandas table

    df = pd.DataFrame.from_dict(results)
    df.index = ['False negative', 'False negative %', 'False Positives', 'False Positives %',
                'True Positives', 'True Positives %', 'Over-segmentations', 'Under-segmentations',
                'gt matched', 'algo matched', 'v1', 'v2', 'total_neurons']
    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    print(tabulate(df, headers='keys', tablefmt='psql'))
    print(f'Means of rows:\n{df.mean(axis=1)}')

    df.to_csv(r'C:\Segmentation_working_area\results\new_acc_metrics\prealigned_vol_accuracy_summary.csv')
    # save the table
    # with open(os.path.join(sv_path, 'dataframe_accuracy_results.pickle')) as p_file:
    #     pickle.dump(df, p_file)
