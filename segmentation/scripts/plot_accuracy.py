import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

sv_path = r'C:\Segmentation_working_area\accuracy'

with open(r'C:\Segmentation_working_area\accuracy\accuracy_results.pickle', 'rb') as file:
    results = pickle.load(file)


    keys = list(results.keys())
    values = list(results.values())

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