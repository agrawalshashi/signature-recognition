import numpy as np


def extract_features(processed):
    """
    Extract 14 statistical features from one signature.
    7 mean features + 7 standard deviation features.
    """

    mean_features = processed.mean(axis=0).values
    std_features = processed.std(axis=0).values

    features = np.concatenate([
        mean_features,
        std_features
    ])

    return features.astype(np.float32)