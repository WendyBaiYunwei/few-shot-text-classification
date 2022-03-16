import numpy as np
import pandas as pd
def get_filtered_diffs(mean, interval):
    filtered_diffs = []

    while len(filtered_diffs) < interval:
        diffs = np.random.normal(loc=mean, scale=3.5, size=int(interval * 2.5))
        # oldMin = np.amin(diffs)
        newDiffs = list(filter(lambda x : (x <= mean and x >= 1), diffs))
        # newDiffs = list(map(lambda x : int((x - oldMin)/(mean - oldMin) * 598), newDiffs))
        filtered_diffs.extend(newDiffs[:interval])

    # filtered_diffs = [mean for i in range(INTERVAL)]
    return filtered_diffs
list = get_filtered_diffs(30, 1000)
s = pd.Series(list)
print(s.describe())