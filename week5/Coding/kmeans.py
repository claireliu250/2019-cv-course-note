import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
import cv2

def assignment(df, centroids, colsmap):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0])**2 + (df['y'] - centroids[i][1])**2
            )
        )
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x : int(x.lstrip('distance_from')))
    df['color'] = df['closest'].map(lambda x : colmap[x])

def update(df, centorids):
    pass

def main():
    pass



if __name__ == '__main__':
    main()
