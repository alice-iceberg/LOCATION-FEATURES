import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import re
import seaborn as sns
from haversine import haversine, Unit
from collections import Counter
import math

# region constants
FILENAME = 'aliceblackwood123@gmail.com-LOCATION_GPS-06-25_17_43_16.csv'
NUM_CLUSTERS = 1
MIN_POINTS_PER_CLUSTER = 3
MAX_DISTANCE = 100  # in meters
LAT_LNG = []
TIMESTAMP = []
# endregion

# region reading the file
with open(FILENAME, "r") as f:
    for line in f:
        values = re.sub('"', '', line.split(",")[1])
        time, lat, lng, speed, accuracy, altitude = values[:-1].split(" ")
        LAT_LNG.append([lat, lng])
        TIMESTAMP.append(int(time))
LAT_LNG = np.array(LAT_LNG).astype('float64')
TIMESTAMP = np.array(TIMESTAMP)

print("Work: ", LAT_LNG[10])
print("Work: ", LAT_LNG[LAT_LNG.__len__() - 10])
print("Home: ", LAT_LNG[70])


# endregion
# region functions definitions
def outlier_cluster(obj, threshold):
    # count number of elements per cluster
    centroids = obj.cluster_centers_
    cnt = Counter(obj.labels_)
    for i, centroid in enumerate(centroids):
        if cnt[i] < threshold:
            return i  # returns an outlier cluster label
    return -1  # returns -1 if no outliers detected


def remove_outlier_cluster(X, outlier_indices):
    for i, item in enumerate(outlier_indices):
        X = np.delete(X, item, axis=0)
        print("Element ", item, "deleted from X")
    return X


def outlier_cluster_data_point_indices(outlier_cluster):
    indices = np.where(y_label == outlier_cluster)[0]
    return indices


# endregion

# region LOCATION CLUSTERS

while True:
    temp = NUM_CLUSTERS
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', random_state=42)
    y_label = kmeans.fit_predict(LAT_LNG)

    # squared distance to cluster center
    X_dist = kmeans.transform(LAT_LNG) ** 2

    max_indices = []
    for label in np.unique(kmeans.labels_):
        X_label_indices = np.where(y_label == label)[0]
        max_label_idx = X_label_indices[np.argmax(X_dist[y_label == label].sum(axis=1))]
        max_indices.append(max_label_idx)

    farthest_points = (LAT_LNG[max_indices])
    centroids = kmeans.cluster_centers_

    # region PLOT
    sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    marker='+',
                    color='black',
                    s=200)
    sns.scatterplot(LAT_LNG[:, 0], LAT_LNG[:, 1], hue=y_label,
                    palette=sns.color_palette("Set1", n_colors=NUM_CLUSTERS))
    # highlight the furthest point in black

    sns.scatterplot(LAT_LNG[max_indices, 0], LAT_LNG[max_indices, 1], color='black')
    plt.show()
    # endregion

    distances = []
    for i, centroid in enumerate(centroids):
        distances.append(haversine(farthest_points[i], centroid, unit=Unit.METERS))
        if distances[i] > MAX_DISTANCE:  # generate new cluster if distance is more than a threshold
            temp = NUM_CLUSTERS + 1

    if temp == NUM_CLUSTERS:
        outlier_label = outlier_cluster(kmeans, MIN_POINTS_PER_CLUSTER)
        if outlier_label == -1:  # no outliers detected
            print("Success! No outliers detected")
            break
        else:
            print("Outlier cluster:", outlier_label)
            indices = outlier_cluster_data_point_indices(outlier_label)
            LAT_LNG = remove_outlier_cluster(LAT_LNG, indices)
    elif temp > NUM_CLUSTERS:
        NUM_CLUSTERS = temp
    else:
        print("Exception occurred: temp < num_clusters")

print("Number of clusters: ", NUM_CLUSTERS)

# endregion

# region ENTROPY
N = NUM_CLUSTERS
max_timestamp = TIMESTAMP[0]
min_timestamp = TIMESTAMP[0]
time_duration = 0
TIME_DURATION_PER_CLUSTER = []  # starts with last cluster

# region time duration per cluster
print(kmeans.labels_)

current_cluster = NUM_CLUSTERS - 1
elements_per_cluster = Counter(kmeans.labels_)

while True:
    print("CURRENT CLUSTER: ", current_cluster)
    print("NUMBER OF ELEMENTS: ", elements_per_cluster[current_cluster])
    elems = np.where(y_label == current_cluster)[0]

    for j, value in enumerate(elems):
        if j == 0:
            min_timestamp = TIMESTAMP[value]
            print("TIMESTAMP MIN", min_timestamp)
        elif (j == elements_per_cluster[current_cluster] - 1) and (
                value - elems[j - 1] > 1):  # if element is one and the last
            break
        elif (value - elems[j - 1] > 1) and (elems[j + 1] - value > 1):  # if element is one
            continue
        elif j == elements_per_cluster[current_cluster] - 1:  # last element of the cluster reached
            max_timestamp = TIMESTAMP[value]
            time_duration = time_duration + (max_timestamp - min_timestamp)
        elif j > 0 and abs(elems[j - 1] - value) > 1:
            max_timestamp = TIMESTAMP[elems[j - 1]]

            if abs(max_timestamp - min_timestamp) != 0:
                time_duration = time_duration + (max_timestamp - min_timestamp)
                min_timestamp = TIMESTAMP[value]

    TIME_DURATION_PER_CLUSTER.append(time_duration)
    time_duration = 0
    current_cluster = current_cluster - 1  # move to the previous cluster

    if current_cluster == -1:  # no clusters remained
        break

print("TIME DURATION PER CLUSTER:", TIME_DURATION_PER_CLUSTER)

# endregion

# region entropy
percentage_per_cluster = []  # starting from the last
log_percentage_per_cluster = []
total_time = sum(TIME_DURATION_PER_CLUSTER)
ENTROPY = 0

for i in range(0, TIME_DURATION_PER_CLUSTER.__len__()):
    percentage_per_cluster.append(TIME_DURATION_PER_CLUSTER[i] / total_time)
    if percentage_per_cluster[i] != 0:
        log_percentage_per_cluster.append(np.math.log(percentage_per_cluster[i], 2))  # log2P
        ENTROPY += percentage_per_cluster[i] * log_percentage_per_cluster[i]
    else:
        N -= 1

ENTROPY = abs(ENTROPY)
print("ENTROPY:", ENTROPY)

# endregion

# region normalized entropy
log_num_clusters = math.log(N, 2)
NORMALIZED_ENTROPY = ENTROPY / log_num_clusters
print("NORMALIZED_ENTROPY:", NORMALIZED_ENTROPY)

# endregion

# endregion


