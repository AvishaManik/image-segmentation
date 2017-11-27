import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

usage = 'main.py number_of_clusters /path/to/images'
MAX_ITERATIONS = 25

if len(sys.argv) < 3:
    print('not enough arguments\n')
    print(usage)
    sys.exit(1)
if len(sys.argv) > 3:
    print('too many arguments\n')
    print(usage)
    sys.exit(1)


def main(argv):
    number_of_clusters = int(argv[1])
    input_data_path = (argv[2])

    directory = 'clusteredImages/k' + str(number_of_clusters)

    if not os.path.exists(directory):
        os.makedirs(directory)

    images = os.listdir(input_data_path + '.')

    for img in images:
        image = cv2.imread(input_data_path + img)

        (h, w) = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        image = image.reshape((image.shape[0] * image.shape[1], 3))

        clt = KMeans(n_clusters=number_of_clusters)
        labels = clt.fit_predict(image)
        clustered_image = clt.cluster_centers_.astype('uint8')[labels]

        clustered_image = clustered_image.reshape((h, w, 3))

        clustered_image = cv2.cvtColor(clustered_image, cv2.COLOR_LAB2BGR)

        cv2.imwrite(directory + '/' + img, np.hstack([clustered_image]))

main(sys.argv)