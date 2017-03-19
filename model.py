import csv
import cv2
import numpy as np

lines = []
with open('../P3_data/recording2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../P3_data/recording2' + filename
    image = cv2.imread(current_path)
    images.append(image)

    measurement = line[4]
	measurements.append(measurement)