import os
import random
import csv

# Constant
reference_dir = '/home/leejiahe/copydetection/data/references/'
query_dev_dir = '/home/leejiahe/copydetection/data/dev_queries/'
ground_truth_path = '/home/leejiahe/copydetection/data/dev_ground_truth.csv'
val_path = '/home/leejiahe/copydetection/data/dev_validation_set.csv'

# Get all reference images file names
reference_images = [f for f in os.listdir(reference_dir)]

# Read ground truth data
val_data = []
with open(ground_truth_path) as csvfile:
    csvreader = csv.reader(csvfile, delimiter = ',')
    for row in csvreader: # Row return query_id, reference_id
        ref_image = f'{row[1]}.jpg'
        query_image = f'{row[0]}.jpg'
    
        rand_image = ref_image
        while(rand_image == ref_image):
            rand_image = random.choice(reference_images) # pick one reference image randomly
        
        val_data.append([ref_image, query_image, 1])
        val_data.append([rand_image, query_image, 0])
        
with open(val_path, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(val_data)