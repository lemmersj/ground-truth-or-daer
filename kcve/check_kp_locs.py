"""Displays keypoints on 2d images for manual checking."""
import csv
import os
import sys
import time
from random import randint
from PIL import Image, ImageDraw

# pylint: disable=C0103, W0702

directory = sys.argv[1]

directory_list = os.listdir(directory)

files = []
which_synset = 0
for synset in directory_list:
    which_synset += 1
    try:
        model_list = os.listdir(os.path.join(directory, synset))
    except NotADirectoryError:
        continue

    which_model = 0
    for model in model_list:
        which_model += 1
        print(which_synset, which_model, len(model_list))
        sample_list = os.listdir(os.path.join(directory, synset, model))
        for sample in sample_list:
            if sample[-4:] == ".png" or sample[-4:] == ".jpg":
                files.append((synset, model, sample[:-4]))

for i in range(10):
    random_file = files[randint(0, len(files))]
    try:
        image = Image.open(os.path.join(
            directory, random_file[0], random_file[1], random_file[2])+".png")
    except:
        image = Image.open(os.path.join(
            directory, random_file[0], random_file[1], random_file[2])+".jpg")

    in_csv = os.path.join(
        directory, random_file[0], random_file[1], random_file[2])+\
            "_keypoint2d.csv"
    draw = ImageDraw.Draw(image)
    with open(in_csv, 'r') as infile:
        csv_reader = csv.reader(infile)
        for row in csv_reader:
            draw.ellipse((
                float(row[1])-2.5, float(row[2])-2.5, float(row[1])+2.5,
                float(row[2])+2.5), fill=(255, 0, 0, 255))
    try:
        image_2 = Image.open(
            os.path.join("/z/dat/clickhere_cnn/syn_images/",
                         random_file[0], random_file[1], random_file[2])+
            ".png")
    except:
        image_2 = Image.open(os.path.join("/z/dat/clickhere_cnn/syn_images/",
                                          random_file[0], random_file[1],
                                          random_file[2])+".jpg")

    in_csv = os.path.join("/z/dat/clickhere_cnn/syn_images/",
                          random_file[0], random_file[1], random_file[2])+\
            "_keypoint2d.csv"
    draw = ImageDraw.Draw(image_2)
    with open(in_csv, 'r') as infile:
        csv_reader = csv.reader(infile)
        for row in csv_reader:
            draw.ellipse(
                (float(row[1])-2.5, float(row[2])-2.5, float(row[1])+2.5,
                 float(row[2])+2.5), fill=(255, 0, 0, 255))
    image.show()
    image_2.show()
    time.sleep(1)
