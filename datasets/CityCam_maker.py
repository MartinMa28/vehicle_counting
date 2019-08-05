import numpy as np
import os
import time
import datetime
from tqdm import tqdm
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
sys.path.append(os.getcwd())
print(os.getcwd())
from vehicle_counting.density_map.k_nearest_gaussian_kernel import density_map_generator


# Global variables
citycam_dir = 'vehicle_counting/CityCam/'
test_train_sep_dir = os.path.join(citycam_dir, 'train_test_separation')
downtown_train_path = os.path.join(test_train_sep_dir, 'Downtown_Train.txt')
downtown_test_path = os.path.join(test_train_sep_dir, 'Downtown_Test.txt')
pathway_train_path = os.path.join(test_train_sep_dir, 'Parkway_Train.txt')
pathway_test_path = os.path.join(test_train_sep_dir, 'Parkway_Test.txt')
# Global variables


def _parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    height = int(root.find('height').text)
    width = int(root.find('width').text)
    assert height == 240
    assert width == 352
    
    vehicles = root.findall('vehicle')
    center_points = []

    for vehicle in vehicles:
        bnd_box = vehicle.find('bndbox')
        x_max = int(bnd_box.find('xmax').text)
        x_min = int(bnd_box.find('xmin').text)
        y_max = int(bnd_box.find('ymax').text)
        y_min = int(bnd_box.find('ymin').text)

        if x_min < 0:
            x_min = 0
        if x_max > width:
            x_max = width
        if y_min < 0:
            y_min = 0
        if y_max > height:
            y_max = height
        
        x_center = (x_max + x_min) // 2
        y_center = (y_max + y_min) // 2

        center_points.append((y_center, x_center))

    return (height, width), center_points

def time_stamp() -> str:
    ts = time.time()
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return time_stamp

def _make_density_map(frame_dir):
    # img_shape (height, width)
    frame_list = sorted(os.listdir(frame_dir))
    for frame in frame_list:
        frame_id, frame_format = frame.split('.')
        if frame_format == 'xml':
            if os.path.exists(os.path.join(frame_dir, frame_id + '_dm' + '.npy')):
                continue

            img_shape, points = _parse_xml(os.path.join(frame_dir, frame))
            try:
                density_map = density_map_generator(img_shape, points)
            except AssertionError as e:
                print(frame_dir)
                raise e
            else:
                np.save(os.path.join(frame_dir, frame_id + '_dm'), density_map)


def make_density_map(target_dir):
    print('Processing: ' + time_stamp() + '- ' + target_dir)
    camera_dir = os.path.join(citycam_dir, target_dir.split('-')[0])
    frame_dir = os.path.join(camera_dir, target_dir)
    _make_density_map(frame_dir)


if __name__ == "__main__":
    train_list = []
    with open(downtown_train_path) as f:
        train_list.extend(f.readlines()[:20])
    
    with open(pathway_train_path) as f:
        train_list.extend(f.readlines()[:10])

    test_list = []
    with open(downtown_test_path) as f:
        test_list.extend(f.readlines()[:10])

    with open(pathway_test_path) as f:
        test_list.extend(f.readlines()[:5])

    train_list = [sample.strip() for sample in train_list]
    test_list = [sample.strip() for sample in test_list]

    print('Computing training set density maps...')

    with ProcessPoolExecutor(max_workers=None) as executor:
        results = executor.map(make_density_map, train_list)
    
    tuple(results)

    print('Computing test set density maps...')
    
    with ProcessPoolExecutor(max_workers=None) as executor:
        results = executor.map(make_density_map, test_list)
    
    tuple(results)