import json
import imageio
import glob
import cv2
import sys
import numpy as np

def label_pruning(x, y, leave_n=1000):
    import collections
    counts = collections.Counter(y)
    list_of_tuples = []
    for img, label in zip(x, y):
        list_of_tuples.append((img, label))

    list_of_tuples = sorted(list_of_tuples, key=lambda x: counts[x[1]], reverse=False)
    print(list_of_tuples[:leave_n])
    return np.array(list_of_tuples[:leave_n][0]), np.array(list_of_tuples[:leave_n][1])

if __name__ == '__main__':
    data_path = '/home/yeongjoon/data/image_cluster/data/hanza_ocr/handwriting'
    src = cv2.imread('/home/yeongjoon/data/image_cluster/data/hanza_ocr/handwriting/00001698.png')

    dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    dst2 = cv2.resize(dst, dsize=(150, 150), interpolation=cv2.INTER_LINEAR)

    print(src.shape, src)
    print(dst.shape, dst)
    print(dst2.shape, dst2)

    syllable_to_idx = {}
    idx_to_label = {}
    id_to_label = {}

    with open(data_path + '/handwriting_data_info1.json', encoding='utf-8') as f:
        max_width = 0
        max_height = 0
        min_width = 10000
        min_height = 10000
        s = json.load(f)
        print(s.keys())
        count = 0
        for image, item in zip(s['images'], s['annotations']):
            if item['attributes']['type'] == '글자(음절)':
                if item['text'] not in syllable_to_idx:
                    syllable_to_idx[item['text']] = count
                count += 1
                id_to_label[image['id']] = syllable_to_idx[item['text']]
                max_width = max(image['width'], max_width)
                max_height = max(image['height'], max_height)
                min_width = min(image['width'], min_width)
                min_height = min(image['height'], min_height)

        idx_to_label = {v: k for k, v in syllable_to_idx.items()}

        print(max_width, max_height, min_width, min_height)
        print("annotation_num:", len(syllable_to_idx))
        print("Whole count:", count)

    images = glob.glob(data_path + '/*.png')
    count = 0
    print(len(images))
    x = list()
    y = list()

    for img in images:
        count += 1
        src = cv2.imread(img)
        dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.resize(dst, dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
        id = img.split('/')[-1].split('.')[0]   # id 부분만 떼어냄
        idx = id_to_label[id]   # id => 몇 번째 label인지로 mapping
        y.append(idx)
        x.append(dst)
    x = np.array(x)
    y = np.array(y, dtype=int)
    x = x.reshape([-1, 100, 100, 1]) / 255.0

    x, y = label_pruning(x, y, leave_n=10)

    print("Done!")
        
    #s = s.replace("\'", '\"')
    #data = json.loads(s)
    #print(len(data))
