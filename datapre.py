import pandas as pd
import os
import cv2



full_path_to_ts_dataset = '/home/giwrgos/FullIJCNN2013'

# #Prohibitory cateogry
# p = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]

# # Danger category:
# d = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

# # Mandatory category:
# m = [33, 34, 35, 36, 37, 38, 39, 40]

# # Other category:
# o = [6, 12, 13, 14, 17, 32, 41, 42]


ann = pd.read_csv(full_path_to_ts_dataset + '/' + 'gt.txt',
                  names=['ImageID',
                         'XMin',
                         'YMin',
                         'XMax',
                         'YMax',
                         'ClassID'],
                  sep=';')

print(ann.head())

ann['CategoryID'] = ''
ann['center x'] = ''
ann['center y'] = ''
ann['width'] = ''
ann['height'] = ''

# To retain all 43 classes.

ann['CategoryID'] = ann['ClassID']

#To create 4 broader classes
# ann.loc[ann['ClassID'].isin(p), 'CategoryID'] = 0
# ann.loc[ann['ClassID'].isin(d), 'CategoryID'] = 1
# ann.loc[ann['ClassID'].isin(m), 'CategoryID'] = 2
# ann.loc[ann['ClassID'].isin(o), 'CategoryID'] = 3


ann['center x'] = (ann['XMax'] + ann['XMin']) / 2
ann['center y'] = (ann['YMax'] + ann['YMin']) / 2
ann['width'] = ann['XMax'] - ann['XMin']
ann['height'] = ann['YMax'] - ann['YMin']


r = ann.loc[:, ['ImageID',
                'CategoryID',
                'center x',
                'center y',
                'width',
                'height']].copy()


print(r.head())



os.chdir(full_path_to_ts_dataset)


for current_dir, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.ppm'):
            image_ppm = cv2.imread(f)
            h, w = image_ppm.shape[:2]
            image_name = f[:-4]

   
            sub_r = r.loc[r['ImageID'] == f].copy()

            sub_r['center x'] = sub_r['center x'] / w
            sub_r['center y'] = sub_r['center y'] / h
            sub_r['width'] = sub_r['width'] / w
            sub_r['height'] = sub_r['height'] / h

            resulted_frame = sub_r.loc[:, ['CategoryID',
                                           'center x',
                                           'center y',
                                           'width',
                                           'height']].copy()

            # if resulted_frame.isnull().values.all():
            #     continue

            path_to_save = full_path_to_ts_dataset + '/' + image_name + '.txt'

            resulted_frame.to_csv(path_to_save, header=False, index=False, sep=' ')

            path_to_save = full_path_to_ts_dataset + '/' + image_name + '.jpg'

            cv2.imwrite(path_to_save, image_ppm)
