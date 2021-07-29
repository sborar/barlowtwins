import glob
import numpy as np
from PIL import Image
from os.path import join

# use the meta name to put things in a certain folder
base = "/azure-ml/mvinterns/tcia-headneck/"

image_dataname = "image*"

raw_data_folder = "train_dataset"

# get all files with pattern
image_files = glob.glob(join(base, image_dataname))

# for each file
for image_file in image_files:
    try:
        pid_scanid = image_file.split('/')[-1][6:-4]
        #
        print(pid_scanid)
        pid = pid_scanid.split('.')[0]
        scanid = pid_scanid.split('-')[-1].split('_')[0]
        image = np.load(image_file, allow_pickle=True)["arr_0"]

        x = image.shape[0]

        for i in range(x):
            output_image_fname = join(raw_data_folder, 'img/1', pid_scanid + "_" + str(i) + ".png")
            Image.fromarray(np.uint8(image[i, :, :])).convert('RGB').save(output_image_fname)

    except Exception as e:
        print(e)
        continue

    print()
