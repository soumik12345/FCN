import re, cv2, os
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class KITTIDataInterface:

    def __init__(self, train_location, batch_size):
        self.train_location = train_location
        self.batch_size = batch_size
        self.read_data()
        self.get_generators()
    
    @staticmethod
    def get_data(data_dir, image_shape):
        image_paths = glob(os.path.join(data_dir, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_dir, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        images = []
        gt_images = []
        for image_file in tqdm(image_paths):
            gt_image_file = label_paths[os.path.basename(image_file)]

            image = resize(cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB), image_shape)
            gt_image = resize(cv2.imread(gt_image_file), image_shape)

            images.append(image)
            gt_images.append(gt_image[:, :, 0])

        return np.array(images), np.expand_dims(np.array(gt_images), axis = 3)
    
    def read_data(self):
        x, y = KITTIDataInterface.get_data(self.train_location, (160, 576))
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size = 0.1)
    
    def get_generators(self):
        SEED = 42
        
        self.image_data_generator = ImageDataGenerator(
            featurewise_center = True,
            featurewise_std_normalization = True,
            zoom_range = 0.1
        ).flow(
            x_train, x_train,
            self.batch_size, seed = SEED
        )
        
        self.mask_data_generator = ImageDataGenerator(
            featurewise_center = True,
            featurewise_std_normalization = True,
            zoom_range = 0.1
        ).flow(
            y_train, y_train,
            self.batch_size, seed = SEED
        )
    
    def visualize(self, figsize = (16, 16)):
        fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize = figsize)
        plt.setp(axes.flat, xticks = [], yticks = [])
        x_batch, _ = self.image_data_generator.next()
        y_batch, _ = self.mask_data_generator.next()
        c = 1
        for i, ax in enumerate(axes.flat):
            if i % 2 == 0:
                ax.imshow(x_batch[c])
                ax.set_xlabel('Image_' + str(c))
            else:
                ax.imshow(y_batch[c].reshape(160, 576), cmap = 'gray')
                ax.set_xlabel('Mask_' + str(c))
                c += 1
        plt.show()