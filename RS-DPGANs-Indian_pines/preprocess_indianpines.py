import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing
from sklearn.decomposition import PCA
from skimage.morphology import reconstruction
from skimage.morphology import erosion
from skimage.morphology import disk
from skimage import util
from sklearn.model_selection import train_test_split
#Load data
dataset = io.loadmat('indianpines_dataset.mat')
number_of_bands = int(dataset['number_of_bands'])
number_of_rows = int(dataset['number_of_rows'])
number_of_columns = int(dataset['number_of_columns'])
pixels = np.transpose(dataset['pixels'])

groundtruth = io.loadmat('indianpines_gt.mat')
gt = np.transpose(groundtruth['pixels'])

sc = StandardScaler()
pixels = sc.fit_transform(pixels)

# colors for each category in the dataset
indianpines_colors = np.array([[255, 255, 255],
                               [255, 254, 137], [3,  28,  241], [255, 89,    1], [5,   255, 133],
                               [255,   2, 251], [89,  1,  255], [3,   171, 255], [12,  255,   7],
                               [172, 175,  84], [160, 78, 158], [101, 173, 255], [60,   91, 112],
                               [104, 192,  63], [139, 69,  46], [119, 255, 172], [254, 255,   3]])

indianpines_colors = sklearn.preprocessing.minmax_scale(indianpines_colors, feature_range=(0, 1))
pixels_normalized = sklearn.preprocessing.minmax_scale(pixels, feature_range=(0, 1))

# build the RGB Image
gt_thematic_map = np.zeros(shape=(number_of_rows, number_of_columns, 3))
rgb_hyperspectral_image = np.zeros(shape=(number_of_rows, number_of_columns, 3))
cont = 0
for i in range(number_of_rows):
    for j in range(number_of_columns):
        rgb_hyperspectral_image[i, j, 0] = pixels_normalized[cont, 29]
        rgb_hyperspectral_image[i, j, 1] = pixels_normalized[cont, 42]
        rgb_hyperspectral_image[i, j, 2] = pixels_normalized[cont, 89]
        gt_thematic_map[i, j, :] = indianpines_colors[gt[cont, 0]]
        cont += 1

# names of the categories in the dataset
indianpines_class_names = ['background',
                           'alfalfa',           'corn-notill',               'corn-min',               'corn',
                           'grass/pasture',     'grass/trees',    'grass/pasture-mowed',      'hay-windrowed',
                           'oats',          'soybeans-notill',           'soybeans-min',      'soybean-clean',
                           'wheat',                   'woods', 'bldg-grass-tree-drives', 'stone-steel towers']

number_of_pc = 4
pca = PCA(n_components=number_of_pc)
pc = pca.fit_transform(pixels)
fig = plt.figure(figsize=(15, 15))
columns = number_of_pc
rows = 1
pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc))
for i in range(number_of_pc):
    pc_images[:, :, i] = np.reshape(pc[:, i], (number_of_rows, number_of_columns))

def opening_by_reconstruction(image, se):
    """
        Performs an Opening by Reconstruction.

        Parameters:
            image: 2D matrix.
            se: structuring element
        Returns:
            2D matrix of the reconstructed image.
    """
    eroded = erosion(image, se)
    reconstructed = reconstruction(eroded, image)
    return reconstructed


def closing_by_reconstruction(image, se):
    """
        Performs a Closing by Reconstruction.

        Parameters:
            image: 2D matrix.
            se: structuring element
        Returns:
            2D matrix of the reconstructed image.
    """
    obr = opening_by_reconstruction(image, se)

    obr_inverted = util.invert(obr)
    obr_inverted_eroded = erosion(obr_inverted, se)
    obr_inverted_eroded_rec = reconstruction(
        obr_inverted_eroded, obr_inverted)
    obr_inverted_eroded_rec_inverted = util.invert(obr_inverted_eroded_rec)
    return obr_inverted_eroded_rec_inverted

def build_morphological_profiles(image, se_size=4, se_size_increment=2, num_openings_closings=4):
    """
        Build the morphological profiles for a given image.

        Parameters:
            base_image: 2d matrix, it is the spectral information part of the MP.
            se_size: int, initial size of the structuring element (or kernel). Structuring Element used: disk
            se_size_increment: int, structuring element increment step
            num_openings_closings: int, number of openings and closings by reconstruction to perform.
        Returns: 
            emp: 3d matrix with both spectral (from the base_image) and spatial information         
    """
    x, y = image.shape

    cbr = np.zeros(shape=(x, y, num_openings_closings))
    obr = np.zeros(shape=(x, y, num_openings_closings))

    it = 0
    tam = se_size
    while it < num_openings_closings:
        se = disk(tam)
        temp = closing_by_reconstruction(image, se)
        cbr[:, :, it] = temp[:, :]
        temp = opening_by_reconstruction(image, se)
        obr[:, :, it] = temp[:, :]
        tam += se_size_increment
        it += 1

    mp = np.zeros(shape=(x, y, (num_openings_closings*2)+1))
    cont = num_openings_closings - 1
    for i in range(num_openings_closings):
        mp[:, :, i] = cbr[:, :, cont]
        cont = cont - 1

    mp[:, :, num_openings_closings] = image[:, :]

    cont = 0
    for i in range(num_openings_closings+1, num_openings_closings*2+1):
        mp[:, :, i] = obr[:, :, cont]
        cont += 1

    return mp


def build_emp(base_image, se_size=4, se_size_increment=2, num_openings_closings=4):
    """
        Build the extended morphological profiles for a given set of images.

        Parameters:
            base_image: 3d matrix, each 'channel' is considered for applying the morphological profile. It is the spectral information part of the EMP.
            se_size: int, initial size of the structuring element (or kernel). Structuring Element used: disk
            se_size_increment: int, structuring element increment step
            num_openings_closings: int, number of openings and closings by reconstruction to perform.
        Returns:
            emp: 3d matrix with both spectral (from the base_image) and spatial information
    """
    base_image_rows, base_image_columns, base_image_channels = base_image.shape
    se_size = se_size
    se_size_increment = se_size_increment
    num_openings_closings = num_openings_closings
    morphological_profile_size = (num_openings_closings * 2) + 1
    emp_size = morphological_profile_size * base_image_channels
    emp = np.zeros(
        shape=(base_image_rows, base_image_columns, emp_size))

    cont = 0
    for i in range(base_image_channels):
        # build MPs
        mp_temp = build_morphological_profiles(
            base_image[:, :, i], se_size, se_size_increment, num_openings_closings)

        aux = morphological_profile_size * (i+1)

        # build the EMP
        cont_aux = 0
        for k in range(cont, aux):
            emp[:, :, k] = mp_temp[:, :, cont_aux]
            cont_aux += 1

        cont = morphological_profile_size * (i+1)

    return emp

pc_images.shape
num_openings_closings = 4
morphological_profile_size = (num_openings_closings * 2) + 1
emp_image = build_emp(base_image=pc_images, num_openings_closings=num_openings_closings)

dim_x, dim_y, dim_z = emp_image.shape
dim = dim_x * dim_y

x = np.zeros(shape=(dim, dim_z))
y = gt

cont = 0
for i in range(dim_x):
    for j in range(dim_y):
        x[cont, :] = emp_image[i, j, :]
        cont += 1

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.75, random_state=0)

