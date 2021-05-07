import numpy as np
import skimage.io
import pandas as pd
from scipy import ndimage as nd
import skimage.morphology as skm
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.colors
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
import cv2
import pickle
from skimage.measure import label, regionprops
import os
import re
import seaborn as sns
from skimage.color import label2rgb
from tifffile import imread, imwrite
import napari


def process_images(file_name):
    """
    This function processes opened images
    in order to produce a matrix for subsequent prediction.
    It applies threshold and filters to the input image.
    """
    global image
    global mask
    global thresh
    global segmented
    global df
    global original_img_data
    image = skimage.io.imread(file_name)
    mask = skimage.io.imread(path + "Mask_Final.tif")
    # Reshape original image to a single dimension array and add to a dataframe (1st column)
    image_reshaped = image.reshape(-1)
    df = pd.DataFrame()
    df['source_image'] = image_reshaped
    # Define sigmas for filers:gaussian,minimum and median. Median filters accepts only integers (int_sigmas)
    sigmas = [0.3, 0.7, 1, 3, 5, 10]
    int_sigmas = [i for i in sigmas if isinstance(i, (int))]
    # Apply gaussian,minimum and median filters, reshape to single dimension array and add to the dataframe
    for i in range(0, len(sigmas)):
        gaussian_img = nd.gaussian_filter(image, sigma=sigmas[i])
        gaussian_img_reshaped = gaussian_img.reshape(-1)
        col_name_gauss = "Gaussian_sigma_" + str(sigmas[i])
        # print(col_name_gauss)
        df[col_name_gauss] = gaussian_img_reshaped

        min_img = nd.minimum_filter(image, size=sigmas[i])
        min_img_reshaped = min_img.reshape(-1)
        col_name_min = "Minimum_sigma_" + str(sigmas[i])
        # print(col_name_min)
        df[col_name_min] = min_img_reshaped

    for i in range(0, len(int_sigmas)):
        median_img = nd.median_filter(image, size=int_sigmas[i])
        median_img_reshaped = median_img.reshape(-1)
        col_name_median = "Median_sigma_" + str(int_sigmas[i])
        # print(col_name_median)
        df[col_name_median] = median_img_reshaped
    ############################################################
    # Apply more filters-features, reshape and add to the dataframe
    # CANNY EDGE
    # edges = cv2.Canny(image, 100, 200)  # Image, min and max values
    # edges1 = edges.reshape(-1)
    # df['canny_edge'] = edges1
    # ROBERTS EDGE
    edge_roberts = roberts(image)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    # SOBEL
    edge_sobel = sobel(image)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    # SCHARR
    edge_scharr = scharr(image)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    # PREWITT
    edge_prewitt = prewitt(image)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    # Reshape the manually labeled mask image and add as Label column to he dataframe
    labeled_img = mask.reshape(-1)
    df['Labels'] = labeled_img
    # Remove the labels column and use the rest for prediction
    original_img_data = df.drop(labels=["Labels"], axis=1)
    df = df[df.Labels != 0]


def training_model(df):
    """
      This function is used to
      train the Random Forest Model
      for pixel classification.
      """
    global model_name
    global model
    # Define Y as the labels that you want to predict
    Y = df["Labels"].values
    # Y_Encoded=LabelEncoder().fit_transform(Y)
    # Define the independent variable X used for the prediction (the dataframe obtained before)
    X = df.drop(labels=["Labels"], axis=1)
    # Split dataset into train and test with 80% train and 20% test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)
    # Define the Random Forest Classifier model
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    # Train the model
    model.fit(X_train, y_train)
    # Retrive info on the feature contributions and print them in order of importance
    feature_list = list(X.columns)
    feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
    print(feature_imp)
    # Save the trained model and make it available for future application
    model_name = "Random_Forest_Intercalation_Model"
    pickle.dump(model, open(model_name, 'wb'))


###############################################################################################################
# Generate a random color scale from a modified Dark2 cmap
vals = np.linspace(0, 1, 256)
np.random.shuffle(vals)
mycmap = plt.cm.colors.ListedColormap(plt.cm.Dark2(vals))
# Define input and output paths and read the spurce images and masks
path = "C:/Users/Federico/Documents/Python_Image_Processing/Test/"
image_name = "test_image.tif"
file_name = path + image_name
# out_folder = path + "output/"
mask = skimage.io.imread(path + "Mask_Final.tif")
###############################################################################################################
# Now recall the process_images function to generate the image matrix for prediction
process_images(file_name)
###############################################################################################################
# Visualize gray values histogram from source image - very useful for thresholding
histogram, bin_edges = np.histogram(image, bins=255, range=(0, 255))
plt.figure()
plt.title("Source image Histogram")
plt.xlabel("Grayscale values")
plt.ylabel("Pixels")
plt.xlim([0.0, 255.0])
plt.xticks(np.arange(0, 255, step=5))
plt.plot(bin_edges[0:-1], histogram)
plt.show()
#
threshold_max = 20
erode_pixels = 2
thresh = image < threshold_max
thresh = skm.binary_erosion(thresh, selem=skm.disk(erode_pixels))
thresh = skm.binary_dilation(thresh, selem=skm.disk(erode_pixels))
thresh = ndimage.binary_fill_holes(thresh)
thresh = label2rgb(label(thresh), bg_label=0)

mask_ov = label2rgb(label(mask), image=image, bg_label=0)
# Visualize the source image and the thresholded binary mask, side-by-side
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.title("Source Image")
plt.imshow(image, cmap='gray')
f.add_subplot(1, 2, 2)
plt.title("Binary Mask")
plt.imshow(thresh, cmap='jet')
# Visualize the source image and the manually-labeled mask, side-by-side
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.title("Source Image")
plt.imshow(image, cmap='gray')
f.add_subplot(1, 2, 2)
plt.title("Labeled Mask")
plt.imshow(mask, cmap=mycmap)
#############################################
#############################################
# Here we train or model based on the open images and respective masks
training_model(df)
#############################################
#############################################
# Here we apply and test the trained model to get a prediction on the desired images
# prediction = model.predict(original_img_data)
model_name = "Random_Forest_Intercalation_Model"
loaded_model = pickle.load(open(model_name, 'rb'))
prediction = loaded_model.predict(original_img_data)
segmented = prediction.reshape(image.shape)
# Subtract 1 to the predicted image to obtain a binary image with values 0 or 1 only
segmented = segmented - 1
# Possible to add more binary operations to the obtained mask e.g. fill holes
segmented = skm.binary_erosion(segmented, selem=skm.disk(2))
segmented = ndimage.binary_fill_holes(segmented)
# Display side-by-side the original image, the thresholded one and the predicted with our model
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
f.add_subplot(1, 2, 2)
plt.title("Random Forest")
plt.imshow(segmented, cmap=mycmap)
###########################################################################################
# Testing the model through all images in a folder
img_files_names = []
img_basename = []
images_arr = []
segm_arr = []
thresh_arr = []
erode_arr = []
dilate_arr = []
fill_arr = []
img_lbl_thr = []
img_lbl_forest = []
img_lbl_thr_rgb = []
img_lbl_forest_rgb = []
count_thr = []
count_forest = []
well_fov = []
well = []
fov = []
time = []
match = []
model_name = "Random_Forest_Intercalation_Model"
loaded_model = pickle.load(open(model_name, 'rb'))
erode_pixels = 2
threshold = 20
#
path2 = "C:/Users/Federico/Documents/Python_Image_Processing/test_forest/"
for i in range(0, len(os.listdir(path2))):
    img_files_names.append(path2 + os.listdir(path2)[i])

    img_basename.append(os.path.basename(img_files_names[i]))
    well_fov.append(re.search(r"(?<=FITC_Well_).*?(?=Fov_).*?(?=_)", img_basename[i]).group(0))
    well.append(re.search(r"(?<=FITC_Well_).*?(?=_Fov)", img_basename[i]).group(0))
    well[i] = well[i].replace(' - ', '')
    fov_timepoint = re.search(r"(?<=_Fov_).*?(?=ms.tif)", img_basename[i]).group(0)
    fov.append(fov_timepoint.split('_')[0])
    match.append(well_fov[i] + "_" + fov[i])
    time_ms = int(fov_timepoint.split('_')[1])
    time_min = time_ms / 60000
    time.append(time_min)

    images_arr.append(skimage.io.imread(img_files_names[i]))
    process_images(img_files_names[i])
    prediction = loaded_model.predict(original_img_data)
    segmented = prediction.reshape(image.shape)
    # Subtract 1 to the predicted image to obtain a binary image with values 0 or 1 only
    segmented = segmented - 1
    # Possible to add more binary operations to the obtained mask e.g. fill holes
    segmented = skm.binary_erosion(segmented, selem=skm.disk(2))
    segmented = ndimage.binary_fill_holes(segmented)
    segm_arr.append(segmented)
    open_img = skimage.io.imread(img_files_names[i])
    # Threshold,Erode, dilate and fill holes from original image
    thresholded = open_img < threshold
    thresholded = skm.binary_erosion(thresholded, selem=skm.disk(erode_pixels))
    thresholded = skm.binary_dilation(thresholded, selem=skm.disk(erode_pixels))
    thresholded = ndimage.binary_fill_holes(thresholded)
    thresh_arr.append(thresholded)
    img_lbl_thr.append(label(thresh_arr[i]))
    # img_lbl_thr_rgb.append(label2rgb(img_lbl_thr[i], image=images_arr[i], bg_label=0))
    img_lbl_thr_rgb.append(label2rgb(img_lbl_thr[i], bg_label=0))
    img_lbl_forest.append(label(segm_arr[i]))
    img_lbl_forest_rgb.append(label2rgb(img_lbl_forest[i], image=images_arr[i], bg_label=0))
    imwrite(path2 + 'Well_' + well[i] + "_FOV_" + fov[i] + "-" + str(time[i]) + '_min_Thr_Segm.tif', img_lbl_thr_rgb[i])
    # imwrite(path2 + 'Well_' + well[i] + "_FOV_" + fov[i] +"_"+str(time[i])+'_min_RF_segm.tif', img_lbl_forest_rgb[i])
    regions_thr = regionprops(img_lbl_thr[i])
    regions_forest = regionprops(img_lbl_forest[i])
    count_thr.append(len(regions_thr))
    count_forest.append(len(regions_forest))

out_path = path2 + "out/thr/"
out_names = []
img_basename = []
well_fov = []
well = []
fov_timepoint = []
fov = []
match = []
for i in range(0, len(os.listdir(out_path))):
    out_names.append(out_path + os.listdir(out_path)[i])
    img_basename.append(os.path.basename(out_names[i]))
    well_fov.append(re.search(r"(?<=Well_).*?(?=FOV_).*?(?=_)", img_basename[i]).group(0))
    well.append(re.search(r"(?<=Well_).*?(?=_FOV)", img_basename[i]).group(0))
    well[i] = well[i].replace(' - ', '')
    fov_timepoint = re.search(r"(?<=_FOV_).*?(?=_min)", img_basename[i]).group(0)
    fov.append(fov_timepoint.split('-')[0])
    match.append(well_fov[i] + "_" + fov[i])

match_unique = np.unique(match)

stack = []
for i in range(0, len(match_unique)):
    stack_arr = []
    for j in range(0, len(out_names)):
        if match_unique[i] in out_names[j]:
            stack_arr.append((out_names[j]))

    stack.insert(i, stack_arr)

for j in range(0, len(stack)):
    image_stack = []
    for i in range(0, len(stack[j])):
        image_stack.append(skimage.io.imread(stack[j][i]))
        basename = os.path.basename(stack[j][i])
        well = (re.search(r"(?<=Well_).*?(?=_FOV)", basename).group(0))
        fov_timepoint = re.search(r"(?<=_FOV_).*?(?=_min)", basename).group(0)
        fov = (fov_timepoint.split('_')[0])

    imwrite(out_path + 'Well_' + well + "_FOV_" + fov + '_stack.tif', image_stack)

# Here we visualize and compare 3 different images: original images, thresholded binary and Random Forest pixel classification
# fig_col = 3
# fig_rows = 3
#
# f = plt.figure()
# f.add_subplot(fig_rows, fig_col, 1)
# plt.imshow(images_arr[0], cmap="gray")
# f.add_subplot(fig_rows, fig_col, 2)
# plt.title("Source Image")
# plt.imshow(images_arr[1], cmap="gray")
# f.add_subplot(fig_rows, fig_col, 3)
# plt.imshow(images_arr[2], cmap="gray")
#
# f.add_subplot(fig_rows, fig_col, 4)
# plt.imshow(thresh_arr[0], cmap=mycmap)
# f.add_subplot(fig_rows, fig_col, 5)
# plt.title("Thresholded")
# plt.imshow(thresh_arr[1], cmap=mycmap)
# f.add_subplot(fig_rows, fig_col, 6)
# plt.imshow(thresh_arr[2], cmap=mycmap)
#
# f.add_subplot(fig_rows, fig_col, 7)
# plt.imshow(segm_arr[0], cmap=mycmap)
# f.add_subplot(fig_rows, fig_col, 8)
# plt.title("Random Forest")
# plt.imshow(segm_arr[1], cmap=mycmap)
# f.add_subplot(fig_rows, fig_col, 9)
# plt.imshow(segm_arr[2], cmap=mycmap)

# Here we create a dataframe containing the well, FOV, timepoint and counted objects for analysis
data_create = {'Well': well,
               'Fov': fov,
               'Time': time,
               'Count_Thr': count_thr,
               'Count_RF': count_forest}
data = pd.DataFrame(data_create)
data = data.assign(Id=data["Well"] + "_" + data["Fov"])
# We read the experimental conditions from an external CSV, create the dictionary to pair wells and conditions, assign the conditions
conditions = pd.read_csv("C:/Users/Federico/Documents/Python_Image_Processing/Exp_Cond.csv")
cond_dict = dict(conditions.to_dict('split')['data'])
data['Condition'] = data['Well'].map(cond_dict)

# Finally we plot our results: events over time for thresholding and random forest segmentation
fig = plt.figure()
plot = sns.scatterplot(data=data, x='Time', y='Count_Thr', hue='Condition', palette="Set2")
sns.lineplot(data=data, x='Time', y='Count_Thr', hue='Condition', palette="Set2")
sns.scatterplot(data=data, x='Time', y='Count_RF', hue='Condition', palette="Set2")
sns.lineplot(data=data, x='Time', y='Count_RF', hue='Condition', palette="Set2")
plot.set_xlabel("Time (min)", fontsize=20)
plot.set_ylabel("Intercalation Events", fontsize=20)

# Here we want to check the results of the segmentation on our images
# First we define well,FOV and timepoint of the images to visualize
# well_vis = "C - 04"
# fov_vis = "03"
# time_vis = 70
# time_ms_vis = str(time_vis * 60000)
# # Then we check for the image name matching the defined parameters
# well_match = [s for s in img_basename if well_vis in s]
# fov_match = [s for s in well_match if fov_vis in s]
# matched = [s for s in fov_match if time_ms_vis in s]
# # And we find the index of the filename in the img_basename array defined before
# match_index = img_basename.index(matched[0])
# # Finally we plot side-by-side the original image, the thresholded binary and the random forest prediction
# f = plt.figure()
# f.add_subplot(1, 3, 1)
# plt.imshow(images_arr[match_index], cmap='gray')
# f.add_subplot(1, 3, 2)
# plt.imshow(img_lbl_thr_rgb[match_index])
# f.add_subplot(1, 3, 3)
# plt.imshow(img_lbl_forest_rgb[match_index])
# plt.show(block=True)

well_vis = "C04"
fov_vis = "03"
vis_files_names = []
vis_files_names_source = []
read_path = out_path + "stack/"
source_path = path2 + "stacks/"
for i in range(0, len(os.listdir(source_path))):
    vis_files_names_source.append(source_path + os.listdir(source_path)[i])
    well_match = [s for s in vis_files_names_source if well_vis in s]
    fov_match_original = [s for s in well_match if fov_vis in s]
for i in range(0, len(os.listdir(read_path))):
    vis_files_names.append(read_path + os.listdir(read_path)[i])
    well_match = [s for s in vis_files_names if well_vis in s]
    fov_match = [s for s in well_match if fov_vis in s]

source_image = imread(fov_match_original)
image_thr = imread(fov_match)
% gui
qt

viewer = napari.Viewer()
viewer.add_image(source_image);
viewer.add_image(image_thr);
