from PIL import Image
import numpy as np
import glob

def load_folder(label, filepath=""):
    img_list = []

    for filename in glob.glob(filepath + "*jpg"):
        im = Image.open(filename)
        im = im.resize( (400, 225), Image.ANTIALIAS)
        set = np.asarray(im, dtype="int32").flatten()

        img_list.append(set)

    arr = np.array(img_list)

    labels = np.empty( len(arr), dtype="int32" )
    labels.fill( label )

    list = [arr, labels]

    return list

def load_all():
    all_list = []

    set1 = load_folder(1, filepath="1/")
    set2 = load_folder(2, filepath="2/")
    set3 = load_folder(3, filepath="3/")

    all_list.append(np.vstack((set1[0], set2[0], set3[0])))
    all_list.append(np.hstack((set1[1], set2[1], set3[1])))

    training_inputs = [np.reshape(x, (270000, 1)) for x in all_list[0]]
    training_results = [vectorized_result(y) for y in all_list[1]]
    training_data = list(zip(training_inputs, training_results))

    return training_data

def load_folder_old(filepath=""):
    img_list = np.array( None )

    for filename in glob.glob(filepath + "*jpg"):
        im=Image.open(filename)
        np.vstack( (img_list, np.asarray(im, dtype="int32")) )

    return img_list
def load_all_images_old():
    list_1, list_2, list_3 = [], [], []

    list_1 = load_folder("1/")
    new_list1 = np.hstack(list_1,[1])
    list_2 = load_folder("2/")
    new_list2 = np.hstack(list_2,[2])
    list_3 = load_folder("3/")
    new_list3 = np.hstack(list_3,[3])

    complete_list = np.stack(new_list1, new_list2, new_list3);

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((3, 1))
    e[j-1] = 1.0
    return e


#set1 = il.load_folder(1, filepath="1/")
