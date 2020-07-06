import argparse
from csv import reader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def calculate_accuracy_overall(actual_labels, predicted_labels):
    """
    Calculate accuracy percentage for all labels (classes).
    """
    correct = sum(1 for i in range(len(actual_labels)) if actual_labels[i] == predicted_labels[i])

    return correct / len(actual_labels) * 100.0


def calculate_accuracy_by_class(labels_indices, predictions):
    """
    Calculate accuracy percentage by label (class).
    Label indices is dictionary that contains labels as key
    and indices where label occurrence as value.
    """
    predictions = np.array(predictions)
    accuracies_by_class = {}

    for label, value in labels_indices.items():
        predictions_by_class = predictions[value]
        correct = np.argwhere(predictions_by_class == label).size
        accuracies_by_class[label] = (correct / len(value)) * 100.0

    return accuracies_by_class


def calculate_euclidean_distances(train_images, test_images):
    """
    Calculate euclidean distance for every image vector in test set
    with image vector in train set.
    Result is [21000, 21000] matrix.
    """
    test_images_vectors = test_images.reshape(test_images.shape[0], 28 * 28)
    train_images_vectors = train_images.reshape(train_images.shape[0], 28 * 28)

    print('calculating distances...')
    distances = cdist(test_images_vectors, train_images_vectors, metric="euclidean")
    print('finished calculating distances')
    return distances


def find_nearest_neighbors(distances, train_labels, knn):
    """
    Predict labels by assigning it most common class among
    it's closest k samples from the training set
    """
    predictions = []
    for distance in distances:
        sort_idx = distance.argsort()[:knn]
        first_k_neighbors = train_labels[sort_idx]
        number_of_occurrences = np.bincount(first_k_neighbors)
        predictions.append(np.argmax(number_of_occurrences))

    return predictions


def display_numbers_by_class(labels, images):
    """
    Display 10 random numbers (images) for each class (label).
    """
    fig, axes = plt.subplots(10, 10)
    fig.suptitle('Random images by class')

    for label in range(10):
        idxs = np.squeeze(np.argwhere(labels == label))
        random_mask = np.random.choice(idxs, size=10, replace=False)

        for idx, image in enumerate(images[random_mask]):
            plt_img = axes[label, idx]
            plt_img.imshow(image, cmap='gray')
            plt_img.set_title(label, rotation='horizontal', x=-0.2, y=0.2)
            plt_img.tick_params(axis='both', which='both', length=0)
            plt.setp(plt_img.get_xticklabels(), visible=False)
            plt.setp(plt_img.get_yticklabels(), visible=False)

    plt.show()


def display_accuracies(accuracy_by_class):
    """
    Display accuracies by class
    """
    labels = accuracy_by_class.keys()
    accuracies = accuracy_by_class.values()

    x_pos = np.arange(len(labels))

    plt.bar(x_pos, accuracies)
    plt.xlabel('Labels')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by class')
    plt.xticks(x_pos, labels)

    plt.show()


def __get_indices_by_label(labels):
    """
    Get all indices grouped by labels (classes).
    Method returns dictionary where key is label(class) and value are list of all indices for given label.
    {0: [43, 1245, 12, ...], 1:[4, 942, 24443], ...}
    """
    classes = np.unique(labels)

    return {label: np.squeeze(np.argwhere(labels == label)) for label in classes}


def shuffle_and_split_data(labels, images, percentage_decimal=0.5):
    """
    Split the labels and images into two subsets, train and test,
    with equal number of classes present in both subsets. Split is random.
    For example, if there is 2000 images of digit 5 in original dataset,
    train and test subsets should both contain 1000 radnom images.
    Dataset are split on half by default but you can set percentage ratio
    if you want to split it in different way.
    """
    indices_by_label = __get_indices_by_label(labels)

    train_idxs = np.array([], dtype=int)
    test_idxs = np.array([], dtype=int)

    for value in indices_by_label.values():
        split_index = int(round(percentage_decimal * len(value)))
        np.random.shuffle(value)
        train_idxs = np.append(train_idxs, value[:split_index])
        test_idxs = np.append(test_idxs, value[split_index:])

    train_labels, test_labels = labels[train_idxs], labels[test_idxs]
    train_images, test_images = images[train_idxs], images[test_idxs]

    return train_labels, train_images, test_labels, test_images


def read_and_transform_data(file_path):
    """
    Read csv file and split dataset to labels and images.
    Stores the labels into 1-dimensional numpy array and images into 3-dimensional numpy array.
    """

    # 3.55 s ± 52.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # dataset = load_csv_file(file_path)

    # 1.35 s ± 21.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    print('Reading dataset file...')
    dataset = pd.read_csv(file_path)
    print('Reading dataset file done')

    labels = dataset['label'].values
    images = dataset.drop(labels=["label"], axis=1).values.reshape(labels.size, 28, 28)

    return labels, images


def load_csv_file(file_path):
    """
    Read and load csv file in memory.
    """
    dataset = []
    with open(file_path, 'r') as file:
        csv_reader = reader(file, delimiter=',')
        next(csv_reader, None)  # skip the header
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    return np.array(dataset, dtype=int)


# Define a main() function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("knn", help="number of k nearest neighbors", type=int)
    parser.add_argument("path", help="path to dataset file")
    args = parser.parse_args()

    knn = args.knn
    file_path = args.path

    # 1.) Read the CSV file with images and labels.
    labels, images = read_and_transform_data(file_path)

    # 2.) Split the labels and images into train and test set
    train_labels, train_images, test_labels, test_images = shuffle_and_split_data(labels, images)

    # 3.) Randomly sample 10 images from the train and test set
    display_numbers_by_class(train_labels, train_images)
    display_numbers_by_class(test_labels, test_images)

    # 4.) Compare every sample in the test set to every sample in the training by calculating Euclidean (L2) distance
    distances = calculate_euclidean_distances(train_images, test_images)
    predictions = find_nearest_neighbors(distances, train_labels, knn)

    # 5.) Calculate and display per-class accuracy metric
    indices = __get_indices_by_label(test_labels)
    accuracies_by_class = calculate_accuracy_by_class(indices, predictions)
    for label, accuracy in accuracies_by_class.items():
        print(f'Accuracy for class {label} is equal: {accuracy}')

    display_accuracies(accuracies_by_class)

    # 6.) Calculate overall test-set accuracy
    percent = calculate_accuracy_overall(test_labels, predictions)
    print(f'Overall accuracy percentage: {percent}')


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
