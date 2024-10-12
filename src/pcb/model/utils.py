import pandas as pd
import random
import shutil
import os
import pathlib

def convert_to_yolo_labels(annotation_df: pd.DataFrame, classes: list) -> list:
    """
    Convert annotations to YOLO format
    :param annotation_df: DataFrame containing annotations
    :param classes: list of class names
    :return: list of YOLO labels
    """
    yolo_labels = []

    for _, annot in annotation_df.iterrows():
        filename = annot['filename']
        width, height = annot['width'], annot['height']
        class_name = annot['class']
        xmin, ymin, xmax, ymax = annot['xmin'], annot['ymin'], annot['xmax'], annot['ymax']

        # Convert bounding box coordinates to YOLO format
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        class_index = classes.index(class_name)

        # Append to YOLO labels list
        yolo_labels.append((filename, class_index, x_center, y_center, bbox_width, bbox_height))

    return yolo_labels


def read_yolo_labels_from_file(file_path: str | pathlib.Path) -> list:
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            values = [float(value) for value in values]
            labels.append(values)
    return labels



def split_images_and_labels(images_dir, labels, output_dir, train_split=0.95, val_split=0.05):
    # os.makedirs(output_dir, exist_ok=True)

    for folder in ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test']:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)


    # Group labels by image filename
    image_labels = {}
    for label in labels:
        filename, class_index, x_center, y_center, bbox_width, bbox_height = label
        if filename not in image_labels:
            image_labels[filename] = []
        image_labels[filename].append(label)

    # Shuffle the image filenames
    image_filenames = list(image_labels.keys())
    random.shuffle(image_filenames)

    # Split the dataset
    num_images = len(image_filenames)
    num_train = int(num_images * train_split)
    num_val = int(num_images * val_split)

    train_filenames = image_filenames[:num_train]
    val_filenames = image_filenames[num_train:num_train + num_val]
    test_filenames = image_filenames[num_train + num_val:]

    # Write train, val, test images and labels
    for dataset, filenames in [('train', train_filenames), ('val', val_filenames), ('test', test_filenames)]:
        for filename in filenames:
            labels = image_labels[filename]
            with open((output_dir / f'labels/{dataset}/{os.path.splitext(filename)[0]}.txt'), 'a') as label_file:
                for label in labels:
                    _, class_index, x_center, y_center, bbox_width, bbox_height = label
                    label_file.write(f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}\n")
            # Copy images to corresponding folders
            shutil.copy(src=images_dir / filename, dst=output_dir / f'images/{dataset}/{filename}')



def yolo_to_original_annot(image_name, yolo_labels, annot_df, classes):
    original_annot = []

    for yolo_label in yolo_labels:
        # Extract original width and height from annotation DataFrame
        original_size = annot_df.loc[annot_df['filename'] == image_name, ['width', 'height']].iloc[0]
        original_width, original_height = original_size['width'], original_size['height']

        # Extract YOLO label components
        class_index, x_center, y_center, bbox_width, bbox_height, confidence = yolo_label

        # Scale bounding box coordinates and dimensions to original size
        original_x_center = x_center * original_width
        original_y_center = y_center * original_height
        original_bbox_width = bbox_width * original_width
        original_bbox_height = bbox_height * original_height

        # Calculate original bounding box coordinates
        original_x_min = original_x_center - original_bbox_width / 2
        original_y_min = original_y_center - original_bbox_height / 2
        original_x_max = original_x_center + original_bbox_width / 2
        original_y_max = original_y_center + original_bbox_height / 2

        # Append original annotation to list
        original_annot.append({
            'filename': image_name,
            'width': int(original_width),
            'height': int(original_height),
            'class': classes[int(class_index)],
            'xmin': int(original_x_min),
            'ymin': int(original_y_min),
            'xmax': int(original_x_max),
            'ymax': int(original_y_max),
            'confidence': confidence
        })

    return pd.DataFrame(original_annot)

def yolo_to_original_annot2(image_name: str, yolo_labels: , annot_df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    # Extract original width and height from annotation DataFrame
    original_size = annot_df.loc[annot_df['filename'] == image_name, ['width', 'height']].iloc[0]
    original_width, original_height = original_size['width'], original_size['height']

    # Convert yolo_labels to a DataFrame for vectorized operations
    yolo_df = pd.DataFrame(yolo_labels, columns=['class_index', 'x_center', 'y_center', 'bbox_width', 'bbox_height', 'confidence'])

    # Scale bounding box coordinates and dimensions to original size
    yolo_df['original_x_center'] = yolo_df['x_center'] * original_width
    yolo_df['original_y_center'] = yolo_df['y_center'] * original_height
    yolo_df['original_bbox_width'] = yolo_df['bbox_width'] * original_width
    yolo_df['original_bbox_height'] = yolo_df['bbox_height'] * original_height

    # Calculate original bounding box coordinates
    yolo_df['original_x_min'] = yolo_df['original_x_center'] - yolo_df['original_bbox_width'] / 2
    yolo_df['original_y_min'] = yolo_df['original_y_center'] - yolo_df['original_bbox_height'] / 2
    yolo_df['original_x_max'] = yolo_df['original_x_center'] + yolo_df['original_bbox_width'] / 2
    yolo_df['original_y_max'] = yolo_df['original_y_center'] + yolo_df['original_bbox_height'] / 2

    # Map class indices to class names
    yolo_df['class'] = yolo_df['class_index'].apply(lambda x: classes[int(x)])

    # Create the final DataFrame
    original_annot = yolo_df[['class', 'original_x_min', 'original_y_min', 'original_x_max', 'original_y_max', 'confidence']].copy()
    original_annot['filename'] = image_name
    original_annot['width'] = int(original_width)
    original_annot['height'] = int(original_height)

    # Rename columns to match the expected output
    original_annot.rename(columns={
        'original_x_min': 'xmin',
        'original_y_min': 'ymin',
        'original_x_max': 'xmax',
        'original_y_max': 'ymax'
    }, inplace=True)

    return original_annot
