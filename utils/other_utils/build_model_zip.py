import time
from zipfile import ZipFile
from datetime import timedelta
import os


def get_all_file_paths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def main(model_name='catboost'):
    file_paths = []
    file_paths += get_all_file_paths(os.path.join('models', model_name))

    for file_name in file_paths:
        print(file_name)

    with ZipFile(model_name + "_" + time.strftime('%d_%H_%M_%S') + '.zip', 'w') as zip:
        for file in file_paths:
            zip.write(file)

    print('All files zipped successfully!')


if __name__ == "__main__":
    main('linear_regression')
