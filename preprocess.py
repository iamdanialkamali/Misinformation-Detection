from anno_file_prep import prepare_files
# Define the train/test split and prepare the annotation files
# This will create data/anno_train and data/anno_test
SPLIT = 0.8
if __name__ == "__main__":
    prepare_files('data/annotation', SPLIT)
