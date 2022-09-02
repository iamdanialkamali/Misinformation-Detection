# Used to prepare the files
import os
from zipfile import ZipFile


# Add the relevant json files to the proper output directory.
def add_to_dir(filename, index, outer_file, output_dir):

  # For all inner files in the outer file:
  for inner_file in os.listdir(filename + '/' + outer_file):

    # Unzip the inner file
    try:
      archive = ZipFile(filename + '/' + outer_file + '/' + inner_file, 'r')
      # For all files in the archive:
      for k in archive.filelist:
        if k.filename.endswith('.json'):
          # Extract the relevant json file into the final folder, renaming it
          # with the current index to prevent duplicate names.
          archive.extract(k.filename, 'data/'+output_dir)
          os.rename('data/'+output_dir+'/admin.json',
                    'data/'+output_dir+'/admin' + str(index) + '.json')
    except:
      continue


# This function prepares files from a webAnno project to be used in our project.
# The final result is two folders of json files for training and testing, split
# along the given "split" argument
def prepare_files(filename, split):
  split_idx = int(split*len(os.listdir(filename)))

  # For all files in the train split:
  for idx, outer_file in enumerate(os.listdir(filename)[:split_idx]):
    add_to_dir(filename, idx, outer_file, "anno_train")
  # For all files in the annotation folder:
  for idx, outer_file in enumerate(os.listdir(filename)[split_idx:]):
    add_to_dir(filename, idx, outer_file, "anno_test")