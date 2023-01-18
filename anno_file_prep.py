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
    except Exception as e:
      print(e)
      continue


# This function prepares files from a webAnno project to be used in our project.
# The final result is two folders of json files for training and testing, split
# along the given "split" argument
def clean_folder(address):
  import os
  address = os.path.join(os.getcwd(),address) 
  files = os.listdir(address)
  for f in files:
    file_address = os.path.join(address,f) 
    os.remove(file_address)

# def prepare_files(filename, split):
#   split_idx = int(split*len(os.listdir(filename)))
#   base_file_address = "/".join(filename.split("/")[:-1])
#   # For all files in the train split:
#   clean_folder(base_file_address+"/anno_train/")
#   for idx, outer_file in enumerate(os.listdir(filename)[:split_idx]):
#     add_to_dir(filename, idx, outer_file, "anno_train")
#   clean_folder(base_file_address+"/anno_test/")
#   # For all files in the annotation folder:
#   for idx, outer_file in enumerate(os.listdir(filename)[split_idx:]):
#     add_to_dir(filename, idx, outer_file, "anno_test")

import sklearn
def prepare_files(filename, split):
  train_data, test_data = sklearn.model_selection.train_test_split(os.listdir(filename),test_size=1-split,random_state=321)
  base_file_address = "/".join(filename.split("/")[:-1])
  # For all files in the train split:
  # For all files in the train split:
  clean_folder(base_file_address+"/anno_train/")
  for idx, outer_file in enumerate(train_data):
    add_to_dir(filename, idx, outer_file, "anno_train")
  # For all files in the annotation folder:
  clean_folder(base_file_address+"/anno_test/")
  for idx, outer_file in enumerate(test_data):
    add_to_dir(filename, idx, outer_file, "anno_test")