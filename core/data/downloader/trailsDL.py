"""Prepare Trails dataset"""
import os
import zipfile

import shutil
import errno
import argparse
import requests

import random
import shutil
#pip install requests

_TARGET_DIR = os.path.expanduser('~/.torch/datasets/trail_dataset') #Directory to store data
DRIVE_ZIP_NAME = 'AER1515_Course_Project_Complete_Dataset' #"Name of zip file"
_FILE_ID= "1U5cXlpS7bipVtRu_Ea0g8Bs-FN7WPsA8" #Use Id for zip file on drive, ensure "Anyone with link can access"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                
def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize Trails dataset.',
        epilog='Example: python trailsDL.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args                

def separate_files_to_train_test_val():
    image_source_folder = os.path.expanduser('~/.torch/datasets/trail_dataset/Complete_Dataset/images')
    mask_source_folder = os.path.expanduser('~/.torch/datasets/trail_dataset/Complete_Dataset/masks')
    
    train_destination_image_folder = os.path.expanduser('../datasets/trail_dataset/Training/Images')
    train_destination_mask_folder = os.path.expanduser('../datasets/trail_dataset/Training/Masks')
    
    test_destination_image_folder = os.path.expanduser('../datasets/trail_dataset/Testing/Images')
    test_destination_mask_folder = os.path.expanduser('../datasets/trail_dataset/Testing/Masks')
    
    val_destination_image_folder =  os.path.expanduser('../datasets/trail_dataset/Validating/Images')
    val_destination_mask_folder =  os.path.expanduser('../datasets/trail_dataset/Validating/Masks')
    
    all_files = os.listdir(image_source_folder)
    
    files_to_move_to_training = int(0.6 * len(all_files))
    files_to_move_to_testing = int(0.2 * len(all_files))
    files_to_move_to_validating = int(0.2 * len(all_files))
    
    random.shuffle(all_files)
    
    for folder in [train_destination_image_folder, train_destination_mask_folder, test_destination_image_folder, test_destination_mask_folder, val_destination_image_folder, val_destination_mask_folder]:
        os.makedirs(folder, exist_ok = False)
        print('Destination Folders Created')
    
    for i, file_name in enumerate(all_files):
        try: 
            image_source_file = os.path.join(image_source_folder, file_name)
            mask_source_file = os.path.join(mask_source_folder, file_name[:-3] + 'png')
            if i < files_to_move_to_training:
                image_destination_file = os.path.join(train_destination_image_folder, file_name)
                mask_destination_file = os.path.join(train_destination_mask_folder, file_name[:-3] + 'png')
            elif i < files_to_move_to_training + files_to_move_to_testing:
                image_destination_file = os.path.join(test_destination_image_folder, file_name)
                mask_destination_file = os.path.join(test_destination_mask_folder, file_name[:-3] + 'png')
            else:
                image_destination_file = os.path.join(val_destination_image_folder, file_name)
                mask_destination_file = os.path.join(val_destination_mask_folder, file_name[:-3] + 'png')
            shutil.move(image_source_file, image_destination_file)
            shutil.move(mask_source_file, mask_destination_file)
        except FileNotFoundError:
            print(f"File not found: {mask_source_file}")

if __name__ == '__main__':
    args = parse_args()
    if args.download_dir is not None:
        _TARGET_DIR = args.download_dir
    try:
        os.makedirs(_TARGET_DIR, exist_ok = False)
        print("Trails dataset directory made")
        
        empty_zip_data = b'PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        zip_path = _TARGET_DIR + '/trails_temp.zip'
        with open(zip_path, 'wb') as zip:
            zip.write(empty_zip_data)
        
        print(f"Download {_FILE_ID} to {zip_path}")
        download_file_from_google_drive(_FILE_ID, zip_path)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(_TARGET_DIR)
            print("Files Unzipped")
            source_dir = _TARGET_DIR + '/' + DRIVE_ZIP_NAME
            target_dir = _TARGET_DIR
            file_names = os.listdir(source_dir)
        
            for file_name in file_names:
                shutil.move(os.path.join(source_dir, file_name), target_dir)
            os.rmdir(source_dir)
            os.remove(zip_path)
            
        except zipfile.BadZipFile:
            print('Not a zip file or a corrupted zip file')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # raises the error again
        print(f'Data is already available locally\nIf updated data is required, try deleting directory: \"{_TARGET_DIR}\", and try again')
    
    separate_files_to_train_test_val()
    

