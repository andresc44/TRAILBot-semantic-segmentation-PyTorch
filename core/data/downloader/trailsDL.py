"""Prepare Trails dataset"""
import os
import zipfile
import shutil
import errno
import argparse
import gdown
import random

# For Full Dataset
_TARGET_DIR = os.path.expanduser('../datasets/trail_dataset') #Directory to store data
_DRIVE_ZIP_NAME = 'Trailbot_full_dataset_with_masks' #"Name of zip file"
_FILE_ID= "1uokpokqeVa18dmjnC3LohmFyo38WtGAZ" #Use Id for zip file on drive, ensure "Anyone with link can access"
                
def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize Trails dataset.',
        epilog='Example: python trailsDL.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--zip-name', default=_DRIVE_ZIP_NAME, help='Name of zip file')
    parser.add_argument('--drive-zip-id', default=_FILE_ID, help='ID of the drive zip file')
    parser.add_argument('--eval', type=bool, default=False, help='True if all data in folder is to be evaluated')
    args = parser.parse_args()
    return args 

def save_with_gdown(id, destination):
    url = 'https://drive.google.com/uc?id='+id
    gdown.download(url, destination, quiet=False)            

def separate_files_to_train_test_val(args):
    image_source_folder = _TARGET_DIR + f'/images'
    mask_source_folder = _TARGET_DIR + f'/masks'
    
    train_destination_image_folder = os.path.expanduser(f'{_TARGET_DIR}/Training/Images')
    train_destination_mask_folder = os.path.expanduser(f'{_TARGET_DIR}/Training/Masks')
    
    test_destination_image_folder = os.path.expanduser(f'{_TARGET_DIR}/Testing/Images')
    test_destination_mask_folder = os.path.expanduser(f'{_TARGET_DIR}/Testing/Masks')
    
    val_destination_image_folder = os.path.expanduser(f'{_TARGET_DIR}/Validating/Images')
    val_destination_mask_folder = os.path.expanduser(f'{_TARGET_DIR}/Validating/Masks') 
    
    all_files = os.listdir(image_source_folder)
    print(train_destination_image_folder)
    files_to_move_to_training = int(0.6 * len(all_files))
    files_to_move_to_testing = int(0.2 * len(all_files))
    files_to_move_to_validating = int(0.2 * len(all_files))
    
    random.shuffle(all_files)

    for folder in [train_destination_image_folder, train_destination_mask_folder, test_destination_image_folder, test_destination_mask_folder, val_destination_image_folder, val_destination_mask_folder]:
        os.makedirs(folder, exist_ok = False)
        print(f'Destination Folders Created: {folder}')
    if (args.eval == True):
        print('Transferring all image pairs to \'Testing\' directory')
    for i, file_name in enumerate(all_files):
        try:
            image_source_file = os.path.join(image_source_folder, file_name)
            mask_source_file = os.path.join(mask_source_folder, file_name[:-3] + 'png')
            if (args.eval == True):
                image_destination_file = os.path.join(test_destination_image_folder, file_name)
                mask_destination_file = os.path.join(test_destination_mask_folder, file_name[:-3] + 'png')
            else:
                if i < files_to_move_to_training:
                    image_destination_file = os.path.join(train_destination_image_folder, file_name)
                    mask_destination_file = os.path.join(train_destination_mask_folder, file_name[:-3] + 'png')
                elif i < files_to_move_to_training + files_to_move_to_testing:
                    image_destination_file = os.path.join(test_destination_image_folder, file_name)
                    mask_destination_file = os.path.join(test_destination_mask_folder, file_name[:-3] + 'png')
                else:
                    image_destination_file = os.path.join(val_destination_image_folder, file_name)
                    mask_destination_file = os.path.join(val_destination_mask_folder, file_name[:-3] + 'png')
            shutil.move(mask_source_file, mask_destination_file)
            shutil.move(image_source_file, image_destination_file)
        except FileNotFoundError:
            print(f"File not found: {mask_source_file}")
            continue
    print("Successfully finished")
    shutil.rmtree(_TARGET_DIR + '/images', ignore_errors=False, onerror=None)
    shutil.rmtree(_TARGET_DIR + '/masks', ignore_errors=False, onerror=None)

    
        
if __name__ == '__main__':
    args = parse_args()
    try:
        os.makedirs(_TARGET_DIR, exist_ok = False)
        print("Trails dataset directory made")        
        zip_path = _TARGET_DIR + '/trails_temp.zip'
        save_with_gdown(args.drive_zip_id, zip_path)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(_TARGET_DIR)
            print("Files Unzipped")
            source_dir = _TARGET_DIR + '/' + args.zip_name
            target_dir = _TARGET_DIR
            file_names = os.listdir(source_dir)
        
            for file_name in file_names:
                shutil.move(os.path.join(source_dir, file_name), target_dir)
            os.rmdir(source_dir)
            os.remove(zip_path)
            separate_files_to_train_test_val(args)
            
        except zipfile.BadZipFile:
            print('Not a zip file or a corrupted zip file')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # raises the error again
        print(f'Data is already available locally\nIf updated data is required, try deleting directory: \"{_TARGET_DIR}\", and try again')
    
    

