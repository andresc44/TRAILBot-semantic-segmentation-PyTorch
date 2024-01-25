"""Prepare Trails dataset"""
import os
import zipfile
import shutil
import argparse
import gdown

# For Full Dataset
_TARGET_DIR = os.path.expanduser('~/.torch/models') #Directory to store data
_DRIVE_ZIP_NAME = 'psp_resnet50_pascal_voc_best_model' #"Name of zip file"
_FILE_ID= "1UXIjqNTTWO76Q6pbibAftmMCMQx8RlHv" #Use Id for zip file on drive, ensure "Anyone with link can access"

def parse_args():
    parser = argparse.ArgumentParser(
        description='Import Trails trained model',
        epilog='Example: python3 trails_trained_model_DL.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--zip-name', default=_DRIVE_ZIP_NAME, help='Name of zip file')
    parser.add_argument('--drive-zip-id', default=_FILE_ID, help='ID of the drive zip file')
    parser.add_argument('--eval', type=bool, default=False, help='True if all data in folder is to be evaluated')
    args = parser.parse_args()
    return args 

def save_with_gdown(id, destination):
    url = 'https://drive.google.com/uc?id='+id
    gdown.download(url, destination, quiet=False)  



if __name__ == '__main__':
    args = parse_args()
    os.makedirs(_TARGET_DIR, exist_ok = True)
    print("~/.torch/models directory made/exists")        
    zip_path = _TARGET_DIR + '/trails_temp_model.zip'
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
        
    except zipfile.BadZipFile:
        print('Not a zip file or a corrupted zip file')    