"""Prepare Trails dataset"""
import os
import zipfile

import shutil
import errno
import argparse
import requests
#pip install requests

_TARGET_DIR = os.path.expanduser('~/.torch/datasets/trail_dataset') #Directory to store data
DRIVE_ZIP_NAME = 'drive_trails_dataset' #"Name of zip file"
_FILE_ID= "1RnhSo0_4d_BmJrhlt2VNBwsrfAuVa9RX" #Use Id for zip file on drive, ensure "Anyone with link can access"

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
    
    

