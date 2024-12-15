import os
import requests
import shutil
import zipfile
import argparse

def download_file_from_google_drive(id, destination):
    URL = f"https://drive.google.com/uc?id={id}"
    response = requests.get(URL, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        print(f"Failed to download file with ID {id}. Status code: {response.status_code}")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
parser.add_argument('--dataset', type=str, default='SIDD', help='all or SIDD or DND')
parser.add_argument('--noise', type=str, required=True, help='real or gaussian')
args = parser.parse_args()

### Google drive IDs ######
SIDD_train = '1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw'      # SIDD Training Data
SIDD_val   = '1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ'      # SIDD Validation Data
SIDD_test  = '11vfqV-lqousZTuAit1Qkqghiv_taY0KZ'      # SIDD Testing Data
DND_test   = '1CYCDhaVxYYcXhSfEVDUwkvJDtGxeQ10G'      # DND Testing Data
WaterlooED = '1a2b3c4d5e6f7g8h9i0j'                  # WaterlooED Training Data
DIV2K      = '2b3c4d5e6f7g8h9i0j1a'                  # DIV2K Training Data
BSD400     = '3c4d5e6f7g8h9i0j1a2b'                  # BSD400 Training Data
Flickr2K   = '4d5e6f7g8h9i0j1a2b3c'                  # Flickr2K Training Data
gaussian_test = '5e6f7g8h9i0j1a2b3c4d'               # Gaussian Denoising Testing Data

# Define noise type
noise = args.noise

for data in args.data.split('-'):
    if noise == 'real':
        if data == 'train':
            print('SIDD Training Data!')
            os.makedirs(os.path.join('Datasets', 'Downloads'), exist_ok=True)
            download_file_from_google_drive(SIDD_train, 'Datasets/Downloads/train.zip')
            print('Extracting SIDD Data...')
            with zipfile.ZipFile('Datasets/Downloads/train.zip', 'r') as zip_ref:
                zip_ref.extractall('Datasets/Downloads')
            os.rename(os.path.join('Datasets', 'Downloads', 'train'), os.path.join('Datasets', 'Downloads', 'SIDD'))
            os.remove('Datasets/Downloads/train.zip')

            print('SIDD Validation Data!')
            download_file_from_google_drive(SIDD_val, 'Datasets/val.zip')
            print('Extracting SIDD Data...')
            with zipfile.ZipFile('Datasets/val.zip', 'r') as zip_ref:
                zip_ref.extractall('Datasets')
            os.remove('Datasets/val.zip')

        if data == 'test':
            if args.dataset == 'all' or args.dataset == 'SIDD':
                print('SIDD Testing Data!')
                download_file_from_google_drive(SIDD_test, 'Datasets/test.zip')
                print('Extracting SIDD Data...')
                with zipfile.ZipFile('Datasets/test.zip', 'r') as zip_ref:
                    zip_ref.extractall('Datasets')
                os.remove('Datasets/test.zip')

            if args.dataset == 'all' or args.dataset == 'DND':
                print('DND Testing Data!')
                download_file_from_google_drive(DND_test, 'Datasets/test.zip')
                print('Extracting DND data...')
                with zipfile.ZipFile('Datasets/test.zip', 'r') as zip_ref:
                    zip_ref.extractall('Datasets')
                os.remove('Datasets/test.zip')

    if noise == 'gaussian':
        if data == 'train':
            os.makedirs(os.path.join('Datasets', 'Downloads'), exist_ok=True)
            print('WaterlooED Training Data!')
            download_file_from_google_drive(WaterlooED, 'Datasets/Downloads/WaterlooED.zip')
            print('Extracting WaterlooED Data...')
            with zipfile.ZipFile('Datasets/Downloads/WaterlooED.zip', 'r') as zip_ref:
                zip_ref.extractall('Datasets/Downloads')
            os.remove('Datasets/Downloads/WaterlooED.zip')

            print('DIV2K Training Data!')
            download_file_from_google_drive(DIV2K, 'Datasets/Downloads/DIV2K.zip')
            print('Extracting DIV2K Data...')
            with zipfile.ZipFile('Datasets/Downloads/DIV2K.zip', 'r') as zip_ref:
                zip_ref.extractall('Datasets/Downloads')
            os.remove('Datasets/Downloads/DIV2K.zip')

            print('BSD400 Training Data!')
            download_file_from_google_drive(BSD400, 'Datasets/Downloads/BSD400.zip')
            print('Extracting BSD400 data...')
            with zipfile.ZipFile('Datasets/Downloads/BSD400.zip', 'r') as zip_ref:
                zip_ref.extractall('Datasets/Downloads')
            os.remove('Datasets/Downloads/BSD400.zip')

            print('Flickr2K Training Data!')
            download_file_from_google_drive(Flickr2K, 'Datasets/Downloads/Flickr2K.zip')
            print('Extracting Flickr2K data...')
            with zipfile.ZipFile('Datasets/Downloads/Flickr2K.zip', 'r') as zip_ref:
                zip_ref.extractall('Datasets/Downloads')
            os.remove('Datasets/Downloads/Flickr2K.zip')

        if data == 'test':
            print('Gaussian Denoising Testing Data!')
            download_file_from_google_drive(gaussian_test, 'Datasets/test.zip')
            print('Extracting Data...')
            with zipfile.ZipFile('Datasets/test.zip', 'r') as zip_ref:
                zip_ref.extractall('Datasets')
            os.remove('Datasets/test.zip')

print('Download completed successfully!')