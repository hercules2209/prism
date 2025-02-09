import os
import shutil
import argparse
import gdown

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
parser.add_argument('--dataset', type=str, default='SIDD', help='all or SIDD or DND')
parser.add_argument('--noise', type=str, required=True, help='real or gaussian')
args = parser.parse_args()

### Google drive IDs ######
SIDD_train = '1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw'      ## https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing
SIDD_val   = '1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ'      ## https://drive.google.com/file/d/1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ/view?usp=sharing
SIDD_test  = '11vfqV-lqousZTuAit1Qkqghiv_taY0KZ'      ## https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/view?usp=sharing
DND_test   = '1CYCDhaVxYYcXhSfEVDUwkvJDtGxeQ10G'      ## https://drive.google.com/file/d/1CYCDhaVxYYcXhSfEVDUwkvJDtGxeQ10G/view?usp=sharing

BSD400    = '1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N'       ## https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing
DIV2K     = '13wLWWXvFkuYYVZMMAYiMVdSA7iVEf2fM'       ## https://drive.google.com/file/d/13wLWWXvFkuYYVZMMAYiMVdSA7iVEf2fM/view?usp=sharing
Flickr2K  = '1J8xjFCrVzeYccD-LF08H7HiIsmi8l2Wn'       ## https://drive.google.com/file/d/1J8xjFCrVzeYccD-LF08H7HiIsmi8l2Wn/view?usp=sharing
WaterlooED = '19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr'      ## https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing
gaussian_test = '1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0'   ## https://drive.google.com/file/d/1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0/view?usp=sharing


noise = args.noise

for data in args.data.split('-'):
    if noise == 'real':
        if data == 'train':
            print('SIDD Training Data!')
            os.makedirs(os.path.join('Datasets', 'Downloads'), exist_ok=True)
            gdown.download(f"https://drive.google.com/uc?id={SIDD_train}", 'Datasets/Downloads/train.zip', quiet=False)
            print('Extracting SIDD Data...')
            shutil.unpack_archive('Datasets/Downloads/train.zip', 'Datasets/Downloads')
            os.rename(os.path.join('Datasets', 'Downloads', 'train'), os.path.join('Datasets', 'Downloads', 'SIDD'))
            os.remove('Datasets/Downloads/train.zip')

            print('SIDD Validation Data!')
            gdown.download(f"https://drive.google.com/uc?id={SIDD_val}", 'Datasets/Downloads/val.zip', quiet=False)
            print('Extracting SIDD Data...')
            shutil.unpack_archive('Datasets/Downloads/val.zip', 'Datasets/Downloads')
            os.rename(os.path.join('Datasets', 'Downloads', 'val'), os.path.join('Datasets', 'Downloads', 'SIDD_val'))
            os.remove('Datasets/Downloads/val.zip')

        if data == 'test':
            if args.dataset == 'all' or args.dataset == 'SIDD':
                print('SIDD Testing Data!')
                gdown.download(f"https://drive.google.com/uc?id={SIDD_test}", 'Datasets/test.zip', quiet=False)
                print('Extracting SIDD Data...')
                shutil.unpack_archive('Datasets/test.zip', 'Datasets')
                os.remove('Datasets/test.zip')

            if args.dataset == 'all' or args.dataset == 'DND':
                print('DND Testing Data!')
                os.system(f'gdrive download {DND_test} --path Datasets/')
                print('Extracting DND data...')
                shutil.unpack_archive('Datasets/test.zip', 'Datasets')
                os.remove('Datasets/test.zip')

    if noise == 'gaussian':
        if data == 'train':
            os.makedirs(os.path.join('Datasets', 'Downloads'), exist_ok=True)
            print('WaterlooED Training Data!')
            os.system(f'gdrive download {WaterlooED} --path Datasets/Downloads/')
            print('Extracting WaterlooED Data...')
            shutil.unpack_archive('Datasets/Downloads/WaterlooED.zip', 'Datasets/Downloads')
            os.remove('Datasets/Downloads/WaterlooED.zip')

            print('DIV2K Training Data!')
            os.system(f'gdrive download {DIV2K} --path Datasets/Downloads/')
            print('Extracting DIV2K Data...')
            shutil.unpack_archive('Datasets/Downloads/DIV2K.zip', 'Datasets/Downloads')
            os.remove('Datasets/Downloads/DIV2K.zip')
            

            print('BSD400 Training Data!')
            os.system(f'gdrive download {BSD400} --path Datasets/Downloads/')
            print('Extracting BSD400 data...')
            shutil.unpack_archive('Datasets/Downloads/BSD400.zip', 'Datasets/Downloads')
            os.remove('Datasets/Downloads/BSD400.zip')
            
            print('Flickr2K Training Data!')
            os.system(f'gdrive download {Flickr2K} --path Datasets/Downloads/')
            print('Extracting Flickr2K data...')
            shutil.unpack_archive('Datasets/Downloads/Flickr2K.zip', 'Datasets/Downloads')
            os.remove('Datasets/Downloads/Flickr2K.zip')

        if data == 'test':
            print('Gaussian Denoising Testing Data!')
            os.system(f'gdrive download {gaussian_test} --path Datasets/')
            print('Extracting Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')

print('Download completed successfully!')
