import logging
import torch
from os import path as osp
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                         make_exp_dirs)
from basicsr.utils.options import dict2str

def main():
    try:
        # parse options, set distributed setting, set random seed
        opt = parse_options(is_train=False)
        
        # Check CUDA and set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Create directories and loggers
        os.makedirs(opt['path']['log'], exist_ok=True)
        log_file = osp.join(opt['path']['log'],
                          f"test_{opt['name']}_{get_time_str()}.log")
        logger = get_root_logger(
            logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(opt))

        # Memory management
        torch.cuda.empty_cache()

        # Create test dataloaders with error handling
        test_loaders = []
        for phase, dataset_opt in sorted(opt['datasets'].items()):
            try:
                test_set = create_dataset(dataset_opt)
                test_loader = create_dataloader(
                    test_set,
                    dataset_opt,
                    num_gpu=opt['num_gpu'],
                    dist=opt['dist'],
                    sampler=None,
                    seed=opt['manual_seed'])
                logger.info(
                    f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
                test_loaders.append(test_loader)
            except Exception as e:
                logger.error(f"Error creating dataloader for {dataset_opt['name']}: {e}")
                continue

        model = create_model(opt)

        # Test each dataset
        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            logger.info(f'Testing {test_set_name}...')
            try:
                # Clear memory before each test set
                torch.cuda.empty_cache()
                
                model.validation(
                    test_loader,
                    current_iter=opt['name'],
                    tb_logger=None,
                    save_img=opt['val']['save_img'],
                    rgb2bgr=opt['val'].get('rgb2bgr', True),
                    use_image=opt['val'].get('use_image', True))
            except Exception as e:
                logger.error(f"Error testing {test_set_name}: {e}")
                continue

    except Exception as e:
        logger.error(f"Fatal error in testing: {e}")
        raise

    finally:
        # Cleanup
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
