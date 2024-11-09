import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--train', type=str2bool, help="Enable model training",default=True)
    args.add_argument('--test', type=str2bool, help="Enable model testing", default=True)
    args.add_argument('--train_dir', type=str, help="Train directory path", default="../data/assignment_train")
    args.add_argument('--test_dir', type=str, help="Test directory path", default="../data/assignment_test")
    args.add_argument('--batch_size', type=int, help="Batch size", default=64)
    args.add_argument('--epochs', type=int, help="Number of epoch", default=10)
    args.add_argument('--checkpoint', type=str, help="Checkpoint path", default="../checkpoint")
    return args.parse_known_args()

args, unknown = _parse_args()