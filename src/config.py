import argparse

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--train", type=str2bool, help="Enable model training",default=False)
    args.add_argument("--test", type=str2bool, help="Enable model testing", default=True)
    args.add_argument("--train_dir", type=str, help="Train directory path", default="../data/assignment_train")
    args.add_argument("--test_dir", type=str, help="Test directory path", default="../data/assignment_test")
    args.add_argument("--num_classes", type=int, help="Number of classes", default=4)
    args.add_argument("--batch_size", type=int, help="Batch size", default=90)
    args.add_argument("--lr", type=float, help="Learning rate", default=0.01)
    args.add_argument("--momentum", type=float, help="Momentum", default=0.9),
    args.add_argument("--weight_decay", type=float, help="Weight decay", default=0.0001)
    args.add_argument("--epochs", type=int, help="Number of epoch", default=20)
    args.add_argument("--checkpoint", type=str, help="Checkpoint path", default="../checkpoint")
    return args.parse_known_args()

args, unknown = _parse_args()