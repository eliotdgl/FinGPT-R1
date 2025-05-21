import argparse
import sys
from datasets import load_dataset

def load_and_save_dataset(dataset_name: str, PATH: str, save:bool = False):    
    dataset = load_dataset(dataset_name)

    if save:
        dataset.save_to_disk(PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and optionally save a Hugging Face dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to load (e.g., 'raeidsaqur/nifty').")
    parser.add_argument("--save", action="store_true", help="Include this flag to save the dataset.")
    parser.add_argument("--path", type=str, help="Directory path to save the dataset, if --save is True")

    args = parser.parse_args()

    if args.save and not args.path:
        print("\nError: --path must be specified if --save is True.\n")
        parser.print_help()
        print("\n")
        sys.exit(1)
  
    
    if args.path and not args.save:
        print("\nWarning: --path is specified but --save is False. Dataset will not be saved.\n")
        parser.print_help()
        print("\n")


    load_and_save_dataset(args.dataset, args.path, args.save)