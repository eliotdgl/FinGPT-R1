import argparse
import sys

def train():


if __name__ == "__main__":
    valid_models = ['BERT', 'BERTEC', 'HashT', 'HashTEC', 'DelT', 'DelTEC']
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--model", 
        nargs="+",
        type=str,
        choices=valid_models,
        required=True, 
        help="Model(s) to train ('BERT', 'BERTEC', 'HashT', 'HashTEC', 'DelT', 'DelTEC').")

    args = parser.parse_args()

    if args.model not in:
        print("\nError: --path must be specified if --save is True.\n")
        parser.print_help()
        print("\n")
        sys.exit(1)
  
    train(args.model)