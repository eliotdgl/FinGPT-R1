from datasets import load_dataset

dataset = load_dataset("raeidsaqur/nifty")
dataset.save_to_disk("nifty_dataset_local")