from datasets import load_dataset

# Load dataset (e.g., CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Split into train, validation, and test sets
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']
