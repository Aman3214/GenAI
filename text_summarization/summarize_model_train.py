from dataManip import * # importing functions from dataManip.py
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from evaluate import load


# Tokenizing
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_data(batch):
    inputs = tokenizer(batch["article"], max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    outputs = tokenizer(batch["highlights"], max_length=150, truncation=True, padding="max_length", return_tensors="pt")
    return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": outputs.input_ids}

# Apply tokenization
train_data = train_data.map(preprocess_data, batched=True)
val_data = val_data.map(preprocess_data, batched=True)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Training
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# data evaluation
metric = load("rouge")
model.eval()
predictions = []
references = []

for batch in tqdm(val_loader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    with torch.no_grad():
        summaries = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150)
    
    decoded_summaries = tokenizer.batch_decode(summaries, skip_special_tokens=True)
    decoded_references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
    

results = metric.compute(predictions=decoded_summaries, references=decoded_references)
print(results)

# saving the model
model.save_pretrained("./summarization_model")
tokenizer.save_pretrained("./summarization_model")

