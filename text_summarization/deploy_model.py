from transformers import BartForConditionalGeneration, AutoTokenizer

model = BartForConditionalGeneration.from_pretrained("./summarization_model")
tokenizer = AutoTokenizer.from_pretrained("./summarization_model")

text = "Your input text here."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Summary:", summary)
