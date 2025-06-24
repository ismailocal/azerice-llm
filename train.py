import json
from datasets import Dataset

# Yerel JSONL dosyasını oku
with open("wiki_az_api.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Hugging Face Dataset nesnesine çevir
raw_dataset = Dataset.from_list(data)

# Train/validation split
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# Model ve Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)

# Tokenizasyon
def tokenize(example):
    encodings = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

train_tokenized = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_tokenized = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

# Eğitim Ayarları
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir="./az-lora-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    report_to=[],
    logging_dir="./logs",
    save_strategy="epoch",
    logging_steps=10,
    fp16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized
)
trainer.train()

# KAYDETME
model.save_pretrained("./az-lora-model")
tokenizer.save_pretrained("./az-lora-model")
