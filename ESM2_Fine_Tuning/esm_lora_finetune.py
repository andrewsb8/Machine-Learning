import torch
import random
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForMaskedLM, AutoTokenizer

base_model_path = "facebook/esm2_t6_8M_UR50D"
base_model = AutoModelForMaskedLM.from_pretrained(base_model_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokens = loaded_tokenizer.get_vocab()

# add dummy function to get around ESM-specific error with LoRA
def prepare_inputs_for_generation(self, inputs, **kwargs):
    return inputs
base_model.prepare_inputs_for_generation = prepare_inputs_for_generation

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "key"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
lora_model = get_peft_model(base_model, config)

# prepare data for training
abeta_1_43 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIAT"
rand_int = random.randint(0,44)
masked_seq = abeta_1_43[:rand_int] + "<mask>" + abeta_1_43[rand_int+1:]
training_set = loaded_tokenizer(masked_seq, return_tensors="pt", truncation=True, max_length=44, padding='max_length')
training_labels = []
training_labels.append([-100 for i in range(44)])
training_labels[0][rand_int] = tokens[abeta_1_43[rand_int]]
training_labels = torch.Tensor(training_labels)
training_labels = training_labels.long()

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return X, y

dataset = MyDataset(training_set["input_ids"], training_labels)
train_loader = torch.utils.data.DataLoader(dataset) #, shuffle=True)

# 4. Set up optimizer and loss function
optimizer = torch.optim.Adam(lora_model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

# 5. Retrain the model
lora_model.train()
for epoch in range(3):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = lora_model(data)
        logits = output["logits"]
        loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
        print(loss)
        loss.backward()
        optimizer.step()

torch.save(lora_model, "esm2-finetune.pt")
