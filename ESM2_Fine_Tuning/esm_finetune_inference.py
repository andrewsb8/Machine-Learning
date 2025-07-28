import torch
import random
from transformers import AutoTokenizer

model = torch.load("esm2-finetune.pt", weights_only=False)
tokenizer = AutoTokenizer.from_pretrained("esm2-finetune-tokens")

abeta_1_43 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIAT"
rand_int = random.randint(0,43)
masked_seq = abeta_1_43[:rand_int] + "<mask>" + abeta_1_43[rand_int+1:]
inputs = tokenizer(masked_seq, return_tensors="pt", truncation=True, max_length=44, padding='max_length')

with torch.no_grad():
    results = model(**inputs)
    logits = results["logits"]  # Shape: (batch, sequence, vocab)

predicted_aa = torch.argmax(logits[0, 10, :]).item()
predicted_seq_token = [torch.argmax(logits[0, i, :]).item() for i in range(len(abeta_1_43))]
amino_acid_sequence = tokenizer.decode(predicted_seq_token, skip_special_tokens=True)
aa_seq_clean = "".join([i for i in amino_acid_sequence if i != " "])
print("Masked Position, Correct AA, Predicted AA: ", rand_int, abeta_1_43[rand_int], aa_seq_clean[rand_int])
print("Original Sequence: ", abeta_1_43)
print("Output Sequence: ", aa_seq_clean)
