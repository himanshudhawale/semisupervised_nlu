import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode some inputs
text_1 = "Book a table for Los arcos"
text_2 = "Which restaurant?"
indexed_tokens_1 = tokenizer.encode(text_1)
indexed_tokens_2 = tokenizer.encode(text_2)

# Convert inputs to PyTorch tensors
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor_1 = tokens_tensor_1.to(device)
tokens_tensor_2 = tokens_tensor_2.to(device)
model.to(device)

# Predict all tokens
with torch.no_grad():
    predictions_1, past = model(tokens_tensor_1)
    # past can be used to reuse precomputed hidden state in a subsequent predictions
    # (see beam-search examples in the run_gpt2.py example).
    predictions_2, past = model(tokens_tensor_2, past=past)

# get the predicted last token
# predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
# predicted_token = tokenizer.decode([predicted_index])

pred = torch.argmax(predictions_2, dim=2)[0].cpu().numpy().tolist()
pred_text = tokenizer.decode(pred)

print(pred_text)