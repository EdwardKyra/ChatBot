from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer with left padding
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Start fresh
chat_history_ids = None

print("Talk to Buddy! (type 'exit' to quit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Buddy: Finally, some peace and quiet. Byeee 👋")
        break

    # Build the input prompt with sass
    prompt = f"{tokenizer.eos_token}User: {user_input}{tokenizer.eos_token}Buddy (with sass):"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt', return_attention_mask=True, padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Add past chat history if it exists
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
        attention_mask = torch.cat([torch.ones_like(chat_history_ids), attention_mask], dim=-1)

    # Generate a new response
    chat_history_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[-1] + 100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # Decode and print only the new Buddy part
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Buddy:", response.strip())
