from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load tokenizer directly from adapter directory (same as base model)
tokenizer = AutoTokenizer.from_pretrained("NEW_FINE_TUNING\\tinyllama-medquad\\new_tinyllama-medquad-finetuned")

# Load base model (TinyLlama)
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load your LoRA adapter
model = PeftModel.from_pretrained(base_model, "NEW_FINE_TUNING\\tinyllama-medquad\\new_tinyllama-medquad-finetuned")

# Inference prompt (as per your training format)
instruction = "What are the symptoms of Ureteral Disorders ?"

prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt")
print("Base model output")
base_mode_outputs = base_model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tokenizer.decode(base_mode_outputs[0], skip_special_tokens=True))

print("////")
print("Tuned model output")
tuned_mode_outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tokenizer.decode(tuned_mode_outputs[0], skip_special_tokens=True))
