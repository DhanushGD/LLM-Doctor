# 🚀 LLM Doctor: TinyLlama Fine-Tuned on MedQuAD with Ollama Integration

This project demonstrates an **end-to-end pipeline** for fine-tuning the **TinyLlama-1.1B** model on the **MedQuAD** medical dataset using **Lora** & **QLoRA** with **Unsloth**, and deploying the fine-tuned model locally using **Ollama**. It showcases how to create domain-specific models efficiently on consumer hardware with **parameter-efficient fine-tuning** techniques.

---

## 📊 Key Highlights

✅ Fine-tuned **TinyLlama-1.1B** on the **MedQuAD** medical dataset  
✅ Used  **Lora** & **QLoRA** (Quantized Low-Rank Adaptation) with **Unsloth** for efficient fine-tuning 
✅ Saved the model in **GGUF format** for easy integration with Ollama using `unsloth.save_gguf()`  
✅ Integrated fine-tuned model with **Ollama** for local inference  
✅ **Colab Notebook** included for step-by-step fine-tuning workflow.  
✅ Proof-of-Concept (PoC) Metrics:
- **Perplexity**: 56.70 → **39.07** (after fine-tuning)
- **Accuracy**: 0.0100 → **0.0048**

---

## 🧑‍💻 Tech Stack

| Tech | Purpose |
|---|---|
| TinyLlama-1.1B | 4-bit model via unsloth |
| MedQuAD Dataset from Kaggle | Medical QA dataset |
| Unsloth | Lora and QLoRA fine-tuning framework |
| LoRA | Parameter-efficient fine-tuning |
| QLoRA | Parameter-efficient fine-tuning |
| Ollama| Local model inference |

---

## 🏗️ Project Workflow

1️⃣ **Prepare Dataset**  
- Format MedQuAD into **instruction-output** pairs  

2️⃣ **Fine-Tune with Lora and QLoRA (via Unsloth)**  
- LoRA settings: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`  
- 4-bit quantization for memory efficiency  - QLoRA
- Trained on Google Colab with **A100 GPU**  
- Training config:
  - Batch Size: 4
  - Gradient Accumulation: 2
  - Epochs: 1 (demo)

3️⃣ **Save the Model for Ollama**  
After training, we **save the fine-tuned model in GGUF format** using Unsloth's `save_gguf()` utility:
```python
from unsloth import FastLanguageModel
FastLanguageModel.save_gguf(
    model=model,
    tokenizer=tokenizer,
    output_dir="path_to_save_model",
    quantization="q4_k_m"  # or your preferred quantization
)
```
4️⃣ Deploy Model via Ollama
- Created **Modelfile** for Ollama
  ```bash
  # Modelfile for Ollama - Fine-tuned TinyLlama on MedQuAD

  FROM ./unsloth.F16.gguf
  SYSTEM You are a helpful AI assistant, fine-tuned on medical data from the MedQuAD dataset.
  PARAMETERS
      num_predict 200
    ```
  - Built with:
    ```bash
    ollama create tinyllama-medquad -f Modelfile
    ollama run tinyllama-medquad
      ```

5️⃣ Evaluate Model
- Measured perplexity and accuracy
- Sample responses compared before/after fine-tuning

## 📓 Colab Notebook
For new users, a Google Colab Notebook is provided in this repository to:
✅ Fine-tune TinyLlama with Lora and QLoRA and Unsloth
✅ Save the model in drive and GGUF format using save_gguf()
✅ Track metrics like perplexity, accuracy, before/after outputs
✅ Export the model for Ollama integration

## 🔬 Proof-of-Concept (PoC) Metrics

✅ Training Logs:

![image](https://github.com/user-attachments/assets/b955de0f-e801-4a48-a225-db82684900e9)

![image](https://github.com/user-attachments/assets/8efbc8e5-e3ec-45d5-a6b4-b1caf8b39c76)

✅ Evaluation Metrics:

![image](https://github.com/user-attachments/assets/173bb025-829b-4d78-8210-ee821fd51fcb)
![image](https://github.com/user-attachments/assets/930781c2-b350-4104-a095-7e42e227e434)

✅ Before/After Model Outputs:

![image](https://github.com/user-attachments/assets/4ae1b0c8-1157-4b57-95d8-1af5e484ab80)

✅ Ollama Run Example:

![image](https://github.com/user-attachments/assets/0a176b2d-844c-4850-ba55-81485af37b15)

✅ Directory :

![image](https://github.com/user-attachments/assets/15bb6f9d-ec0d-4f82-9aa6-c5027beee994)

## Contributing
Feel free to fork this repository, make improvements, or submit issues for bugs or enhancements. Contributions are always welcome!
