# 🧠 Meta-LLaMA3 8B Instruct — Fine-tuned on Paul Graham Essays

**Model Name:** `junaid17/meta-llama3-8b-instruct-finetuned-paul-graham`  
**Base Model:** [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)  

---

## 🧩 Overview

This model is a fine-tuned version of Meta's LLaMA 3 8B Instruct, trained specifically on the complete corpus of **Paul Graham's essays**. The goal was to capture not only his ideas but also his *voice, reasoning style, and tone*—making this a thought-partner in the spirit of PG himself.

> Imagine having a late-night conversation with Paul Graham about startups, ambition, hacking, or education. This model aims to simulate exactly that.

---

## 📚 Training Data

- ✅ ~75 high-quality essays by Paul Graham, scraped and cleaned from [paulgraham.com](http://paulgraham.com)
- ✅ Text was normalized, tokenized, and structured in chat-style format (`role: user` / `role: assistant`)
- ✅ Special attention was given to preserving paragraph structure and reflective tone

---
![Screenshot 2025-06-29 215001](https://github.com/user-attachments/assets/e5c87256-3124-44fc-9d6b-c8da6b365d24)

## 🛠️ How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model = AutoModelForCausalLM.from_pretrained("junaid17/meta-llama3-8b-instruct-finetuned-paul-graham", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("junaid17/meta-llama3-8b-instruct-finetuned-paul-graham")

messages = [
    {"role": "user", "content": "Why do most startups fail?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)

_ = model.generate(
    input_ids=inputs.input_ids,
    streamer=streamer,
    max_new_tokens=500,
    use_cache=True
)



