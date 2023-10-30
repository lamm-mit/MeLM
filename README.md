# Language Modeling Strategies for Mechanics and Materials

For centuries, researchers have sought out ways to connect disparate areas of knowledge. While early scientists and engineers (e.g. Galileo, da Vinci, and others) were often scholars across fields, specialization has taken hold later. Now, with the advent of AI we can explore deep relationships across areas that venture to connect technical disciplines (e.g., mechanics and chemistry) or general domains of knowledge (e.g., failure mechanics and art). Here we propose a workflow to develop a fine-tuned Large Language Model (LLM), exemplified for a subset of knowledge in materials failure and multiscale modeling, and discuss its application in various use cases. The modeling strategy includes the use of general-purpose LLMs to extract question-answer pairs from raw data followed by fine-tuning a LLM. The resulting MechGPT LLM is used in a series of computational experiments to explore its capacity for knowledge retrieval, language tasks, hypothesis generation, and connecting knowledge across disparate areas of science. We further explore the use of LLMs to generate ontological knowledge graphs, or ologs, to elucidate mechanistic, interpretable graph structures that provide explanatory insights, frameworks for new research questions, and visual representations of knowledge.  This work shows the potential of LLMs to complement the way we model problems in mechanics and materials, enabling faster, more efficient, and more accurate research and engineering. The flexible multi-stage training strategy is transferrable and offers a path to obtain other fine-tuned models in other fields of mechanics. Three versions of MechGPT are discussed, featuring different sizes from13 billion to 70 billion parameters, and reaching context lengths of more than 10,000 tokens. 

![image](https://github.com/lamm-mit/MeLM/assets/101393859/6378fc94-198c-4a50-95ce-52ff88e0d8de)

This repository also features codes for the multi-modal mechanics language model, MeLM, applied to solve various nonlinear forward and inverse problems, that can deal with a set of instructions, numbers and microstructure data. The framework is applied to various examples including bio-inspired hierarchical honeycomb design, carbon nanotube mechanics, and protein unfolding. In spite of the flexible nature of the model–which allows us to easily incorporate diverse materials, scales, and mechanical features–the model performs well across disparate forward and inverse tasks. Based on an autoregressive attention-model, MeLM effectively represents a large multi-particle system consisting of hundreds of millions of neurons, where the interaction potentials are discovered through graph-forming self-attention mechanisms that are then used to identify relationships from emergent structures, while taking advantage of synergies discovered in the training data. We show that the model can solve complex degenerate mechanics design problems and determine novel material architectures across a range of hierarchical levels, providing an avenue for materials discovery and analysis. To illustrate the use case for broader possibilities, we outline a human-machine interactive MechGPT model, here trained on a set of 1,103 Wikipedia articles related to mechanics, showing how the general framework can be used not only to solve forward and inverse problems but in addition, for complex language tasks like summarization, generation of new research concepts, and knowledge extraction. Looking beyond the demonstrations reported in this paper, we discuss other opportunities in applied mechanics and general considerations about the use of large language models in modeling, design, and analysis that can span a broad spectrum of material properties from mechanical, thermal, optical, to electronic.

![image](https://github.com/lamm-mit/MeLM/assets/101393859/8191f1a0-1f4c-4221-96b1-0680a1c2d57d)

# Dataset, weights, and codes

Weights for the MechGPT-13b-v106C model: https://www.dropbox.com/scl/fi/3q9w685uvcdjh7qfdo9fg/MechGPT-13b_v106C.zip?rlkey=igjmux7waxllb9i2vgvjh7ez7&dl=0 

Associated dataset: https://www.dropbox.com/scl/fi/jwe8t6mv5s99kul2bjtl9/QA_dataset_mechanics.csv?rlkey=acy491zfwsvu5bdexf4w4gaub&dl=0 

Various codes are included as Jupyter notebooks.

Install PyTorch and other associated packages. Additional packages needed can be installed as follows:

```
pip install -U transformers peft bitsandbytes gradio pymed scholarly
```

To load the model:

```
bnb_config4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model_base = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config= bnb_config4bit,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model_base, peft_model_id,
                                )
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```
Generation:
```
outputs = model.generate(input_ids=inputs.to(device), 
                                   max_new_tokens=max_new_tokens,
                                   temperature=0.4, 
                                   num_beams=1,
                                   top_k = 50,
                                   top_p = 0.9,
                                   num_return_sequences = 1, eos_token_id=[2, 32000],
                                   do_sample =True,
                                   repetition_penalty=repetition_penalty,
                                  )
tokenizer.batch_decode(outputs[:,inputs.shape[1]:].detach().cpu().numpy(), skip_special_tokens=True)
```

# Training

The dataset used for training is provided in 'text_list' is a list of text used for training, here: Question-answer pairs formatted in relevant prompt format.

Prompt format:
```
User: [Question]<|end_of_turn|>Assistant: [Answer]<|end_of_turn|>
```
Training code: 
```
from transformers import TrainingArguments, DataCollatorForSeq2Seq

train_dataset = Dataset( text_list )

output_dir = output_dir
per_device_train_batch_size = batch_size
gradient_accumulation_steps = gradient_accumulation_steps
optim = "paged_adamw_32bit"
save_steps = steps_per_epoch 
logging_steps = steps_per_epoch
learning_rate = learning_rate
max_grad_norm = 0.3
max_steps = total_steps
 
lr_scheduler_type = "cosine"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    #fp16=True,
    bf16=True,  
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=0.03,
    lr_scheduler_type=lr_scheduler_type,
    save_total_limit=20,
    report_to= "none", #change if you want to use wandb
)

trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_arguments,
        data_collator= DataCollatorForSeq2Seq(    tokenizer, return_tensors="pt", padding=True),
        #callbacks=[push_callback],
    )

trainer.train()
```
