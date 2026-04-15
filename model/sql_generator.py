from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
import torch
from peft import LoraConfig, get_peft_model, TaskType

class CausalSQLModel:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-Coder-1.5B",
        finetune_type="lora",
        device=None
    ):
        self.model_name = model_name
        self.finetune_type = finetune_type.lower()

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.finetune_type == "qlora":
            print("Loading model with QLoRA (4-bit)...")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )

        else:
            print("Loading model with standard LoRA...")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name
            ).to(self.device)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def _format_prompt(self, question, schema):
        return f"""
            You are an expert SQL generator.

            ### Instructions:
            - Generate a correct SQL query
            - Use only given schema
            - Do not hallucinate columns or tables

            ### Database Schema:
            {schema}

            ### Question:
            {question}

            ### SQL Query:
        """

    def predict(self, question, schema, max_length=200):
        prompt = self._format_prompt(question, schema)

        output = self.generator(
            prompt,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return output[0]["generated_text"].split("### SQL Query:")[-1].strip()

    def batch_predict(self, inputs, max_length=200):
        prompts = [self._format_prompt(i["question"], i["schema"]) for i in inputs]

        outputs = self.generator(
            prompts,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return [
            o["generated_text"].split("### SQL Query:")[-1].strip()
            for o in outputs
        ]

    def fine_tune(self, train_dataset, output_dir="./sql_model", epochs=3):
        def preprocess(example):
            prompt = self._format_prompt(example["question"], example["schema"])
            full_text = prompt + example["sql"]

            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=512
            )

            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = train_dataset.map(preprocess)
        keep_columns = {"input_ids", "attention_mask", "labels", "token_type_ids"}
        remove_columns = [
            c for c in tokenized_dataset.column_names if c not in keep_columns
        ]
        if remove_columns:
            tokenized_dataset = tokenized_dataset.remove_columns(remove_columns)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )

        self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2, 
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            learning_rate=2e-4,
            logging_steps=50,
            save_steps=200,
            fp16=torch.cuda.is_available(),
            report_to=[],
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset
        )

        trainer.train()

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def save(self, path="./sql_model"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.model.to(self.device)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )