from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    Trainer,
    TrainingArguments
)
import torch
from peft import LoraConfig, get_peft_model, TaskType

class T5SmallSQL:
    def __init__(self, model_name="cssupport/t5-small-awesome-text-to-sql", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def predict(self, question, schema, max_length=128):
        prompt = f"question: {question} table: {schema}"

        output = self.generator(
            prompt,
            max_length=max_length,
            do_sample=True,
            temperature=0.7
        )

        return output[0]["generated_text"]

    def batch_predict(self, inputs, max_length=128):
        prompts = [
            f"question: {item['question']} table: {item['schema']}"
            for item in inputs
        ]

        outputs = self.generator(prompts, max_length=max_length)

        return [o["generated_text"] for o in outputs]

    def fine_tune_lora(self, train_dataset, output_dir="./t5_sql_lora", epochs=3):
        def preprocess(example):
            inputs = self.tokenizer(
                example["input_text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )

            targets = self.tokenizer(
                example["target_text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )

            inputs["labels"] = targets["input_ids"]
            return inputs

        tokenized_dataset = train_dataset.map(preprocess)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )

        self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            num_train_epochs=epochs,
            learning_rate=3e-4,
            logging_steps=50,
            save_steps=200,
            fp16=torch.cuda.is_available()
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset
        )

        trainer.train()

        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )


    def save(self, path="./t5_sql_model"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.model.to(self.device)

        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )