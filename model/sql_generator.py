from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
import torch
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
try:
    from datasets import Dataset, concatenate_datasets
except Exception:
    Dataset = None
    concatenate_datasets = None

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
        self.is_qlora = self.finetune_type == "qlora"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.is_qlora:
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
                device_map={"": 0} if self.device == "cuda" else None
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

    def _format_prompt_question_inference(self, sql, schema, evidence=None):
        evidence_block = f"\n### Evidence:\n{evidence}\n" if evidence else ""
        return f"""
            You are an expert at understanding SQL semantics.

            ### Instructions:
            - Infer the natural language question answered by the SQL query
            - Use the database schema and evidence carefully
            - Keep the question precise and unambiguous

            ### Database Schema:
            {schema}
            {evidence_block}
            ### SQL Query:
            {sql}

            ### Inferred Question:
        """

    def _format_prompt_evidence_inference(self, question, schema, sql, candidates: list[str]):
        numbered_candidates = "\n".join(
            [f"{idx + 1}. {candidate}" for idx, candidate in enumerate(candidates or [])]
        )
        return f"""
            You are an expert at evidence selection for Text-to-SQL.

            ### Instructions:
            - Select the single most relevant evidence candidate
            - Base your decision on question, schema, and SQL
            - Return exactly one evidence string

            ### Database Schema:
            {schema}

            ### Question:
            {question}

            ### SQL Query:
            {sql}

            ### Evidence Candidates:
            {numbered_candidates}

            ### Best Evidence:
        """

    def _format_prompt_self_refine(self, question, schema, previous_sql, execution_result, evidence=None):
        evidence_block = f"\n### Evidence:\n{evidence}\n" if evidence else ""
        return f"""
            You are an expert SQL debugger and corrector.

            ### Instructions:
            - Fix the previous SQL query using execution feedback
            - Use only the provided schema and evidence
            - Return only the corrected SQL query

            ### Database Schema:
            {schema}
            {evidence_block}
            ### Question:
            {question}

            ### Previous SQL:
            {previous_sql}

            ### Execution Result / Error:
            {execution_result}

            ### Corrected SQL:
        """

    def _generate_from_prompt(self, prompt, output_tag, max_length=200):
        output = self.generator(
            prompt,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return output[0]["generated_text"].split(output_tag)[-1].strip()

    def predict(self, question, schema, max_length=200):
        prompt = self._format_prompt(question, schema)
        return self._generate_from_prompt(prompt, "### SQL Query:", max_length=max_length)

    def infer_question(self, sql, schema, evidence=None, max_length=200) -> str:
        prompt = self._format_prompt_question_inference(sql, schema, evidence=evidence)
        return self._generate_from_prompt(prompt, "### Inferred Question:", max_length=max_length)

    def infer_evidence(self, question, schema, sql, candidates, max_length=200) -> str:
        prompt = self._format_prompt_evidence_inference(question, schema, sql, candidates)
        return self._generate_from_prompt(prompt, "### Best Evidence:", max_length=max_length)

    def infer_self_refine(self, question, schema, previous_sql, execution_result, evidence=None, max_length=200) -> str:
        prompt = self._format_prompt_self_refine(
            question,
            schema,
            previous_sql,
            execution_result,
            evidence=evidence
        )
        return self._generate_from_prompt(prompt, "### Corrected SQL:", max_length=max_length)

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

    @staticmethod
    def build_multitask_dataset(datasets: dict) -> Dataset:
        if Dataset is None or concatenate_datasets is None:
            raise ImportError(
                "Package 'datasets' is required for build_multitask_dataset. "
                "Install with: pip install datasets"
            )
        prepared = []
        for task_type, ds in datasets.items():
            ds_with_task = ds
            if "task_type" in ds_with_task.column_names:
                ds_with_task = ds_with_task.map(lambda x, t=task_type: {"task_type": x.get("task_type", t)})
            else:
                ds_with_task = ds_with_task.add_column("task_type", [task_type] * len(ds_with_task))
            prepared.append(ds_with_task)

        combined = concatenate_datasets(prepared)
        return combined.shuffle(seed=42)

    def fine_tune(
        self,
        train_dataset,
        output_dir="./sql_model",
        epochs=3,
        task_weights: dict = {
            "text_to_sql": 1.0,
            "question_inference": 1.0,
            "evidence_inference": 1.0,
            "self_refine": 1.0
        }
    ):
        if "task_type" in train_dataset.column_names and concatenate_datasets is None:
            raise ImportError(
                "Package 'datasets' is required for multi-task fine-tuning with task weights. "
                "Install with: pip install datasets"
            )
        weighted_dataset = train_dataset
        if "task_type" in train_dataset.column_names:
            buckets = []
            for task_name, weight in task_weights.items():
                if weight <= 0:
                    continue
                task_ds = train_dataset.filter(lambda x, t=task_name: x.get("task_type", "text_to_sql") == t)
                if len(task_ds) == 0:
                    continue
                repeat = max(1, int(round(weight)))
                buckets.extend([task_ds] * repeat)
            if buckets:
                weighted_dataset = concatenate_datasets(buckets).shuffle(seed=42)

        def preprocess(example):
            task_type = example.get("task_type", "text_to_sql")

            if task_type == "question_inference":
                prompt = self._format_prompt_question_inference(
                    example.get("sql", ""),
                    example.get("schema", ""),
                    evidence=example.get("evidence")
                )
                target = example.get("question", "")
            elif task_type == "evidence_inference":
                prompt = self._format_prompt_evidence_inference(
                    example.get("question", ""),
                    example.get("schema", ""),
                    example.get("sql", ""),
                    example.get("evidence_candidates", [])
                )
                target = example.get("correct_evidence", "")
            elif task_type == "self_refine":
                prompt = self._format_prompt_self_refine(
                    example.get("question", ""),
                    example.get("schema", ""),
                    example.get("previous_sql", ""),
                    example.get("execution_result", ""),
                    evidence=example.get("evidence")
                )
                target = example.get("sql", "")
            else:
                prompt = self._format_prompt(example.get("question", ""), example.get("schema", ""))
                target = example.get("sql", "")

            full_text = prompt + target

            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=512
            )

            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=512
            )
            
            prompt_length = min(len(prompt_tokens["input_ids"]), len(tokenized["input_ids"]))
            labels = tokenized["input_ids"].copy()
            labels[:prompt_length] = [-100] * prompt_length
            tokenized["labels"] = labels
            return tokenized

        tokenized_dataset = weighted_dataset.map(preprocess)

        keep_columns = {"input_ids", "attention_mask", "labels"}
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )

        if self.is_qlora:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.gradient_checkpointing_enable()

        self.model = get_peft_model(self.model, lora_config)
        self.model.config.use_cache = False
        self.model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            learning_rate=2e-4,
            logging_steps=1,
            save_steps=200,
            fp16=torch.cuda.is_available(),
            report_to=[],
            dataloader_pin_memory=torch.cuda.is_available(),
            remove_unused_columns=False,
            optim="paged_adamw_8bit" if self.is_qlora else "adamw_torch",
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
            device_map="auto" if self.is_qlora else None,
            device=0 if (self.device == "cuda" and not self.is_qlora) else -1
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