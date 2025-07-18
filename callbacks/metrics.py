from datasets import load_dataset

class MetricLoggerCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, eval_steps=500, text_column="prompt", target_column="output",n_samples=50):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset  
        self.text_column = text_column    
        self.target_column = target_column  

     def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            # Run full decoding + metrics
            # generate_and_log_metrics()
    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        metrics = {}

        eval_samples = self.eval_dataset.shuffle().select(range(100))
        
        all_predictions = []
        all_references = []

        for sample in eval_samples:
            prompt = sample[self.text_column]
            reference = sample[self.target_column]  # Ground truth
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=50)
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            all_predictions.append(prediction)
            all_references.append([reference])  # BLEU needs list of references

        metrics.update(bleu.compute(predictions=all_predictions, references=all_references))
        metrics.update(rouge.compute(predictions=all_predictions, references=[ref[0] for ref in all_references]))
        metrics.update(
            bertscore.compute(
                predictions=all_predictions,
                references=[ref[0] for ref in all_references],
                lang="en",
                device=model.device,
            )
        )

        wandb.log({
            "eval/bleu": metrics["bleu"],
            "eval/rougeL": metrics["rougeL"],
            "eval/bertscore_f1": np.mean(metrics["f1"]),
            "samples": wandb.Table(
                columns=["Prompt", "Prediction", "Reference"],
                data=list(zip(
                    eval_samples[self.text_column],
                    all_predictions,
                    [ref[0] for ref in all_references]
                ))
            )
        })


val_dataset = load_dataset("your_dataset")["validation"]  
trainer = Trainer(
    callbacks=[
        MetricLoggerCallback(
            tokenizer=tokenizer,
            eval_dataset=val_dataset,
            text_column="question",  
            target_column="answer"
        )
    ]
)
