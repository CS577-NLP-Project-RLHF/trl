import evaluate

eval_results = task_evaluator.compute(
    model_or_pipeline="huggyllama/llama-7b",
    data=data,
    metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]),
    label_mapping={"NEGATIVE": 0, "POSITIVE": 1}
)
print(eval_results)