from datasets import load_metric

def calculate_bleu(predictions, references):
    bleu_metric = load_metric("sacrebleu")
    results = bleu_metric.compute(predictions=predictions, references=references)
    return results["score"]