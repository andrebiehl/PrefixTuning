import argparse
import logging
import os
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import VisualBertForCausalLM, VisualBertProcessor, AdamW, get_linear_schedule_with_warmup
from PIL import Image

from model import VisualBERTCaptionGenerator
from utils import calculate_bleu

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train VisualBERT for image captioning")
    parser.add_argument("--data_dir", type=str, default="flickr8k", help="Directory containing the dataset")
    parser.add_argument("--model_name_or_path", type=str, default="uclanlp/visualbert-vqa-coco-pre", help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to store the trained model and logs")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--prefix_length", type=int, default=10, help="Length of the prefix for prefix-tuning")
    args = parser.parse_args()
    return args

def preprocess_function(examples):
    images = [Image.open(image_path).convert("RGB") for image_path, _ in examples]
    captions = [random.choice(captions) for _, captions in examples]
    inputs = processor(images=images, text=captions, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

def main(args):
    # Load dataset
    train_image_ids = [line.strip() for line in open(os.path.join(args.data_dir, "Flickr8k_text", "Flickr_8k.trainImages.txt")).readlines()]
    dev_image_ids = [line.strip() for line in open(os.path.join(args.data_dir, "Flickr8k_text", "Flickr_8k.devImages.txt")).readlines()]
    test_image_ids = [line.strip() for line in open(os.path.join(args.data_dir, "Flickr8k_text", "Flickr_8k.testImages.txt")).readlines()]

    captions = {}
    with open(os.path.join(args.data_dir, "Flickr8k_text", "Flickr8k.token.txt"), "r") as f:
        for line in f:
            image_id, caption = line.strip().split("\t")
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)

    train_dataset = [(os.path.join(args.data_dir, "Flickr8k_Dataset", image_id), captions[image_id]) for image_id in train_image_ids]
    dev_dataset = [(os.path.join(args.data_dir, "Flickr8k_Dataset", image_id), captions[image_id]) for image_id in dev_image_ids]
    test_dataset = [(os.path.join(args.data_dir, "Flickr8k_Dataset", image_id), captions[image_id]) for image_id in test_image_ids]

    # Preprocess dataset
    processor = VisualBertProcessor.from_pretrained(args.model_name_or_path)
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    dev_dataset = dev_dataset.map(preprocess_function, batched=True, remove_columns=dev_dataset.column_names)
    test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

    # Load model
    model = VisualBERTCaptionGenerator.from_pretrained(args.model_name_or_path)
    model.prefix_length = args.prefix_length

    # Set up training
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(args.num_train_epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()
        predictions = []
        references = []
        for batch in dev_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model.generate(pixel_values=batch["pixel_values"], max_length=50)
            predicted_captions = processor.batch_decode(outputs, skip_special_tokens=True)
            reference_captions = processor.batch_decode(batch["labels"], skip_special_tokens=True)
            predictions.extend(predicted_captions)
            references.extend(reference_captions)

        bleu_score = calculate_bleu(predictions, references)
        logger.info(f"Epoch {epoch}: BLEU score = {bleu_score}")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)