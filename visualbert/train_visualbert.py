import argparse
import logging
import os
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import VisualBertModel, VisualBertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

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

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image

def get_visual_embeddings(images, device):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    visual_embeds = []
    for image in images:
        image = transform(image)
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            visual_embed = model.visual_bert.visual_embedding(image)
        visual_embeds.append(visual_embed)
    visual_embeds = torch.cat(visual_embeds, dim=0)
    return visual_embeds

def preprocess_function(examples, tokenizer, transform, device):
    images = [preprocess_image(image_path, transform) for image_path, _ in examples]
    captions = [random.choice(captions) for _, captions in examples]
    
    visual_embeds = torch.stack(images).to(device)
    text_inputs = tokenizer(captions, padding="max_length", truncation=True, return_tensors="pt")
    
    inputs = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device),
        "visual_embeds": visual_embeds,
        "visual_attention_mask": torch.ones(visual_embeds.shape[:-1]).to(device),
        "visual_token_type_ids": torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device),
    }
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
            image_id = image_id.split("#")[0]
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)

    train_dataset = [(os.path.join(args.data_dir, "Flickr8k_Dataset", image_id), captions[image_id]) for image_id in train_image_ids]
    dev_dataset = [(os.path.join(args.data_dir, "Flickr8k_Dataset", image_id), captions[image_id]) for image_id in dev_image_ids]
    test_dataset = [(os.path.join(args.data_dir, "Flickr8k_Dataset", image_id), captions[image_id]) for image_id in test_image_ids]

        # Preprocess dataset
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = [preprocess_function(example, tokenizer, transform, device) for example in train_dataset]
    dev_dataset = [preprocess_function(example, tokenizer, transform, device) for example in dev_dataset]
    test_dataset = [preprocess_function(example, tokenizer, transform, device) for example in test_dataset]

    # Load model
    config = VisualBertConfig.from_pretrained(args.model_name_or_path)
    config.output_hidden_states = True
    model = VisualBERTCaptionGenerator.from_pretrained(args.model_name_or_path, config=config)
    model.prefix_length = args.prefix_length
    model.to(device)
    print("Model loaded and prefix length set.")

    # Set up training
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print("Starting training...")

    # Training loop
    for epoch in range(args.num_train_epochs):
        print(f"--- Epoch {epoch+1} ---")
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()
        print("Starting evaluation...")
        predictions = []
        references = []
        for batch in dev_dataloader:
            with torch.no_grad():
                outputs = model.generate(input_ids=batch['input_ids'], 
                                         attention_mask=batch['attention_mask'],
                                         visual_embeds=batch['visual_embeds'],
                                         visual_attention_mask=batch['visual_attention_mask'],
                                         visual_token_type_ids=batch['visual_token_type_ids'],
                                         max_length=50)
            predicted_captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            reference_captions = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            predictions.extend(predicted_captions)
            references.extend(reference_captions)

        bleu_score = calculate_bleu(predictions, references)
        logger.info(f"Epoch {epoch}: BLEU score = {bleu_score}")
        print("Training complete. Saving model...")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
