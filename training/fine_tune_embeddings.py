"""
Fine-tune a sentence-embedding model on domain-specific query/passage pairs
using contrastive learning (MultipleNegativesRankingLoss).

This is the "PyTorch" part of the project: sentence-transformers is built on
top of PyTorch, and this script runs a real training loop, not just an
API call to a hosted embedding model.
"""
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = Path(__file__).parent.parent / "data" / "training_pairs.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "models" / "fine_tuned_embedder"

EPOCHS = 10
BATCH_SIZE = 8
WARMUP_STEPS = 10


def load_training_pairs(path: Path) -> list[InputExample]:
    examples = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            examples.append(InputExample(texts=[row["query"], row["passage"]]))
    return examples


def main():
    print(f"Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    print(f"Loading training pairs from {DATA_PATH}")
    train_examples = load_training_pairs(DATA_PATH)
    print(f"Loaded {len(train_examples)} query/passage pairs")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # MultipleNegativesRankingLoss treats every other passage in the batch
    # as a negative for a given query -> pulls matching query/passage pairs
    # together in embedding space, pushes non-matching pairs apart.
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("Starting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        show_progress_bar=True,
    )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_PATH))
    print(f"Fine-tuned model saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
