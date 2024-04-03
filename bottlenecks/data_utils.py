import sys
from configs import *
import transformers
from datasets import load_dataset
from datasets import DatasetDict


def collate_fn(batch):
    return {
        "image": [x["image"] for x in batch],
        "labels": [x["labels"] for x in batch],
    }


def remove_prefixes(strings):
    prefixes = ["a", "an", "the"]
    result = []

    for string in strings:
        words = string.split()
        if words[0].lower() in prefixes:
            result.append(" ".join(words[1:]))
        else:
            result.append(string)

    return result


def preprocess_loader(
    loader, concepts: list, backbone_name: str = "openai/clip-vit-base-patch32"
):
    preprocessed_batches = []
    if backbone_name == Constants.altclip_link:
        processor = transformers.AltCLIPProcessor.from_pretrained(backbone_name)
    elif backbone_name == Constants.align_link:
        processor = transformers.AlignProcessor.from_pretrained(backbone_name)
    elif backbone_name in [
        Constants.siglip_so_link,
        Constants.siglip_base_link,
        Constants.siglip_large_link,
        Constants.siglip_large_256_link,
    ]:
        processor = transformers.AutoProcessor.from_pretrained(backbone_name)
    else:
        processor = transformers.CLIPProcessor.from_pretrained(backbone_name)
    for batch in tqdm(loader):
        preprocessed_batch = preprocess_batch(batch, processor, concepts)
        preprocessed_batches.append(preprocessed_batch)
    return preprocessed_batches


def preprocess_batch(batch, processor, concepts: list):
    return (
        processor(
            text=concepts, images=batch["image"], return_tensors="pt", padding=True
        ),
        batch["labels"],
    )


def prepared_dataloaders(
    hf_link: str,
    concepts: list,
    test_size: int = 0.2,
    prep_loaders="all",
    batch_size: int = 32,
    backbone_name: str = "openai/clip-vit-base-patch32",
):
    dataset = load_dataset(hf_link)
    if "label" in dataset["train"].features:
        dataset = dataset.rename_column("label", "labels")
    dataset = dataset["train"].train_test_split(test_size=test_size)
    val_test = dataset["test"].train_test_split(test_size=0.5)
    dataset = DatasetDict(
        {
            "train": dataset["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    train_loader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if prep_loaders == "all":
        train_loader_preprocessed = preprocess_loader(
            train_loader, concepts, backbone_name=backbone_name
        )
        val_loader_preprocessed = preprocess_loader(
            val_loader, concepts, backbone_name=backbone_name
        )
        test_loader_preprocessed = preprocess_loader(
            test_loader, concepts, backbone_name=backbone_name
        )
        return (
            train_loader_preprocessed,
            val_loader_preprocessed,
            test_loader_preprocessed,
        )
    elif prep_loaders == "train":
        train_loader_preprocessed = preprocess_loader(
            train_loader, concepts, backbone_name=backbone_name
        )
        return train_loader_preprocessed
    elif prep_loaders == "val":
        val_loader_preprocessed = preprocess_loader(
            val_loader, concepts, backbone_name=backbone_name
        )
        return val_loader_preprocessed
    elif prep_loaders == "test":
        test_loader_preprocessed = preprocess_loader(
            test_loader, concepts, backbone_name=backbone_name
        )
        return test_loader_preprocessed
