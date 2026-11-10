
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeepfakeDataset(Dataset):

    def __init__(self, root_dir, split="train", image_size=224):

        self.root_dir = Path(root_dir)
        self.split = split

        self.class_to_idx = {
            "real": 0,
            "fake": 1
        }

        self.image_paths = []
        self.labels = []

        split_dir = self.root_dir / split

        for class_name, label in self.class_to_idx.items():

            class_dir = split_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Directory not found: {class_dir}"
                )

            for image_path in sorted(class_dir.iterdir()):

                if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.image_paths.append(image_path)
                    self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label
