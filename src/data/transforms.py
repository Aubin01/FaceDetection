from torchvision import transforms


def build_transforms(image_size: int, augment_cfg: dict):
    train_transforms = [
        transforms.Resize((image_size, image_size)),
    ]

    if augment_cfg.get("horizontal_flip", False):
        train_transforms.append(transforms.RandomHorizontalFlip())
    if augment_cfg.get("rotation", 0):
        train_transforms.append(transforms.RandomRotation(augment_cfg["rotation"]))
    
    # Add brightness and contrast for lighting variation
    if augment_cfg.get("brightness") or augment_cfg.get("contrast"):
        brightness = augment_cfg.get("brightness", 0)
        contrast = augment_cfg.get("contrast", 0)
        train_transforms.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
    
    if augment_cfg.get("color_jitter", None):
        cj = augment_cfg["color_jitter"]
        train_transforms.append(transforms.ColorJitter(*cj))
    
    # Add Gaussian blur to simulate camera quality
    if augment_cfg.get("gaussian_blur"):
        train_transforms.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, augment_cfg["gaussian_blur"])))

    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    eval_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return transforms.Compose(train_transforms), eval_transforms
