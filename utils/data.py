import os
import random
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils.augmentation import CIFAR10Policy, Cutout


class CIFAR10_DVS_AUG(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        new_data = []
        for t in range(data.size(0)):
            new_data.append(self.tensorx(self.resize(self.imgx(data[t, ...]))))
        data = torch.stack(new_data, dim=0)
        if self.transform is not None:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


def load_dataset(name: str, root: str, cutout: bool = False, auto_aug: bool = False):
    """
    Load and preprocess specified dataset by name.

    Args:
        name (str): Dataset name (e.g., 'CIFAR10', 'IMAGENET'). Case-insensitive.
        root (str): Root directory for dataset storage.
        cutout (bool, optional): Whether to use Cutout augmentation for CIFAR. Defaults to False.
        auto_aug (bool, optional): Whether to use AutoAugment for CIFAR. Defaults to False.

    Raises:
        NotImplementedError: If the requested dataset name is not supported.

    Returns:
        tuple: (train_data, val_data, num_class, input_size)
    """
    if name is None:
        raise ValueError("Dataset name cannot be None.")
    
    name = name.upper()

    if name in ['CIFAR10', 'CIFAR100']:
        # CIFAR dataset configuration
        input_size = (3, 32, 32)
        
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if auto_aug:
            train_transform_list.append(CIFAR10Policy())
        
        train_transform_list.append(transforms.ToTensor())
        
        if cutout:
            train_transform_list.append(Cutout(n_holes=1, length=16))

        if name == 'CIFAR10':
            num_class = 10
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            dataset_loader = datasets.CIFAR10
        else: # CIFAR100
            num_class = 100
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            dataset_loader = datasets.CIFAR100
            
        train_transform_list.append(normalize)
        train_transform = transforms.Compose(train_transform_list)
        val_transform = transforms.Compose([transforms.ToTensor(), normalize])

        train_data = dataset_loader(root=root, train=True, download=True, transform=train_transform)
        val_data = dataset_loader(root=root, train=False, download=True, transform=val_transform)

    elif name == 'IMAGENET':
        # ImageNet dataset configuration
        num_class = 1000
        input_size = (3, 224, 224)
        traindir = os.path.join(root, 'train')
        valdir = os.path.join(root, 'val')
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_data = datasets.ImageFolder(root=traindir, transform=train_transform)
        val_data = datasets.ImageFolder(root=valdir, transform=val_transform)

    elif name == 'MNIST':
        # MNIST dataset configuration
        num_class = 10
        input_size = (1, 28, 28)
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.ToTensor(), normalize])

        train_data = datasets.MNIST(root=root, train=True, download=True, transform=train_transform)
        val_data = datasets.MNIST(root=root, train=False, download=True, transform=val_transform)
    
    elif name == 'CIFAR10_DVS_AUG':
        # Custom CIFAR10_DVS_AUG dataset configuration
        num_class = 10
        input_size = (2, 48, 48)
        
        train_path = os.path.join(root, 'train')
        val_path = os.path.join(root, 'test')
        
        train_data = CIFAR10_DVS_AUG(root=train_path, transform=False)
        val_data = CIFAR10_DVS_AUG(root=val_path)

    else:
        raise NotImplementedError(f"Dataset '{name}' is not implemented or supported.")

    return train_data, val_data, num_class, input_size
