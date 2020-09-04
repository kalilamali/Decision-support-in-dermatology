import torch
import torchvision
import myutils

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_loaders(dfs, mean, std, size, batch_size, num_workers):
    """
    Function that takes a dictionary of dataframes and
    returns 2 dictionaries of pytorch dataloaders and dataset_sizes
    """
    # Reproducibility
    myutils.myseed(seed=42)

    # Custom pytorch dataloader for this dataset
    class Derm(Dataset):
        """
        Read a pandas dataframe with
        images paths and labels
        """
        def __init__(self, df, transform=None):
            self.df = df
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, index):
            try:
                # Load image data and get label
                X = Image.open(self.df['filenames'][index]).convert('RGB')
                y = torch.tensor(self.df.iloc[index,2:])
            except IOError as err:
                pass

            if self.transform:
                X = self.transform(X)
            # Sanity check
            #print('id:', self.df['id'][index], 'label', y)
            return index, X, y

    # Transforms
    data_transforms = {'train' : transforms.Compose([transforms.Resize(size),
                                              transforms.CenterCrop((size,size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean,std)]),
                       'val' : transforms.Compose([transforms.Resize(size),
                                              transforms.CenterCrop((size,size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean,std)]),
                       'test' : transforms.Compose([transforms.Resize(size),
                                              transforms.CenterCrop((size,size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean,std)]),
                       'unknown' : transforms.Compose([transforms.Resize(size),
                                              transforms.CenterCrop((size,size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean,std)])}

    # Sets
    image_datasets = {x: Derm(dfs[x], transform=data_transforms[x]) for x in dfs.keys()}
    # Sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in dfs.keys()}
    # Loaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size, num_workers, pin_memory=False) for x in dfs.keys()}

    return dataloaders, dataset_sizes
