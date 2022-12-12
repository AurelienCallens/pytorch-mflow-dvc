import yaml
import torch
import torchvision.transforms as T


import sys
sys.path.append('src/')
from data.CustomImageDataset import CustomImageDataset
from utils.mlflow_run_decorator import mlflow_run


@mlflow_run
def get_mean_and_std(loader):
    mean = 0. 
    std = 0. 
    total_images_count = 0

    for images,_ in loader:
        image_count_batch = images.size(0)
        images = images.view(image_count_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_batch
    
    mean /= total_images_count
    std /= total_images_count
    return mean, std


if __name__ == '__main__':
    # Read params
    params = yaml.safe_load(open('./params.yaml'))
    image_size = params['prepare']['image_size']

    transformf = T.Compose(
        [T.ToPILImage(),
        T.Resize(size=image_size),
        T.ToTensor()])
    
    # Load all the images in dataloader 
    dataset = CustomImageDataset(annotations_file='./data/interim/filepath.csv', transform=transformf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Compute mean and std for RGB channels
    mean, std = get_mean_and_std(dataloader)

    # Save for standardization step later
    dict_mean_std = {'mean': mean.tolist(), 'std': std.tolist()}

    with open("./data/processed/image_mean_std.yaml", "w") as outfile:
        yaml.dump(dict_mean_std, outfile)

    print(dict_mean_std)


