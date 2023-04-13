import random
import torch
from torch.utils.data import Dataset


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

class DiscPool(Dataset):
    """This class implements a buffer that stores the previous discriminator map for each image in the dataset.

    This buffer enables us to recall the outputs of the discriminator in the previous epoch
    """

    def __init__(self, opt, device, isTrain=True, disc_out_size=30):
        """Initialize the DiscPool class

        Parameters:
            opt: stores all the experiment flags; needs to be a subclass of BaseOptions
            device: the device used
            isTrain: whether this class is instanced during the train or test phase
            disc_out_size: the size of the ouput tensor of the discriminator
        """
        from data import create_dataset
        self.dataset_len = dataset_len = len(create_dataset(opt))

        if isTrain:
            # Initially the discriminator doesn't know real/fake because is not trained yet
            self.disc_out = torch.rand((dataset_len, 1, disc_out_size, disc_out_size), dtype=torch.float32)
        else:
            # At the end, the discriminator is expected to be near its maximum entropy state (D_i = 1/2)
            self.disc_out = 0.5 + 0.001 * torch.randn((dataset_len, 1, disc_out_size, disc_out_size), dtype=torch.float32)
        
        self.disc_out = self.disc_out.to(device)
    
    def __getitem__(self, _):
        raise NotImplementedError('DiscPool does not support this operation')

    def __len__(self):
        return self.dataset_len
    
    def query(self, img_idx):
        """Return the last discriminator map from the pool, corresponding to given image indices.

        Parameters:
            img_idx: indices of the images that the discriminator just processed

        Returns discriminator map from the buffer.
        """
        return self.disc_out[img_idx]
    
    def insert(self, disc_out, img_idx):
        """Insert the last discriminator map in the pool, corresponding to given image index.

        Parameters:
            disc_out: output from the discriminator in the backward pass of generator
            img_idx: indices of the images that the discriminator just processed
        """
        self.disc_out[img_idx] = disc_out