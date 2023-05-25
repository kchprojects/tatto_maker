from src.gan_playground.Discriminator import Discriminator
from src.gan_playground.Generator import Generator
import torch
import torch.nn.functional as F
from numpy.random import shuffle
import cv2
import numpy as np


class GAN:

    def __init__(self,input_size,channels,latent_size,g_depth=3):
        #TODO: generalize size
        self.latent_size = latent_size
        self.generator = Generator(input_size,channels,g_depth,latent_size)
        self.discriminator = Discriminator(input_size,channels)

    def train_discriminator(self,real_images, device, batch_size, opt):
        # Clear discriminator gradients
        opt.zero_grad()

        # Pass real images through discriminator
        real_preds = self.discriminator(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=device)
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()
        
        # Generate fake images
        latent = torch.randn(batch_size, self.latent_size, 1, 1, device=device)
        fake_images = self.generator(latent)

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
        fake_preds = self.discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        opt.step()

        return loss.item(), real_score, fake_score
    def gen_imgs(self, batch_size,device):
        latent = torch.randn(batch_size, self.latent_size, 1, 1, device=device)
        return self.generator(latent)

    def train_generator(self,batch_size, device,opt):
        # Clear generator gradients
        opt.zero_grad()
        
        # Generate fake images
        latent = torch.randn(batch_size, self.latent_size, 1, 1, device=device)
        fake_images = self.generator(latent)
        
        # Try to fool the discriminator
        preds = self.discriminator(fake_images)
        targets = torch.ones(batch_size, 1, device=device)
        loss = F.binary_cross_entropy(preds, targets)
        
        # Update generator weights
        loss.backward()
        opt.step()
        return loss.item()
    
    def fit(self,train_images,batch_size, device,epochs, lr, tag):
        torch.cuda.empty_cache()
        cv2.namedWindow("gen",cv2.WINDOW_NORMAL)
        
        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []
        
        # Create optimizers
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        batch_indices = np.asarray(range(batch_size*len(train_images)//batch_size))

        for epoch in range(epochs):
            shuffle(batch_indices)
            for batch_id in range(len(batch_indices)//batch_size):
                real_images = train_images[batch_indices[batch_id*batch_size:(batch_id+1)*batch_size],:,:,:]
                # Train discriminator
                loss_d, real_score, fake_score = self.train_discriminator(real_images,device,batch_size,opt_d)
                # Train generator
                loss_g = self.train_generator(batch_size,device,opt_g)
                
            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)
            
            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        
            # Save generated images
            mosaic = []
            x_size = 5
            y_size = 5
            genimgs = self.gen_imgs(x_size*y_size,device).detach().numpy()
            for j in range(y_size):
                images = []
                for i in range(x_size):
                    new_img = genimgs[x_size*j+i,:,:,:]
                    new_img = np.moveaxis(new_img, -2, 0)
                    new_img = np.moveaxis(new_img, -1, 1)
                    images.append(new_img)
                mosaic.append(np.hstack(images))
            #denorm
            img = np.vstack(mosaic)*0.5 + 0.5                
            cv2.imshow("gen", img)
            cv2.waitKey(5)
            cv2.imwrite(f"{tag}-{epoch}.png",img*255)

        
        return losses_g, losses_d, real_scores, fake_scores