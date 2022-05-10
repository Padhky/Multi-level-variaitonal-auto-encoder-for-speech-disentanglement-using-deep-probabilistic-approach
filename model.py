import paderbox as pb
import padertorch as pt
import padercontrib as pc
from paderbox.notebook import *
import paderbox as pb
from padertorch.data.utils import collate_fn

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
import numpy as np

from lazy_dataset.core import DynamicBucket
import random

from torch.distributions import Normal, Bernoulli
from sklearn.manifold import TSNE
from padertorch.modules.normalization import InputNormalization
from torchvision.utils import make_grid
from padertorch.contrib.je.modules.conv import  CNN1d, CNNTranspose1d 

"""Main Model"""    
class ML_VAE(pt.base.Model):
    loss = nn.MSELoss(reduction='sum')
    def __init__(self, style_dim=256, content_dim=32, training=True):
        super().__init__()
        self.training = training
        self.style_dim = style_dim
        self.content_dim = content_dim
        self.features = 64
        self.normalize = InputNormalization(data_format='bct', shape=(None, self.features, None), statistics_axis='bt',
                            independent_axis=None,)
        
        """Common encoder network"""
        self.encoder = CNN1d(in_channels=self.features, out_channels=[16, 32], kernel_size=[3,3], pad_type=2*[None])
        
        """Style network"""
        self.style = CNN1d(in_channels=32, out_channels=[32, 64, 128, 256, 64, self.style_dim*2], kernel_size=[5,5,5,5,5,5],    
                           pad_type=6*[None], stride=2*[1]+1*[2]+3*[1])
        
        """Content_network"""
        self.content = CNN1d(in_channels=32, out_channels=[32, 64, 128, self.content_dim*2], kernel_size=[5,5,5,5],
                             pad_type=4*[None], norm = 'sequence')

        """Decoder network"""
        self.decoder = CNNTranspose1d(in_channels=self.style_dim+self.content_dim, 
                          out_channels=[64, 128, 256, 128, 64,64,self.features], kernel_size=[5,5,5,5,3,2,2], pad_type=7*[None])
    def encode(self, x):
        
        x = self.normalize(x)
        mu_logvar, _ = self.encoder(x)

        """Style"""
        style,_ = self.style(mu_logvar)
        style = style.view(style.size(0),2, -1)
        style_mu = style[:,0,:]
        style_mu = style_mu.view(style.size(0),self.style_dim, -1)
        style_logvar = style[:,1,:]
        style_logvar = style_logvar.view(style.size(0),self.style_dim, -1)
        style_mu = torch.mean(style_mu, dim=2)
        style_logvar = torch.mean(style_logvar, dim=2)

        """Content"""
        content, _ = self.content(mu_logvar)
        content = content.view(content.size(0), 2, -1)
        content_mu = content[:,0,:]
        content_mu = content_mu.view(content.size(0), self.content_dim, -1)
        content_logvar = content[:,1,:] 
        content_logvar = content_logvar.view(content.size(0), self.content_dim, -1)
        content_shape = content_mu

        return style_mu, style_logvar, content_mu, content_logvar, content_shape, x
    
    
    def decode(self, style_dim, content_dim, shape):
        
        """Expanding the dimensions of style and content"""
        feature_shape = shape
        
        """Style dimension"""
        style_dim = torch.unsqueeze(style_dim, dim=2)
        expand = [-1]+[-1]+[feature_shape.size(2)//style_dim.size(2)]
        style = style_dim.expand(*expand)
        
        """Decoder"""
        out = torch.cat((content_dim, style), dim=1)
        out,_ = self.decoder(out)
        
        return out
   
    def forward(self, training):
        
        features = training['features']
        speakers = training['speakers']
        
        """Encoder"""
        style_mu, style_logvar, content_mu, content_logvar, shape, feat = self.encode(features.float()) 

        """Accumulating group evidence"""
        group_mu, group_var, labels = self.accumulate_evidence(style_mu, style_logvar, speakers)

        """Style reaparameterisation"""
        content_latent_space = self.reparameterise(content_mu, content_logvar)

        """Content reaparameterisation"""
        style_latent_space = self.group_wise_reparameterise(group_mu, group_var, speakers)

        """Reconstruct the samples"""
        Recon_image = self.decode(style_latent_space, content_latent_space, shape)

        return dict(speech = Recon_image, 
                    style_mu = group_mu, style_logvar = group_var, 
                    content_mu = content_mu, content_logvar = content_logvar, labels=labels, feature=feat)

    def review(self, training, outputs):
        """KL divergence loss for style latent space"""
        self.style_mu = outputs['style_mu']
        self.style_logvar = outputs['style_logvar']
        style_kl_loss = 0.5 * torch.sum(self.style_logvar[:].exp() - self.style_logvar[:] - 1 + self.style_mu[:].pow(2), dim=1)
        style_kl_loss = torch.sum(style_kl_loss)
        
        """KL divergence loss for content latent space""" 
        self.content_mu = outputs['content_mu']
        self.content_logvar = outputs['content_logvar'] 
        content_kl_loss = 0.5 * torch.sum(self.content_logvar.exp() - self.content_logvar - 1 + self.content_mu.pow(2), dim=1) 
        content_kl_loss = torch.sum(content_kl_loss)

        """MSE Loss"""
        mse = self.loss(outputs["speech"], outputs["feature"])

        """ELBO"""
        loss = (mse + style_kl_loss + content_kl_loss)/len(outputs['labels'])

        review = dict(
                      losses = dict(ELBO=loss, 
                                    MSE = mse/len(outputs['labels']), 
                                    style_loss = style_kl_loss/len(outputs['labels']), 
                                    content_loss= content_kl_loss/len(outputs['labels'])),  
                      scalars = dict(), 
                      images = dict(
                                    features = torch.unsqueeze(outputs['feature'],dim=1)[:6], 
                                    output = torch.unsqueeze(outputs['speech'],dim=1)[:6])  
                                     ) 
        return review

    def modify_summary(self, summary):
        
        for key, image in summary['images'].items():
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        summary = super().modify_summary(summary)
        
        return summary

    """Function used to accumulate evidence and group_wise reparameterise"""
    
    def accumulate_evidence(self, mu,logvar, speakers):
        content_mu = []
        content_logvar = []
        batch_labels = []
        """Convert the batch of tensor digits to list"""
        group_labels = speakers.tolist()
        """Number of groups in the batch"""
        labels = set(speakers.tolist())
        l = 0
        for label in labels:
            group = speakers.eq(label).nonzero().squeeze()
            group_length = []
            for i in range(len(group_labels)):
                """Sort the batch group in the ordered list"""
                if l==0:
                    batch_labels.append(group_labels[i])
                    l=1   
                """For if condition we calculate the group_length"""
                if label == group_labels[i]:
                    group_length.append(group_labels[i])
            if len(group_length) > 0:
                """Calculating the group_mu and group variane"""
                group_var = -logvar[group,:]
                inv_group_var = torch.exp(group_var)
                group_mu = mu[group,:]*inv_group_var
            if len(group_length) > 1:
                """Sum the group_mu and group_variance"""
                group_mu = torch.sum(group_mu, dim=0)
                inv_group_var = torch.logsumexp(inv_group_var, dim=0)
            content_mu.append(group_mu)       
            content_logvar.append(inv_group_var) 
        content_mu = torch.stack(content_mu, dim=0)
        content_logvar = torch.stack(content_logvar, dim=0)
        """Calculate the sum of the content_mu and Content_varaiance"""
        content_logvar = - content_logvar
        content_mu = content_mu * torch.exp(content_logvar)

        return content_mu, content_logvar, labels

    def group_wise_reparameterise(self, content_mu, content_logvar, speakers):

        std = content_logvar.mul(0.5).exp_()
        eps = {}
        content_latent_space = []
        labels = set(speakers.tolist())
        group_labels = speakers.tolist()
        for label in labels:
            if self.training:
                eps[label] = torch.FloatTensor(1, std.size(1)).normal_(mean=0, std=0.1).to(std.device)
            else:
                eps[label] = torch.zeros((1, std.size(1))).float().to(std.device)                   
        for i in group_labels:
            for j,label in enumerate(labels):
                if label == i:
                    reparameterise = std[j]*eps[label] + content_mu[j]
                    content_latent_space.append(reparameterise)
        content_latent_space = torch.cat(content_latent_space, dim=0)
        
        return content_latent_space

    def reparameterise(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.zeros_like(std).normal_()
        return eps.mul(std).add_(mu)
