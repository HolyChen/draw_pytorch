#!/usr/bin/python3 
# -*- coding: utf-8 -*-

"""
Paper:
    DRAW: A Recurrent Neural Network For Image Generation
        -- Karol Gregor etc. Google DeepMind 2015
Created by chenzhaomin123
Refined and commented by HolyChen
"""

import torch
import torch.nn as nn
from utility import *
import torch.functional as F

class DrawModel(nn.Module):
    """DRAW Model

    Use LSTM generate image with attention, the model is defined in the paper:
        "DRAW: A Recurrent Neural Network For Image Generation".
    """

    def __init__(self ,T, A, B, z_size, N, dec_size, enc_size):
        """Init the model.

        Args:
            T: generation sequence length
            A: width of image
            B: height of image
            z_size: the number of latent variable
            N: read/write window size
            dec_size: decoder hidden variable size
            enc_size: encoder hidden variable size
        """
        super().__init__()
        self.T = T
        # self.batch_size = batch_size
        self.A = A
        self.B = B
        self.z_size = z_size
        self.N = N
        self.dec_size = dec_size
        self.enc_size = enc_size
        self.cs = [0] * T
        self.logsigmas, self.sigmas, self.mus = [0] * T, [0] * T, [0] * T

        # encoder definition
        self.encoder = nn.LSTMCell(2 * N * N + dec_size, enc_size)
        self.encoder_gru = nn.GRUCell(2 * N * N + dec_size, enc_size)
        # map h^enc_(t) to z, that is sampling from pdf Q(Z_t|h^enc_(t))
        self.mu_linear = nn.Linear(enc_size, z_size)
        # map h^enc_(t) to sigma, that approximate sigma of Q(Z_t|h^enc_(t))
        self.sigma_linear = nn.Linear(dec_size, z_size)

        # decoder definition
        self.decoder = nn.LSTMCell(z_size, dec_size)
        self.decoder_gru = nn.GRUCell(z_size, dec_size)
        # map h^dec_(t) to g_X, g_Y, log(variance), log(stride), log(scalar intensity)
        self.dec_linear = nn.Linear(dec_size, 5)
        self.dec_w_linear = nn.Linear(dec_size, N * N)

        # use sigmoid function to get "error image"
        self.sigmoid = nn.Sigmoid()

    def _compute_mu(self, g, rng, delta):
        """Compute single-dimension mu(mean) of gaussian filters, def by formula (19), (20)

        Args:
            g: X or Y coordinate of the filter
            rng: range of filter indices
            delta: stride of filter
        
        Returns:
            the single-dimension mu of the gaussion filters.
            A torch.autograd.Variable. Size: [self.batch_size, self.N]
        """
        rng_t, delta_t = align(rng, delta)
        tmp = (rng_t - self.N / 2 - 0.5) * delta_t
        tmp_t, g_t = align(tmp, g)
        mu = tmp_t + g_t
        return mu

    def _filterbank_matrices(self, a, mu, sigma2, epsilon=1e-9):
        """Genrate filterbank matrix, call by function filterbank
        
        Args:
            a: x or y coordinates [0, self.A or self.B), size (1, 1, size.A or size.B)
            mu_x: single-dimension mu of gaussian kernel, size (self.batch_size, self.N, 1)
            sigma2: variance of gaussian kernel, size (self.batch_size, self.N, 1)
            epsilon: minimun value to keep out 0-divisor
        """
        t_a, t_mu = align(a, mu)
        temp = t_a - t_mu
        temp, t_sigma = align(temp, sigma2)
        temp = temp / (t_sigma * 2)
        F = torch.exp(-torch.pow(temp, 2))
        # normalize for each kernel
        F = F / (F.sum(2, True).expand_as(F) + epsilon)
        return F

    def _filterbank(self, gx, gy, sigma2, delta):
        """Calculate filter bank defined by formulas (25), (26)
        
        Args: 
            gx: grid center X coodinate, def by formula (22)
            gx: grid center Y coodinate, def by formula (23)
            sigma2: variance of gaussian kernel, def by formula (21)
            delta: stride between the gaussian kernel, def by formula (24)
        """
        rng = Variable(torch.arange(0, self.N).view(1, -1))
        mu_x = self._compute_mu(gx, rng, delta)
        mu_y = self._compute_mu(gy, rng, delta)

        a = Variable(torch.arange(0, self.A).view(1, 1, -1))
        b = Variable(torch.arange(0, self.B).view(1, 1, -1))

        mu_x = mu_x.view(-1, self.N, 1)
        mu_y = mu_y.view(-1, self.N, 1)
        sigma2 = sigma2.view(-1, 1, 1)

        Fx = self._filterbank_matrices(a, mu_x, sigma2) # (25)
        Fy = self._filterbank_matrices(b, mu_y, sigma2) # (26)

        return Fx, Fy

    def _attn_window(self, h_dec):
        """Generate attention window, def in formula (19) ~ (26)

        Args:
            h_dec: decoder hidden variable
        """
        # get the attention window 
        params = self.dec_linear(h_dec)
        # note that, params is a generate from a batch, every row belongs to same batch
        gx_, gy_, log_sigma_2, log_delta, log_gamma = params.split(1, 1)  # (21)

        gx = (self.A + 1) / 2 * (gx_ + 1)    # (22)
        gy = (self.B + 1) / 2 * (gy_ + 1)    # (23)
        delta = (max(self.A, self.B) - 1) / (self.N - 1) * torch.exp(log_delta)  # (24)
        sigma2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self._filterbank(gx, gy, sigma2, delta), gamma # (19), (20, (25, (26)

    def _read(self, x, x_hat, h_dec_prev):
        """Read operation, def in formula (27)

        Args:
            x: source image reshpaed as vector, size [self.batch_size, width * height]
            x_hat: error image reshaped as vector, size [self.batch_size, width * height]
            h_dec_prev: decoder hidden variable last step
        """

        def filter_img(img, Fx, Fy, gamma, A, B, N):
            """Apply gaussion filter grid to image

            Args:
                img: images reshape as vector, size [self.batch_size, width * height]
                Fx: x coorinate filter bank matrix, size [self.batch_size, self.N, A]
                Fy: y coorinate filter bank matrix, size [self.batch_size, self.N, B]
                gamma: scalar intensity, def in formula (21)
                A: width of the image
                B: height of the image
                N: size of filter window
            """
            Fxt = Fx.transpose(2, 1)
            img = img.view(-1, B, A)
            glimpse = Fy.bmm(img.bmm(Fxt))
            glimpse = glimpse.view(-1, N * N)
            return glimpse * gamma.view(-1, 1).expand_as(glimpse)

        (Fx, Fy), gamma = self._attn_window(h_dec_prev)
        x = filter_img(x, Fx, Fy, gamma, self.A, self.B, self.N)
        x_hat = filter_img(x_hat, Fx, Fy, gamma, self.A, self.B, self.N)
        return torch.cat((x, x_hat), 1)

    # correct
    def _write(self, h_dec=0):
        w = self.dec_w_linear(h_dec)
        w = w.view(self.batch_size,self.N,self.N)
        # w = Variable(torch.ones(4,5,5) * 3)
        # self.batch_size = 4
        (Fx,Fy),gamma = self._attn_window(h_dec)
        Fyt = Fy.transpose(2,1)
        # wr = matmul(Fyt,matmul(w,Fx))
        wr = Fyt.bmm(w.bmm(Fx))
        wr = wr.view(self.batch_size,self.A*self.B)
        return wr / gamma.view(-1,1).expand_as(wr)

    def _normalSample(self):
        """Get random white noise matrix to be applied to sigma
        """
        return Variable(torch.randn(self.batch_size, self.z_size))

    def _sampleQ(self, h_enc):
        """Sample from distribution Q(Z_t|h^enc_(t))
        
        Args:
            h_enc: encoder hidden variable this step
        """
        rand_offset = self._normalSample()
        mu = self.mu_linear(h_enc)           # (1)
        log_sigma = self.sigma_linear(h_enc) # (2)
        sigma = torch.exp(log_sigma)

        return (mu + sigma * rand_offset), mu, log_sigma, sigma

    def loss(self, x):
        self.forward(x)
        criterion = nn.BCELoss()
        x_recons = self.sigmoid(self.cs[-1])
        Lx = criterion(x_recons, x) * self.A * self.B
        Lz = 0
        kl_terms = [0] * T
        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]
            # Lz += (0.5 * (mu_2 + sigma_2 - 2 * logsigma))    # 11
            kl_terms[t] = 0.5 * torch.sum(mu_2 + sigma_2 - 2 * logsigma, 1) - self.T * 0.5 # (11)
            Lz += kl_terms[t]
        # Lz -= self.T / 2
        Lz = torch.mean(Lz)    # We use minibatch trainning, so the error should be blend
        loss = Lz + Lx    # 12
        return loss

    def forward(self, x):
        self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size))
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))

        enc_state = Variable(torch.zeros(self.batch_size, self.enc_size))
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))

        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)) if t == 0 else self.cs[t - 1]
            x_hat = x - self.sigmoid(c_prev) # (3)
            r_t = self._read(x,x_hat, h_dec_prev)
            h_enc_prev, enc_state = self.encoder(torch.cat((r_t, h_dec_prev), 1), (h_enc_prev, enc_state))
            # h_enc = self.encoder_gru(torch.cat((r_t,h_dec_prev),1),h_enc_prev)
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self._sampleQ(h_enc_prev)
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            # h_dec = self.decoder_gru(z, h_dec_prev)
            self.cs[t] = c_prev + self._write(h_dec)
            h_dec_prev = h_dec

    def generate(self, batch_size=64):
        self.batch_size = batch_size
        h_dec_prev = Variable(torch.zeros(self.batch_size,self.dec_size),volatile = True)
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size),volatile = True)

        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)) if t == 0 else self.cs[t - 1]
            # we assume z follow standard normal distribution
            z = self._normalSample()
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self._write(h_dec)
            h_dec_prev = h_dec
        imgs = []
        for img in self.cs:
            imgs.append(self.sigmoid(img).cpu().data.numpy())
        return imgs




# model = DrawModel(10,5,5,10,5,128,128)
# x = Variable(torch.ones(4,25))
# x_hat = Variable(torch.ones(4,25)*2)
# r = model.write()
# print r
# g = Variable(torch.ones(4,1))
# delta = Variable(torch.ones(4,1)  * 3)
# sigma = Variable(torch.ones(4,1))
# rng = Variable(torch.arange(0,5).view(1,-1))
# mu_x = model.compute_mu(g,rng,delta)
# a = Variable(torch.arange(0,5).view(1,1,-1))
# mu_x = mu_x.view(-1,5,1)
# sigma = sigma.view(-1,1,1)
# F = model.filterbank_matrices(a,mu_x,sigma)
# print F
# def test_normalSample():
#     print model.normalSample()
#
# def test_write():
#     h_dec = Variable(torch.zeros(8,128))
#     model.write(h_dec)
#
# def test_read():
#     x = Variable(torch.zeros(8,28*28))
#     x_hat = Variable((torch.zeros(8,28*28)))
#     h_dec = Variable(torch.zeros(8, 128))
#     model.read(x,x_hat,h_dec)
#
# def test_loss():
#     x = Variable(torch.zeros(8,28*28))
#     loss = model.loss(x)
#     print loss

