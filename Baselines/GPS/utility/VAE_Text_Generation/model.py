# -*- coding: utf-8 -*-
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from utility.VAE_Text_Generation.helper_functions import get_cuda  # Ensure get_cuda is defined elsewhere

class Highway(nn.Module):
    def __init__(self, opt):
        super(Highway, self).__init__()
        self.n_layers = opt.n_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(opt.n_embed, opt.n_embed) for _ in range(self.n_layers)])
        self.linear = nn.ModuleList([nn.Linear(opt.n_embed, opt.n_embed) for _ in range(self.n_layers)])
        self.gate = nn.ModuleList([nn.Linear(opt.n_embed, opt.n_embed) for _ in range(self.n_layers)])

    def forward(self, x):
        for layer in range(self.n_layers):
            gate = torch.sigmoid(self.gate[layer](x))  # Use torch.sigmoid instead of F.sigmoid
            non_linear = F.relu(self.non_linear[layer](x))
            linear = self.linear[layer](x)
            x = gate * non_linear + (1 - gate) * linear  # Combine linear and non-linear information

        return x

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.highway = Highway(opt)
        self.n_hidden_E = opt.n_hidden_E
        self.n_layers_E = opt.n_layers_E
        self.lstm = nn.LSTM(input_size=opt.n_embed, hidden_size=opt.n_hidden_E, 
                            num_layers=opt.n_layers_E, batch_first=True, bidirectional=True)

    def init_hidden(self, batch_size):
        # Initialize hidden state for LSTM with zero tensors
        h_0 = T.zeros(2 * self.n_layers_E, batch_size, self.n_hidden_E)
        c_0 = T.zeros(2 * self.n_layers_E, batch_size, self.n_hidden_E)
        self.hidden = (get_cuda(h_0), get_cuda(c_0))

    def forward(self, x):
        batch_size, n_seq, n_embed = x.size()
        x = self.highway(x)  # Apply the highway network to the input
        self.init_hidden(batch_size)
        _, (self.hidden, _) = self.lstm(x, self.hidden)  # Get the last hidden state of the LSTM
        self.hidden = self.hidden.view(self.n_layers_E, 2, batch_size, self.n_hidden_E)
        self.hidden = self.hidden[-1]  # Get the hidden state from the final LSTM layer
        e_hidden = T.cat(list(self.hidden), dim=1)  # Concatenate hidden states from both directions
        return e_hidden

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.n_hidden_G = opt.n_hidden_G
        self.n_layers_G = opt.n_layers_G
        self.n_z = opt.n_z
        self.lstm = nn.LSTM(input_size=opt.n_embed + opt.n_z, hidden_size=opt.n_hidden_G, 
                            num_layers=opt.n_layers_G, batch_first=True)
        self.fc = nn.Linear(opt.n_hidden_G, opt.n_vocab)  # Linear layer for output

    def init_hidden(self, batch_size):
        # Initialize hidden state for the generator LSTM
        h_0 = T.zeros(self.n_layers_G, batch_size, self.n_hidden_G)
        c_0 = T.zeros(self.n_layers_G, batch_size, self.n_hidden_G)
        self.hidden = (get_cuda(h_0), get_cuda(c_0))

    def forward(self, x, z, g_hidden=None):
        batch_size, n_seq, n_embed = x.size()
        z = T.cat([z] * n_seq, 1).view(batch_size, n_seq, self.n_z)  # Repeat z across the sequence
        x = T.cat([x, z], dim=2)  # Concatenate z with x at each time step

        if g_hidden is None:  # If testing, initialize hidden state
            self.init_hidden(batch_size)
        else:  # If training, use given hidden state
            self.hidden = g_hidden

        # Generate output using LSTM and pass through a fully connected layer
        output, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(output)  # Convert LSTM output to vocab size logits

        return output, self.hidden  # Return the output logits and the hidden state

class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(opt.n_vocab, opt.n_embed)
        self.encoder = Encoder(opt)
        self.hidden_to_mu = nn.Linear(2 * opt.n_hidden_E, opt.n_z)
        self.hidden_to_logvar = nn.Linear(2 * opt.n_hidden_G, opt.n_z)
        self.generator = Generator(opt)
        self.n_z = opt.n_z

    def forward(self, x, G_inp, z=None, G_hidden=None):
        if z is None:  # If testing, sample z from noise
            batch_size, n_seq = x.size()
            x = self.embedding(x)  # Get embeddings for the input
            E_hidden = self.encoder(x)  # Get encoder's hidden states
            mu = self.hidden_to_mu(E_hidden)  # Get the mean for the latent space
            logvar = self.hidden_to_logvar(E_hidden)  # Get log variance
            z = get_cuda(T.randn([batch_size, self.n_z]))  # Sample from a normal distribution
            z = mu + z * T.exp(0.5 * logvar)  # Reparameterization trick: z = μ + ε * σ
            kld = -0.5 * T.sum(logvar - mu.pow(2) - logvar.exp() + 1, 1).mean()  # Compute KL divergence
        else:
            kld = None  # If training with text, no KL divergence

        G_inp = self.embedding(G_inp)  # Get embeddings for generator input

        logit, G_hidden = self.generator(G_inp, z, G_hidden)  # Generate text using the generator
        return logit, G_hidden, kld  # Return logits, hidden state, and KL divergence loss

