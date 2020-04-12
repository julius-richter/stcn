import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2, 
                 activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        
        self.resample = (nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, 1)) 
                         if in_channels != out_channels else None)
        self.padding = nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0)
        self.convolution = nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, 
                                                 kernel_size, dilation=dilation))
        self.activation = activation
        self.dropout = nn.Dropout(dropout)     
        
        self.init_weights()
        
    def init_weights(self):
        self.convolution.weight.data.normal_(0, 0.01)
        if self.resample is not None:
            self.resample.weight.data.normal_(0, 0.01)
        
    def forward(self, x):
        x = x if self.resample is None else self.resample(x)
        y = self.dropout(self.activation(self.convolution(self.padding(x))))
        return self.activation(x + y)


class TCN(nn.Module):
    def __init__(self, channels, kernel_size=2, dropout=0.2, activation=nn.ReLU()):
        super(TCN, self).__init__()
        
        self.channels = channels
        
        self.layers = nn.Sequential(*[ResidualBlock(channels[i], channels[i+1],
            kernel_size, 2**i, dropout, activation) for i in range(len(channels)-1)]) 
    
    def representations(self, x):
        # bottom-up
        d = [x]
        for i in range(len(self.channels)-1):
            d += [self.layers[i](d[-1])]
        return d[1:]
    
    def forward(self, x):
        return self.layers(x)


class ObservationModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(ObservationModel, self).__init__()
        
        self.num_hidden_layers = num_hidden_layers
        
        self.dec_in = nn.Conv1d(input_dim, hidden_dim, 1)
        self.dec_hidden = nn.Sequential(*[nn.Conv1d(hidden_dim, hidden_dim, 1) 
                                         for _ in range(num_hidden_layers)])
        self.dec_out_1 = nn.Conv1d(hidden_dim, output_dim, 1)
        self.dec_out_2 = nn.Conv1d(hidden_dim, output_dim, 1)

    def decode(self, z):
        h = torch.tanh(self.dec_in(z))
        for i in range(self.num_hidden_layers):
            h = torch.tanh(self.dec_hidden[i](h))     
        return self.dec_out_1(h), self.dec_out_2(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, z):
        z = torch.cat(z, dim=1)
        mu, logvar = self.decode(z)
        x = self.reparameterize(mu, logvar)
        return x


class LatentLayer(nn.Module):
    def __init__(self, tcn_dim, latent_dim_in, latent_dim_out, hidden_dim, num_hidden_layers):
        super(LatentLayer, self).__init__()
        
        self.num_hidden_layers = num_hidden_layers
        
        self.enc_in = nn.Conv1d(tcn_dim+latent_dim_in, hidden_dim, 1)
        self.enc_hidden = nn.Sequential(*[nn.Conv1d(hidden_dim, hidden_dim, 1) 
                                         for _ in range(num_hidden_layers)])
        self.enc_out_1 = nn.Conv1d(hidden_dim, latent_dim_out, 1)
        self.enc_out_2 = nn.Conv1d(hidden_dim, latent_dim_out, 1)

    def forward(self, x):
        h = torch.tanh(self.enc_in(x))
        for i in range(self.num_hidden_layers):
            h = torch.tanh(self.enc_hidden[i](h))     
        return self.enc_out_1(h), self.enc_out_2(h)


class GenerativeModel(nn.Module):
    def __init__(self, tcn_channels, latent_channels, num_hidden_layers):
        super(GenerativeModel, self).__init__()
        
        self.layers = [LatentLayer(tcn_channels[i], latent_channels[i+1], latent_channels[i], 
                              latent_channels[i], num_hidden_layers) for i in range(len(tcn_channels)-1)]  
        self.layers += [LatentLayer(tcn_channels[-1], 0, latent_channels[-1], latent_channels[-1], 
                                num_hidden_layers)]
        self.layers = nn.ModuleList(self.layers)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
                
    def forward(self, d):
        # top-down
        _mu, _logvar = self.layers[-1](d[-1])
        mu = [_mu]; logvar = [_logvar]
        z = [self.reparameterize(_mu, _logvar)]
        for i in reversed(range(len(self.layers)-1)):
            _mu, _logvar = self.layers[i](torch.cat((d[i], z[-1]), dim=1))
            z += [self.reparameterize(_mu, _logvar)]
            mu = [_mu] + mu
            logvar = [_logvar] + logvar
        return z, mu, logvar


class InferenceModel(nn.Module):
    def __init__(self, tcn_channels, latent_channels, num_hidden_layers):
        super(InferenceModel, self).__init__()
        
        self.layers = [LatentLayer(tcn_channels[i], latent_channels[i+1], latent_channels[i], 
                              latent_channels[i], num_hidden_layers) for i in range(len(tcn_channels)-1)]  
        self.layers += [LatentLayer(tcn_channels[-1], 0, latent_channels[-1], latent_channels[-1], 
                                num_hidden_layers)]
        self.layers = nn.ModuleList(self.layers)   
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, d, mu_p, logvar_p):
        # top-down
        mu_q_hat, logvar_q_hat = self.layers[-1](d[-1])   
        logvar_q = torch.log(1/(torch.pow(torch.exp(logvar_q_hat), -2) 
                                + torch.pow(torch.exp(logvar_p[-1]), -2)))
        mu_q = logvar_q*(mu_q_hat*torch.sqrt(logvar_q_hat) + mu_p[-1]*torch.sqrt(logvar_p[-1]))      
        z = [self.reparameterize(mu_q, logvar_q)]
        for i in reversed(range(len(self.layers)-1)):
            mu_q_hat, logvar_q_hat = self.layers[i](torch.cat((d[i], z[-1]), dim=1))
            logvar_q = torch.log(1/(torch.pow(torch.exp(logvar_q_hat), -2) 
                                + torch.pow(torch.exp(logvar_p[i]), -2)))
            mu_q = logvar_q*(mu_q_hat * torch.sqrt(logvar_q_hat) + mu_p[i] * torch.sqrt(logvar_p[i]))             
            z += [self.reparameterize(mu_q, logvar_q)]        
        return z


class STCN(nn.Module):
    def __init__(self, input_dim, tcn_channels, latent_channels, mode='inference',
                 kernel_size=2, dropout=0.2, activation=nn.ReLU()):
        super(STCN, self).__init__()
        
        self.mode = mode    
        self.tcn = TCN([input_dim]+tcn_channels, kernel_size, dropout, activation)   
        self.generative_model = GenerativeModel(tcn_channels, latent_channels, 
                                                num_hidden_layers=2)
        self.inference_model = InferenceModel(tcn_channels, latent_channels, 
                                                num_hidden_layers=2)    
        self.observation_model = ObservationModel(input_dim=sum(latent_channels), output_dim=input_dim, 
                                                  hidden_dim=256, num_hidden_layers=5)
        
    def generate(self, x):
        d = self.tcn.representations(x) 
        d_shift = [(nn.functional.pad(d[i], pad=(1, 0))[:,:,:-1]) for i in range(len(d))]  
        
        z_p, _, _ = self.generative_model(d_shift)
        x_hat = self.observation_model(z_p)          
        return x_hat
    
    def infer(self, x):
        d = self.tcn.representations(x) 
        d_shift = [(nn.functional.pad(d[i], pad=(1, 0))[:,:,:-1]) for i in range(len(d))]  
        
        z_p, mu_p, logvar_p = self.generative_model(d_shift)
        z_q = self.inference_model(d, mu_p, logvar_p)
        x_hat = self.observation_model(z_q)          
        return x_hat
    
    def forward(self, x):
        if self.mode == 'inference':
            x_hat = self.infer(x)
        elif self.mode == 'generation':
            x_hat = self.generate(x)
        else:
            return None
        return x_hat 
      