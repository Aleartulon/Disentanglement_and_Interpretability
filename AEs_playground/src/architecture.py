import numpy as np
import torch as tc
import torch.nn.functional as F
from torch import nn

class Convolutional_Encoder_AE(nn.Module):
    def __init__(self, dim_input, kernel, filters, stride, input_dfnn, output_dfnn, last_activation):
        """Initialization parameters of the Encoder

        Args:
            dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
            kernel (list): list of the kernel sizes of the encoder per layer
            filters (list): list of the number of filters of the encoder per layer
            stride (list): list of the stride of the encoder per layer
            input_dfnn (int): dimension of the (flattened) vector which is output of the last convolutional layer and input to the linear layer which maps it into the reduced representation
            output_dfnn (int): dimension of the latent space
            last_activation (bool) : if true, after the final linear layer an activation function is used
        """        
        super().__init__()
        
        self.channels = np.concatenate(([dim_input[0]], filters))
        self.size_kernel = [(k - 1) // 2 for k in kernel]  # Adjust padding based on kernel size

        # Activation function (use only one for now)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.activation = self.gelu
        self.tanh = nn.Tanh()

        # Convolutional layers and BatchNorm layers
        self.convolutionals = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.len_kernel = len(kernel)
        self.last_activation = last_activation

        if dim_input[1] == 1:
            for i in range(self.len_kernel):
                if i  == 0 or i == 2:
                    self.convolutionals.append(nn.Conv1d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
                    #self.norm_layers.append(nn.BatchNorm1d(filters[i]))
                else:
                    self.convolutionals.append(nn.Conv1d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))

                nn.init.kaiming_uniform_(self.convolutionals[i].weight)  # Xavier initialization

        elif dim_input[1] == 2:
            for i in range(self.len_kernel):
                if i == 0 or i ==1:
                    self.convolutionals.append(nn.Conv2d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
                    #self.norm_layers.append(nn.BatchNorm2d(filters[i]))
                else:
                    self.convolutionals.append(nn.Conv2d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
                    
                nn.init.kaiming_uniform_(self.convolutionals[i].weight)  # Xavier initialization

        # Dense fully-connected layer
        self.dfnn = nn.Linear(input_dfnn, output_dfnn, bias=True)
        nn.init.kaiming_uniform_(self.dfnn.weight)  # Xavier initialization

    def forward(self, x):
        """Forward pass of the Encoder

        Args:
            x (torch.Tensor): input of the Encoder, dimension is [B * T, C, dim_x_1, dim_x_2, ...], where B is batch_size, T is the length of the full time series, C is the number of channels of the 
            solution field, dim_x_1, dim_x_2, ... are the dimensions of the first spatial dimension, second spatial dimension, etc
        Returns:
            torch.Tensor: returns a tensor of size [B, output_dfnn], where B is the batch_size and output_dfnn is the dimension of the latent space
        """        
        for i, conv_layer in enumerate(self.convolutionals):
            x = conv_layer(x)
            x = self.activation(x)
        x = tc.flatten(x, 1)  # Flatten across the batch dimension
        x = self.dfnn(x)
        if self.last_activation:
            x = self.activation(x)
        return x
    
class Convolutional_Decoder_AE(nn.Module):
    def __init__(self , dim_input, kernel, filters, stride, input_dfnn, output_dfnn, final_reduction, initial_activation, number_channels_input_cnns_deco):
        """Parameters for the initialization of the Decoder

        Args:
            dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
            kernel (list): list of the kernel sizes of the decoder per layer
            filters (list): list of the number of filters of the decoder per layer
            stride (list): list of the stride of the decoder per layer
            input_dfnn (int): dimension of the vector which is input of the initial linear layer of the decoder, i.e., the latent dimension. 
            output_dfnn (int): dimension of the (flattened) vector which is output of the last convolutional layer of the encoder. It is chosen to be the same to make the decoder symmetric to the encoder
            final_reduction (int): final length of each spatial dimension after repeated reduction (halving) operated by the encoder. Used here to make the decoder symmetric to the encoder
            initial_activation (bool) : if true, after the initial linear layer an activation function is used
        """        
        super().__init__()
        self.inputs_cnns = np.concatenate(([number_channels_input_cnns_deco],filters))
        self.channels = np.concatenate((filters,[dim_input[0]]))
        self.transposed_convolutionals = nn.ModuleList([])
        self.size_kernel = int((kernel[0]-1)/2)
        self.dim_input = dim_input[1]
        self.final_reduction = final_reduction

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.activation = self.gelu
        self.initial_activation = initial_activation

        self.dfnn = nn.Linear(input_dfnn, output_dfnn)
        nn.init.kaiming_uniform_(self.dfnn.weight)

        if self.dim_input == 1:
            for i in range(len(kernel)):
                self.transposed_convolutionals.extend([nn.ConvTranspose1d(self.inputs_cnns[i],self.channels[i],kernel[i],stride[i],padding=self.size_kernel, output_padding = 0)])
                nn.init.kaiming_uniform_(self.transposed_convolutionals[i].weight)

        elif self.dim_input == 2:
            for i in range(len(kernel)):
                self.transposed_convolutionals.extend([nn.ConvTranspose2d(self.inputs_cnns[i],self.channels[i],kernel[i],stride[i],padding=self.size_kernel, output_padding = 0)])
                nn.init.kaiming_uniform_(self.transposed_convolutionals[i].weight)


    def forward(self,x):
        
        """Forward pass of the decoder

        Args:
            x (torch.tensor): a tensor of latent vectors of dimension [B*T, latent_dim], where B is batch size, T is the len of the time series and latent dim is the dimension of the latent space

        Returns:
            torch.tensor: a tensor of size [B * T, C, dim_x_1, dim_x_2, ...], where B is batch_size, T is the length of the full time series, C is the number of channels of the 
            predicted solution field, dim_x_1, dim_x_2, ... are the dimensions of the first spatial dimension, second spatial dimension, etc
        """        

        x = self.dfnn(x)
        if self.initial_activation:
            x = self.activation(x) #do not use this if relu in encoder and decoder !!!!!!!!

        if self.dim_input == 1:
            x = x.view(x.size()[0], self.inputs_cnns[0], self.final_reduction)
        elif self.dim_input == 2:
            x = x.view(x.size()[0], self.inputs_cnns[0], self.final_reduction, self.final_reduction)
    
        length = len(self.transposed_convolutionals)
        for i in range(length-1):
            x = self.transposed_convolutionals[i](x)
            x = self.activation(x)
        x = self.transposed_convolutionals[-1](x)
        return x
    
class Convolutional_Encoder_VAE(nn.Module):

    def __init__(self, dim_input, kernel, filters, stride, input_dfnn, output_dfnn, last_activation):
        """Initialization parameters of the Encoder

        Args:
            dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
            kernel (list): list of the kernel sizes of the encoder per layer
            filters (list): list of the number of filters of the encoder per layer
            stride (list): list of the stride of the encoder per layer
            input_dfnn (int): dimension of the (flattened) vector which is output of the last convolutional layer and input to the linear layer which maps it into the reduced representation
            output_dfnn (int): dimension of the latent space
            last_activation (bool) : if true, after the final linear layer an activation function is used
        """        
        super().__init__()
        
        self.channels = np.concatenate(([dim_input[0]], filters))
        self.size_kernel = [(k - 1) // 2 for k in kernel]  # Adjust padding based on kernel size

        # Activation function (use only one for now)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.activation = self.gelu
        self.tanh = nn.Tanh()

        # Convolutional layers and BatchNorm layers
        self.convolutionals = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.len_kernel = len(kernel)
        self.last_activation = last_activation

        if dim_input[1] == 1:
            for i in range(self.len_kernel):
                if i  == 0 or i == 2:
                    self.convolutionals.append(nn.Conv1d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
                    #self.norm_layers.append(nn.BatchNorm1d(filters[i]))
                else:
                    self.convolutionals.append(nn.Conv1d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))

                nn.init.kaiming_uniform_(self.convolutionals[i].weight)  # Xavier initialization

        elif dim_input[1] == 2:
            for i in range(self.len_kernel):
                if i == 0 or i ==1:
                    self.convolutionals.append(nn.Conv2d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
                    #self.norm_layers.append(nn.BatchNorm2d(filters[i]))
                else:
                    self.convolutionals.append(nn.Conv2d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
                    
                nn.init.kaiming_uniform_(self.convolutionals[i].weight)  # Xavier initialization

        # Dense fully-connected layer
        self.means = nn.Linear(input_dfnn, output_dfnn, bias=True)
        self.log_variances = nn.Linear(input_dfnn, output_dfnn, bias=True)
        nn.init.kaiming_uniform_(self.means.weight)  # Xavier initialization
        nn.init.kaiming_uniform_(self.log_variances.weight)
    
    def reparametrization(self, means, log_variances):
        random = tc.randn_like(means)
        sampled_vector = means + random * tc.exp(0.5 * log_variances)
        return sampled_vector

    def forward(self, x):
        """Forward pass of the Encoder

        Args:
            x (torch.Tensor): input of the Encoder, dimension is [B * T, C, dim_x_1, dim_x_2, ...], where B is batch_size, T is the length of the full time series, C is the number of channels of the 
            solution field, dim_x_1, dim_x_2, ... are the dimensions of the first spatial dimension, second spatial dimension, etc
        Returns:
            torch.Tensor: returns a tensor of size [B, output_dfnn], where B is the batch_size and output_dfnn is the dimension of the latent space
        """        
        for i, conv_layer in enumerate(self.convolutionals):
            x = conv_layer(x)
            x = self.activation(x)
        x = tc.flatten(x, 2)
        means = self.means(x[:,0,:])
        log_variances = self.log_variances(x[:,1,:])
        if self.last_activation:
            means = self.activation(means)
            log_variances = self.activation(log_variances)

        sampled_vector = self.reparametrization(means, log_variances)
        return sampled_vector, means, log_variances

class Convolutional_Decoder_VAE(nn.Module):
    def __init__(self , dim_input, kernel, filters, stride, input_dfnn, output_dfnn, final_reduction, initial_activation, number_channels_input_cnns_deco):
        """Parameters for the initialization of the Decoder

        Args:
            dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
            kernel (list): list of the kernel sizes of the decoder per layer
            filters (list): list of the number of filters of the decoder per layer
            stride (list): list of the stride of the decoder per layer
            input_dfnn (int): dimension of the vector which is input of the initial linear layer of the decoder, i.e., the latent dimension. 
            output_dfnn (int): dimension of the (flattened) vector which is output of the last convolutional layer of the encoder. It is chosen to be the same to make the decoder symmetric to the encoder
            final_reduction (int): final length of each spatial dimension after repeated reduction (halving) operated by the encoder. Used here to make the decoder symmetric to the encoder
            initial_activation (bool) : if true, after the initial linear layer an activation function is used
        """        
        super().__init__()
        self.inputs_cnns = np.concatenate(([number_channels_input_cnns_deco], filters))
        self.channels = np.concatenate((filters,[dim_input[0]]))
        self.transposed_convolutionals = nn.ModuleList([])
        self.size_kernel = int((kernel[0]-1)/2)
        self.dim_input = dim_input[1]
        self.final_reduction = final_reduction

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.activation = self.gelu
        self.initial_activation = initial_activation

        #self.log_variances_dfnn = nn.Linear(input_dfnn, dim_input[0]) #get the logsigma for the whole field (sigmas if multiple fields)
        self.dfnn = nn.Linear(input_dfnn, output_dfnn)
        nn.init.kaiming_uniform_(self.dfnn.weight)
        self.log_variances = tc.nn.Parameter(tc.tensor(0.0)) 

        if self.dim_input == 1:
            for i in range(len(kernel)):
                self.transposed_convolutionals.extend([nn.ConvTranspose1d(self.inputs_cnns[i],self.channels[i],kernel[i],stride[i],padding=self.size_kernel, output_padding = 0)])
                nn.init.kaiming_uniform_(self.transposed_convolutionals[i].weight)

        elif self.dim_input == 2:
            for i in range(len(kernel)):
                self.transposed_convolutionals.extend([nn.ConvTranspose2d(self.inputs_cnns[i],self.channels[i],kernel[i],stride[i],padding=self.size_kernel, output_padding = 0)])
                nn.init.kaiming_uniform_(self.transposed_convolutionals[i].weight)

    def forward(self,x):
        
        """Forward pass of the decoder

        Args:
            x (torch.tensor): a tensor of latent vectors of dimension [B*T, latent_dim], where B is batch size, T is the len of the time series and latent dim is the dimension of the latent space

        Returns:
            torch.tensor: a tensor of size [B * T, C, dim_x_1, dim_x_2, ...], where B is batch_size, T is the length of the full time series, C is the number of channels of the 
            predicted solution field, dim_x_1, dim_x_2, ... are the dimensions of the first spatial dimension, second spatial dimension, etc
        """        
        #log_variances = self.log_variances_dfnn(x)
        x = self.dfnn(x)
        if self.initial_activation:
            x = self.activation(x) #do not use this if relu in encoder and decoder !!!!!!!!

        if self.dim_input == 1:
            x = x.view(x.size()[0], self.inputs_cnns[0], self.final_reduction)
        elif self.dim_input == 2:
            x = x.view(x.size()[0], self.inputs_cnns[0], self.final_reduction, self.final_reduction)
    
        length = len(self.transposed_convolutionals)
        for i in range(length-1):
            x = self.transposed_convolutionals[i](x)
            x = self.activation(x)
        x = self.transposed_convolutionals[-1](x)
        return x[:,0,...].unsqueeze(1), self.log_variances # means, log_variances


