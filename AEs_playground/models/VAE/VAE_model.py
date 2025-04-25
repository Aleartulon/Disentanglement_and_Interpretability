from src.data_functions import *
from models.VAE.training_validation_functions import *

class Convolutional_Variational_Encoder(nn.Module):

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

                tc.nn.init.xavier_uniform_(self.convolutionals[i].weight)  # Xavier initialization

        elif dim_input[1] == 2:
            for i in range(self.len_kernel):
                if i == 0 or i ==1:
                    self.convolutionals.append(nn.Conv2d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
                    #self.norm_layers.append(nn.BatchNorm2d(filters[i]))
                else:
                    self.convolutionals.append(nn.Conv2d(self.channels[i], filters[i], kernel[i], stride=stride[i], padding=self.size_kernel[i], padding_mode='replicate', bias=True))
                    
                tc.nn.init.xavier_uniform_(self.convolutionals[i].weight)  # Xavier initialization

        # Dense fully-connected layer
        self.means = nn.Linear(input_dfnn, output_dfnn, bias=True)
        self.log_variances = nn.Linear(input_dfnn, output_dfnn, bias=True)
        tc.nn.init.xavier_uniform_(self.means.weight)  # Xavier initialization
        tc.nn.init.xavier_uniform_(self.log_variances.weight)


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

        sampled_vector = sample_from_gaussian(means, log_variances)
        return sampled_vector, means, log_variances

class Convolutional_Variational_Decoder(nn.Module):
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

        self.log_variances_dfnn = nn.Linear(input_dfnn * 2, dim_input[0]) #get the logsigma for the whole field (sigmas if multiple fields)
        self.dfnn = nn.Linear(input_dfnn, output_dfnn)
        tc.nn.init.xavier_uniform_(self.dfnn.weight)

        if self.dim_input == 1:
            for i in range(len(kernel)):
                self.transposed_convolutionals.extend([nn.ConvTranspose1d(self.inputs_cnns[i],self.channels[i],kernel[i],stride[i],padding=self.size_kernel, output_padding = 0)])
                tc.nn.init.xavier_uniform_(self.transposed_convolutionals[i].weight)

        elif self.dim_input == 2:
            for i in range(len(kernel)):
                self.transposed_convolutionals.extend([nn.ConvTranspose2d(self.inputs_cnns[i],self.channels[i],kernel[i],stride[i],padding=self.size_kernel, output_padding = 0)])
                tc.nn.init.xavier_uniform_(self.transposed_convolutionals[i].weight)

    def forward(self,x, means, log_variances):
        
        """Forward pass of the decoder

        Args:
            x (torch.tensor): a tensor of latent vectors of dimension [B*T, latent_dim], where B is batch size, T is the len of the time series and latent dim is the dimension of the latent space

        Returns:
            torch.tensor: a tensor of size [B * T, C, dim_x_1, dim_x_2, ...], where B is batch_size, T is the length of the full time series, C is the number of channels of the 
            predicted solution field, dim_x_1, dim_x_2, ... are the dimensions of the first spatial dimension, second spatial dimension, etc
        """        
        log_variances_reconstruction = self.log_variances_dfnn(tc.cat((means, log_variances),-1))
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
        #return x[:,0,...].unsqueeze(1), log_variances_reconstruction # means, log_variances
        return x, log_variances_reconstruction # means, log_variances

class Variational_AutoEncoder:
    def __init__(self , global_info, model_info):

        self.dim_input = global_info['dim_input']
        self.side_size = global_info['side_size']
        self.device = global_info['device']
        self.epochs = global_info['epochs']
        self.PATH = global_info['PATH']
        self.checkpoint = global_info['checkpoint']
        self.warmup_lr = global_info['warmup_lr']
        self.clipping = global_info['clipping']
        self.kernel_encoder = model_info['kernel_encoder']
        self.kernel_decoder = model_info['kernel_decoder']
        self.filters_encoder = model_info['filters_encoder']
        self.filters_decoder = model_info['filters_decoder']
        self.stride_encoder = model_info['stride_encoder']
        self.stride_decoder = model_info['stride_decoder']
        self.latent_dimension = model_info['latent_dimension']
        self.activation = model_info['final_and_initial_activation']
        self.number_channels_input_cnns_deco = model_info['number_channels_input_cnns_deco']
        self.validation_loss_coefficients = model_info['validation_loss_coefficients']
        self.training_loss_coefficients = model_info['training_loss_coefficients']
        self.dynamically_increasing_losses = model_info['dynamically_increasing_losses']

        n_halving = len(np.where(np.array(self.stride_encoder)==2)[0]) 
        if self.dim_input[1] == 1:
            final_reduction = int(self.side_size /(2**n_halving))
            input_dfnn_encoder = int(final_reduction * self.filters_encoder[-1] ) #dimension of the vector the final linear layer of the encoder receives
        elif self.dim_input[1] == 2:
            final_reduction = int(self.side_size /(2**n_halving))
            input_dfnn_encoder = int(final_reduction**2 * self.filters_encoder[-1] ) #dimension of the vector the final linear layer of the encoder receives
        self.encoder = Convolutional_Variational_Encoder(self.dim_input, self.kernel_encoder, self.filters_encoder, self.stride_encoder, final_reduction**2, self.latent_dimension, self.activation)
        self.decoder = Convolutional_Variational_Decoder(self.dim_input, self.kernel_decoder, self.filters_decoder, self.stride_decoder, self.latent_dimension, input_dfnn_encoder, final_reduction, self.activation, self.number_channels_input_cnns_deco)

    def send_to_device_and_set_optimizer(self, learning_rate, gamma_scheduler , warmup_time, weight_decay, device):
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

         #define optimizer, the pre scheduler for the warmup of the model and the scheduler

        params_to_optimize = [
            {'params': self.encoder.parameters(), 'weight_decay': weight_decay['encoder']},
            {'params': self.decoder.parameters(), 'weight_decay': weight_decay['decoder']}
        ]

        optim = tc.optim.Adam(params_to_optimize, lr = learning_rate)
        lambda1 = lambda i : i / warmup_time
        pre_scheduler = tc.optim.lr_scheduler.LambdaLR(optim,lambda1) #warm up of the learning rate
        scheduler = tc.optim.lr_scheduler.ExponentialLR(optim, gamma_scheduler)

        return optim, pre_scheduler, scheduler

    def train(self, training_data, validation_data, ma_mi, optim, pre_scheduler, scheduler):
        if not self.checkpoint:

            # create losses file
            os.makedirs(self.PATH+'/losses/',exist_ok=True)
            os.makedirs(self.PATH+'/checkpoint/',exist_ok=True)
            #start the training

            print("------------------TRAINING STARTS------------------")
            early_stopping = 0 
            full_training_count = 1
            loss_value = 1e4
            training_losses = {} #define training losses dictionary
            validation_losses = {} #define training losses dictionary
            max_regularization_coefficient = self.training_loss_coefficients['kl_regularization']

            for key in self.training_loss_coefficients:
                training_losses[key] = np.zeros(self.epochs)

            for key in self.validation_loss_coefficients:
                validation_losses[key] = np.zeros(self.epochs)
            
            for i in range(self.epochs):
                early_stopping += 1

                if early_stopping == 200:
                    print('Training stopped due to early stopping')
                    break

                time1 = time.time()
                if i < self.warmup_lr: #use only AR
                    self.training_loss_coefficients['kl_regularization'] = 0
                    train_losses_data = train_epoch(self.encoder, self.decoder, ma_mi, self.device, optim, training_data, self.training_loss_coefficients, self.dim_input, self.clipping)
                    valid_losses_data = valid_epoch(self.encoder, self.decoder, ma_mi, self.device, validation_data, self.validation_loss_coefficients, self.dim_input)
                    valid_losses_data['total'] = 1e6 # do not save anything during warming up
                else:
                    self.training_loss_coefficients['kl_regularization'] = max_regularization_coefficient* self.dynamically_increasing_losses['kl_regularization_strength'] * full_training_count #increase dynamically the strength of the latent regularization term
                    full_training_count +=1
                    
                    if self.training_loss_coefficients['kl_regularization'] >= max_regularization_coefficient:
                        self.training_loss_coefficients['kl_regularization'] = max_regularization_coefficient
                    
                    train_losses_data = train_epoch(self.encoder, self.decoder, ma_mi, self.device, optim, training_data, self.training_loss_coefficients, self.dim_input, self.clipping)
                    valid_losses_data = valid_epoch(self.encoder, self.decoder, ma_mi, self.device, validation_data,self.validation_loss_coefficients, self.dim_input)

                time2 = time.time()

                if i > self.warmup_lr:
                    scheduler.step()
                else:
                    pre_scheduler.step()
                
                for key in self.training_loss_coefficients:
                    training_losses[key][i] = train_losses_data[key]
                for key in self.validation_loss_coefficients:
                    validation_losses[key][i] = valid_losses_data[key]
    
                with open(self.PATH + "/losses/training_losses.json", "wb") as f:
                    pickle.dump(training_losses, f)
                with open(self.PATH + "/losses/validation_losses.json", "wb") as f:
                    pickle.dump(validation_losses, f)

                print("Epoch: " +str(i)+', ' + str(time2-time1)+ ' s')
                print('Training loss coefficients: ', *[f"{k}: {v}" for k, v in self.training_loss_coefficients.items()], sep=", ")
                print('Validation loss coefficients: ', *[f"{k}: {v}" for k, v in self.validation_loss_coefficients.items()], sep=", ")
                print(' ')
                print('Training losses: ', *[f"{k}: {v}" for k, v in train_losses_data.items()], sep=", ")
                print('Validation losses: ', *[f"{k}: {v}" for k, v in valid_losses_data.items()], sep=", ")
                print('The validation loss has not decreased for ' + str(early_stopping) + ' epochs!')
                
                print('------------------------------------------------------')

                #check if training a noncoupled system and adjust accordingly the validatin losses to be checked for early stopping

                if valid_losses_data['total'] < loss_value: #careful valid loss tot!!
                    loss_value = valid_losses_data['total']
                    print('Models saved!')
                    save_checkpoint(self.encoder, self.decoder, optim, scheduler, i, loss_value ,full_training_count, self.PATH+'/checkpoint/check.pt')
                    early_stopping = 0
        
        else:

            self.encoder, self.decoder, optim, scheduler, start_epoch, loss, full_training_count = load_checkpoint(self.encoder, self.decoder, optim, scheduler, self.PATH+'/checkpoint/check.pt', self.device)
            self.encoder.to(self.device)
            self.decoder.to(self.device) 

            #start the training
            print("------------------TRAINING STARTS------------------")
            loss_value = loss
            early_stopping = 0 
            max_regularization_coefficient = self.training_loss_coefficients['kl_regularization']

            with open(self.PATH + "/losses/training_losses.json", "rb") as f:
                training_losses = pickle.load(f)

            with open(self.PATH + "/losses/validation_losses.json", "rb") as f:
                validation_losses = pickle.load(f)
            
            for i in np.arange(start_epoch+1, self.epochs, 1):
                early_stopping += 1

                if early_stopping == 200:
                    print('Training stopped due to early stopping')
                    break

                time1 = time.time()
                if i < self.warmup_lr: #use only AR
                    self.training_loss_coefficients['kl_regularization'] = 0
                    train_losses_data = train_epoch(self.encoder, self.decoder, ma_mi, self.device, optim, training_data, self.training_loss_coefficients, self.dim_input, self.clipping)
                    valid_losses_data = valid_epoch(self.encoder, self.decoder, ma_mi, self.device, validation_data, self.validation_loss_coefficients, self.dim_input)
                    valid_losses_data['total'] = 1e6 # do not save anything during warming up
                else:
                    self.training_loss_coefficients['kl_regularization'] = max_regularization_coefficient* self.dynamically_increasing_losses['kl_regularization_strength'] * full_training_count #increase dynamically the strength of the latent regularization term
                    full_training_count +=1
                    
                    if self.training_loss_coefficients['kl_regularization'] >= max_regularization_coefficient:
                        self.training_loss_coefficients['kl_regularization'] = max_regularization_coefficient
                    
                    train_losses_data = train_epoch(self.encoder, self.decoder, ma_mi, self.device, optim, training_data, self.training_loss_coefficients, self.dim_input, self.clipping)
                    valid_losses_data = valid_epoch(self.encoder, self.decoder, ma_mi, self.device, validation_data,self.validation_loss_coefficients, self.dim_input)

                time2 = time.time()

                if i > self.warmup_lr:
                    scheduler.step()
                else:
                    pre_scheduler.step()

                for key in self.training_loss_coefficients:
                    training_losses[key][i] = train_losses_data[key]
                    np.save(self.PATH + "/losses/"+str(key) +"_training.npy", training_losses[key])

                for key in self.validation_loss_coefficients:
                    validation_losses[key][i] = valid_losses_data[key]
                    np.save(self.PATH + "/losses/"+str(key) +"_validation.npy", validation_losses[key])

                print("Epoch: " +str(i)+', ' + str(time2-time1)+ ' s')
                print('Training loss coefficients: ', *[f"{k}: {v}" for k, v in self.training_loss_coefficients.items()], sep=", ")
                print('Validation loss coefficients: ', *[f"{k}: {v}" for k, v in self.validation_loss_coefficients.items()], sep=", ")
                print(' ')
                print('Training losses: ', *[f"{k}: {v}" for k, v in train_losses_data.items()], sep=", ")
                print('Validation losses: ', *[f"{k}: {v}" for k, v in valid_losses_data.items()], sep=", ")
                print('The validation loss has not decreased for ' + str(early_stopping) + ' epochs!')
                
                print('------------------------------------------------------')

                #check if training a noncoupled system and adjust accordingly the validatin losses to be checked for early stopping

                if valid_losses_data['total'] < loss_value: #careful valid loss tot!!
                    loss_value = valid_losses_data['total']
                    print('Models saved!')
                    save_checkpoint(self.encoder, self.decoder, optim, scheduler, i, loss_value , self.training_loss_coefficients['kl_regularization'], full_training_count, self.PATH+'/checkpoint/check.pt')
                    early_stopping = 0


            return 0

