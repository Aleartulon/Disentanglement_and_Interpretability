from src.data_functions import *

def train_epoch(conv_encoder, conv_decoder, ma_mi, device, optim, training_data, loss_coeff, dim_input, clipping):
    
    l1_loss = 0
    regularization_loss = 0
    loss = 0
    count = 0

    conv_encoder.train()
    conv_decoder.train()

    for field in training_data:
        
        optim.zero_grad()
        # do with dictionary
        l1, regularization_latent  = loss_sup_mixed(conv_encoder, conv_decoder, field ,ma_mi, device, loss_coeff, dim_input, True)

        (l1+regularization_latent).backward()
        if clipping[0]:
            tc.nn.utils.clip_grad_norm_(conv_encoder.parameters(), max_norm=clipping[1])
            tc.nn.utils.clip_grad_norm_(conv_decoder.parameters(), max_norm=clipping[1])
        optim.step()
        loss += (l1+regularization_latent).detach().cpu().item()
        
        l1_loss += l1.detach().cpu().item()
        regularization_loss += regularization_latent.detach().cpu().item()
        count += 1
        dictionary_losses_values = {'l_reconstruction':l1_loss/count, 'kl_regularization':regularization_loss/count, 'total': loss/count}
    return dictionary_losses_values


def valid_epoch(conv_encoder, conv_decoder, ma_mi, device, validation_data, loss_coeff, dim_input):
    """ This function performs the validation cycle for one epoch, i.e., it cycles on the validation batches and gets the corresponding validation metrics. 
    Those metrics are used to verify whether the model is overfitting by using early-stop.
    Args:
        conv_encoder (class 'src.architecture.Convolutional_Encoder): Encoder
        f (src.architecture.F_Latent): function f of the ODE of the latent dynamics
        conv_decoder (src.architecture.Convolutional_Decoder): Decoder
        ma_mi (list): list of lists of maximum and minima of fields and parameters
        device (torch.device): device where the training and validation are done
        validation_data (torch.utils.data.dataloader.DataLoader): data_loader for validation dataset
        loss_coeff (list): list of importance weights of the loss function terms
        RK (dict): dictionary with Butcher tablue for Runge-Kutta algorithms
        k (int): stage of Runge-Kutta algorithm
        start_backprop (list): list with values that determine up to which time-step in the past backpropagate the gradients
        dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
        lambda_regularization (float): coefficients that multiplies the regularization term of the latent vector
        time_dependence_in_f (bool): if true, the function f depends on time as well.
    Returns:
       [float, float, float, float, float, float, float] : The output are the mean values of the 7 training losses: [L_1, L_1 unnormalized, L_2^T, L_2^A, L_3, sum_of_previous, loss when predicting the full solution autoregressively]
    """
    conv_encoder.eval()
    conv_decoder.eval()

    l1_loss = 0
    l1_loss_unnorm = 0
    regularization_loss = 0
    loss = 0
    count = 0
    with tc.no_grad():
        for field in validation_data:

            l1, regularization_latent  = loss_sup_mixed(conv_encoder , conv_decoder, field, ma_mi, device, loss_coeff, dim_input, False)

            loss += (l1[0]+regularization_latent).detach().item()
            l1_loss += l1[0].detach().cpu().item()
            l1_loss_unnorm += l1[1].detach().cpu().item()
            regularization_loss += regularization_latent.detach().cpu().item()
            count += 1
    dictionary_losses_values = {'l_reconstruction':l1_loss/count, 'l_reconstruction_unnormed':l1_loss_unnorm/count, 'kl_regularization':regularization_loss/count ,'total': loss/count}
    return dictionary_losses_values


def loss_sup_mixed(conv_encoder, conv_decoder, field, ma_mi, device, loss_coeff, dim_input, train):

    """ this function defines all the terms that make up the loss function, L_1, L_2^T, L_2^A and L_3.

    Args:
        conv_encoder (class 'src.architecture.Convolutional_Encoder): Encoder
        f (src.architecture.F_Latent): function f of the ODE of the latent dynamics
        conv_decoder (src.architecture.Convolutional_Decoder): Decoder
        dt (torch.tensor): a tensor containing the dts used to advance each snapshot in time. It has dimensions [B, T-1], where B is the batch size and T is the length of the time series. it assumes each batch evolves accordingly to the same dts 
        param (torch.tensor): tensor of dimension [B, T, num_params], where B is the batch size and T is the length of the time series and num_params the number of parameters of the system
        ma_mi (list): list of lists of maximum and minima of fields and parameters
        device (torch.device): device where the training and validation are done
        loss_coeff (list): list of importance weights of the loss function terms
        RK (dict): dictionary with Butcher tablue for Runge-Kutta algorithms
        k (int): stage of Runge-Kutta algorithm
        start_backprop (list): list with values that determine up to which time-step in the past backpropagate the gradients
        dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
        lambda_regularization (float): coefficients that multiplies the regularization term of the latent vector
        train (bool): if true, it means the function was called inside the training, if false inside the validation
        time_dependence_in_f (bool): if true, the function f depends on time as well.

    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor: it returns the mean values of 6 losses, L_1, L_2^T, L_2^A, L_3, the loss which gets all the fields predicted
          autoregressively starting from the initial condition (used only at validation)  and the loss that regularizes the latent space
    """    
    field = field.to(device)
    size = field.size()

    #normalization and reshaping of the fields. Be careful, normalization here is an in-place hoperation
    if dim_input[1] == 1:
        grid = [size[-1]]

        field = normalize_field_known_values_field(field, ma_mi[0], ma_mi[1])
        input_encoder = field.reshape(size[0] * size[1] , dim_input[0], grid[0])
        
    elif dim_input[1] == 2:
        grid = [size[-1], size[-2]]

        field = normalize_field_known_values_field(field, ma_mi[0], ma_mi[1])
        input_encoder = field.reshape(size[0] * size[1] , dim_input[0], grid[1], grid[0])

    #First loss: invertibility of autoencoder enc-dec-enc
    l1, means, log_variances = l1_loss(conv_encoder, conv_decoder, input_encoder, dim_input, loss_coeff, ma_mi, train)
    #Second loss: regularization of the latent space: for AE it is simple l1 norm, form VAE it is KL divergence between posterior and prior
    l_regularization = l_regularization_latent_space(loss_coeff['kl_regularization'], means, log_variances)
    return l1, l_regularization

def l1_loss(conv_encoder, conv_decoder, input_encoder, dim_input, loss_coeff, ma_mi, train ):
    latent_space, means, log_variances = conv_encoder(input_encoder)
    back_to_physical, log_variances_reconstruction = conv_decoder(latent_space)
    l1 = reconstruction_loss_VAE(back_to_physical,input_encoder, log_variances_reconstruction, dim_input[1]) * loss_coeff['l_reconstruction']
    if not train:
        back_to_physical = inverse_normalization_field(back_to_physical, ma_mi[0], ma_mi[1], dim_input[1])
        l1_unnorm = L2_relative_loss(back_to_physical, inverse_normalization_field(input_encoder, ma_mi[0], ma_mi[1], dim_input[1]) , dim_input[1], False) * loss_coeff['l_reconstruction_unnormed']
        l1 = [l1, l1_unnorm]

    return l1, means, log_variances

def l_regularization_latent_space(lambda_regularization, means, log_variances):
    if lambda_regularization == 0.0:
        return tc.tensor(0.0)

    regularization_latent = tc.mean(KL_divergence(means, log_variances)) * lambda_regularization

    return regularization_latent

def KL_divergence(means,log_variances):
    KL = 0.5 * tc.sum(means**2+tc.exp(log_variances)-1-log_variances,dim=-1)
    return KL

def reconstruction_loss_VAE(inp, target, log_variances, dim_input): #how to treat sigmas of decoder is tricky and can bring instabilities to the training
    eps = tc.tensor(1e-8)
    if dim_input > 1:
        inp = inp.flatten(start_dim=-dim_input)
        target = target.flatten(start_dim=-dim_input)
        #log_variances = log_variances.flatten(start_dim=-dim_input)
    
    #log_variances = tc.clamp(log_variances, min=-6, max=6)
    #variances = tc.exp(log_variances) + eps 
    #loss = tc.mean(tc.sum((inp - target)**2, dim=-1) / (2 * variances) + inp.size(-1) * 0.5 * log_variances)
    loss = tc.mean(tc.sum((inp - target)**2, dim=-1) / 2)
    #if tc.isnan(loss):
    #    print('reconstruction_loss_VAE encountered NaN')
    #    print(log_variances)
    #    exit()
    return loss

def sample_from_gaussian(means, log_variances):
        random = tc.randn_like(means)
        sampled_vector = means + random * tc.exp( 0.5 * log_variances)
        return sampled_vector

def L2_relative_loss(inp, target, dim_inp, latent):
    """computes the relative (normalized) L2 norm between predicted tensor and expected tensor. It is used both for the predicted fields and for the predicted latent vectors. it is used 
    to compute L_1, L_2^T, L_2^A and L_3. 

    Args:
        inp (torch.Tensor): this is the tensor is the batch prediction of the field/latent vector
        target (torch.Tensor): this is the tensor is the expected batch of the field/latent vector
        dim_inp (int): spatial dimension of the solution field
        latent (bool): if true it means inp and target are reduced vectors

    Returns:
        torch.tensor: scalar tensor, it is the (normalized) L2 loss
    """    

    eps = tc.tensor(1e-8)
    if latent:
        #with tc.no_grad():
        norm = tc.sum(target**2, dim=-1, keepdim=True)**0.5
        L2_relative = tc.mean(tc.sum((inp - target)**2,dim=-1, keepdim=True)**0.5 / tc.max(norm, eps))
        return L2_relative
    else:
        if dim_inp > 1:
            inp = inp.flatten(start_dim=-dim_inp)
            target = target.flatten(start_dim=-dim_inp)
        norm = tc.linalg.vector_norm(target, dim=-1)
        L2_relative = tc.mean(tc.linalg.vector_norm(inp - target, dim=-1) / tc.max(norm, eps))
        return L2_relative