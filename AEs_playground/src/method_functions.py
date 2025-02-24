from src.data_functions import *

def l1_loss(conv_encoder, conv_decoder, input_encoder, dim_input, loss_coeff, ma_mi, train, Auto_Encoder ):

    if Auto_Encoder == 'AE':
        latent_space = conv_encoder(input_encoder)

    elif Auto_Encoder == 'VAE':
        latent_space, means, log_variances = conv_encoder(input_encoder)

    if Auto_Encoder == 'AE':
        back_to_physical = conv_decoder(latent_space)
        l1 = L2_relative(back_to_physical,input_encoder, dim_input[1], False) * loss_coeff[0]

    if Auto_Encoder == 'VAE':
        back_to_physical, log_variances_reconstruction = conv_decoder(latent_space)
        l1 = reconstruction_loss_VAE(back_to_physical,input_encoder, log_variances_reconstruction, dim_input[1])

    if not train:
        back_to_physical = inverse_normalization_field(back_to_physical, ma_mi[0], ma_mi[1], dim_input[1])
        l1_unnorm = L2_relative(back_to_physical, inverse_normalization_field(input_encoder, ma_mi[0], ma_mi[1], dim_input[1]) , dim_input[1], False) * loss_coeff[0]
        l1 = [l1, l1_unnorm]

    if Auto_Encoder == 'AE':
        return l1, latent_space
    elif Auto_Encoder == 'VAE':
        return l1, means, log_variances

def l_regularization_latent_space(train, latent_space, lambda_regularization, Auto_Encoder, means = None, log_variances = None):
    if lambda_regularization == 0.0:
        return tc.tensor(0.0)
    if Auto_Encoder == 'AE':
        regularization_latent = l1_latent_regularization(latent_space, lambda_regularization)
    
    elif Auto_Encoder == 'VAE':
        regularization_latent = tc.mean(KL_divergence(means, log_variances)) * lambda_regularization
    return regularization_latent

def reconstruction_loss_VAE(inp,target, log_variances, dim_input):
    eps = tc.tensor(1e-8)
    if dim_input > 1:
        inp = inp.flatten(start_dim=-dim_input)
        target = target.flatten(start_dim=-dim_input)
        log_variances = log_variances.flatten(start_dim=-dim_input)
    loss = tc.mean(tc.sum((inp - target)**2/(2*tc.exp(log_variances))+0.5 * log_variances,dim=-1))
    return loss

def l1_latent_regularization(x, lambda_l1):
    """ It is used to push as many as possible latent dimensions towards 0 

    Args:
        x (torch.tensor): tensor of dimension [B, T, latent_dim], where B is the batch size and T is the length of the time series and latent_dim is the dimension of the latent space
        lambda_l1 (float): it is multiplied to the loss that regularizes the latent space (the larger the stronger the regularization effect)
    Returns:
        torch.tensor: scalar tensor, it is the regularization loss
    """    
    l1_norm = tc.mean(tc.abs(x))

    return lambda_l1 * l1_norm

def KL_divergence(means,log_variances):
    KL = 0.5 * tc.sum(means**2+tc.exp(log_variances)-1-log_variances,dim=-1)
    return KL

def L2_relative(inp, target, dim_inp, latent):
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

def sample_from_gaussian(means, log_variances):
        random = tc.rand_like(means)
        sampled_vector = means + random * tc.exp( 0.5 * log_variances)
        return sampled_vector

def loss_sup_mixed(conv_encoder, conv_decoder, field, ma_mi, device, loss_coeff, dim_input, train, Auto_Encoder):

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
    if Auto_Encoder == 'AE':
        l1, latent_space = l1_loss(conv_encoder, conv_decoder, input_encoder, dim_input, loss_coeff, ma_mi, train, Auto_Encoder)

    elif Auto_Encoder == 'VAE':
        l1, means, log_variances = l1_loss(conv_encoder, conv_decoder, input_encoder, dim_input, loss_coeff, ma_mi, train, Auto_Encoder)
        latent_space = tc.cat((means, log_variances), dim=-1)

    #Second loss: regularization of the latent space: for AE it is simple l1 norm, form VAE it is KL divergence between posterior and prior
    if Auto_Encoder == 'AE':
        l_regularization = l_regularization_latent_space(train, latent_space, loss_coeff[-1], Auto_Encoder)
    elif Auto_Encoder == 'VAE':
        l_regularization = l_regularization_latent_space(train, latent_space, loss_coeff[-1], Auto_Encoder, means, log_variances)
    return l1, l_regularization
    