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

def l2_loss(conv_encoder, f, conv_decoder , input_encoder, size, loss_coeff, k, RK, ma_mi, device, time_dependence_in_f, dim_input, start_backprop, train, grid, latent_dim, latent_space, input_processor, dt, param, Auto_Encoder):

    if loss_coeff[1] <= 0:
        l2_TF = tc.tensor(0.0)
    else:
        e2_latent_TF = processor_First_Order(f,input_processor,dt, param, k, RK, ma_mi, device, time_dependence_in_f)
        e2_latent_TF = e2_latent_TF.reshape(size[0], (size[1]-1), latent_dim)
        l2_TF = L2_relative(e2_latent_TF, latent_space[:, 1:, :], 1, True) * loss_coeff[1]

    if loss_coeff[2] <= 0:
        with tc.no_grad():
            l2_AR = tc.tensor(0.0)
            l_final = tc.tensor(0.0)
    else:
        if dim_input[1] == 1:
            input_encoder = input_encoder.reshape(size[0] , size[1] , dim_input[0], grid[0])
        elif dim_input[1] ==2:
            input_encoder = input_encoder.reshape(size[0] , size[1] , dim_input[0], grid[1] , grid[0])

        l2_AR, l_final = advance_from_ic(conv_encoder, f, conv_decoder, input_encoder ,latent_space, tc.reshape(dt,(size[0],size[1]-1)).unsqueeze(-1), param.reshape(size[0] , (size[1]-1) , param.size(-1)), k, RK, ma_mi, device, start_backprop, size, loss_coeff[2], dim_input,train, time_dependence_in_f, Auto_Encoder)

    return l2_TF, l2_AR, l_final

def l3_loss( size, loss_coeff, f, k, RK, ma_mi, device, time_dependence_in_f, latent_dim, latent_space, input_processor, dt, param):
    if loss_coeff[3] <= 0:
        l3 = tc.tensor(0.0)
    else:
        random_dt = tc.rand(size[0]*(size[1]-1),1, device=device) * dt
        e2_middle_latent = processor_First_Order(f, input_processor,random_dt, param, k, RK, ma_mi, device, time_dependence_in_f)
        e2_final = processor_First_Order(f, e2_middle_latent, dt-random_dt, param, k, RK, ma_mi, device, time_dependence_in_f)
        e2_final = e2_final.reshape(size[0], (size[1]-1), latent_dim)
        l3 = L2_relative(e2_final, latent_space[:, 1:, :], 1, True) * loss_coeff[3]
    return l3

def l4_regularization_latent_space(train, latent_space, lambda_regularization, Auto_Encoder, means = None, log_variances = None):
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

def loss_sup_mixed(conv_encoder, f, conv_decoder, field, dt, param, ma_mi, device, loss_coeff, RK, k, start_backprop, dim_input, lambda_regularization,train, time_dependence_in_f, Auto_Encoder):

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
    dt = dt.to(device)
    param = param.to(device)
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

    #Second loss: dynamics of latent space for TF and AR
    latent_dim = latent_space.size()[-1]
    latent_space = latent_space.reshape(size[0], size[1], latent_dim)
    input_processor = latent_space[:, 0:-1, :].reshape(size[0]*(size[1]-1),latent_dim)
    dt = tc.reshape(dt,(size[0]*(size[1]-1), 1))
    param = param[:,0:-1,:].reshape(size[0] * (size[1]-1) , param.size(-1))

    l2_TF, l2_AR, l_final = l2_loss(conv_encoder, f, conv_decoder , input_encoder, size, loss_coeff, k, RK, ma_mi, device, time_dependence_in_f, dim_input, start_backprop, train, grid, latent_dim, latent_space, input_processor, dt, param, Auto_Encoder)

    #Third loss: Forcing learning one f which is able to predict in smaller dt (for prediction in latent space)
    l3 = l3_loss(size, loss_coeff, f, k, RK, ma_mi, device, time_dependence_in_f, latent_dim, latent_space, input_processor, dt, param)

    # fourth loss: regularization of the latent space: for AE it is simple l1 norm, form VAE it is KL divergence between posterior and prior
    if Auto_Encoder == 'AE':
        l4_regularization = l4_regularization_latent_space(train, latent_space, lambda_regularization, Auto_Encoder)
    elif Auto_Encoder == 'VAE':
        l4_regularization = l4_regularization_latent_space(train, latent_space, lambda_regularization, Auto_Encoder, means, log_variances)
    return l1, l2_TF, l2_AR, l3, l_final, l4_regularization
    
def advance_from_ic(conv_encoder, f, conv_decoder, input_encoder, true_latent, dt, param, k, RK, ma_mi, device, start_backprop, size, coeff, dim_input,train, time_dependence_in_f, Auto_Encoder):
    """ this function advances autoregressively the reduced vector at time t = 0 across the whole time series, effectively mimicking what happens at testing time. It is thus used to compute
        L_2^A. We use this function to compute L_2^A and to get the full predicted time series of solution fields (needed during validation to compute a validation metric).
        This function can implement 3 different algorithms depending on the value of start_backprop[0]:
        - start_backprop[0] = 0: the initial confition is encoded and the latent vectors are predicted autoregressively keeping the gradients since the encoding
        - start_backprop[0] = 1: the initial confition is encoded and the latent vectors are predicted autoregressively keeping the gradients since up to the previous start_backprop[1] time steps
        - start_backprop[0] = 2: the initial confition is encoded and the latent vectors are predicted autoregressively keeping the gradients since up to the previous start_backprop[1] time steps,
        where start_backprop[1]=1 in the first epoch and is then increased by 1 dynamically every TBPP_dynamic[1] epochs, where TBPP_dynamic is specified in initial_information.yaml.

    Args:
        conv_encoder (class 'src.architecture.Convolutional_Encoder): Encoder
        f (src.architecture.F_Latent): function f of the ODE of the latent dynamics
        conv_decoder (src.architecture.Convolutional_Decoder): Decoder
        input_encoder (torch.Tensor): tensor of dimensions [B,T,C,x_dim_1, dim_x_1, dim_x_2, ...], where B is batch_size, T is the length of the full time series, C is the number of channels of the 
            predicted solution field, dim_x_1, dim_x_2, ... are the dimensions of the first spatial dimension, second spatial dimension, etc. It is the field solution over time
        true_latent (torch.Tensor): tensor of dimensions [B,T,latent_dim], where B is batch_size, T is the length of the full time series and latent_dim is the dimension of the latent space. It is the expected latent vectors
        dt (torch.Tensor): a tensor containing the dts used to advance each snapshot in time. It has dimensions [B, T-1], where B is the batch size and T is the length of the time series. it assumes each batch evolves accordingly to the same dts 
        param (torch.Tensor): tensor of dimension [B, T, num_params], where B is the batch size and T is the length of the time series and num_params the number of parameters of the system
        k (int): stage of Runge-Kutta algorithm
        RK (dict): dictionary with Butcher tablue for Runge-Kutta algorithms
        ma_mi (list): list of lists of maximum and minima of fields and parameters
        device (torch.device): device where the training and validation are done
        start_backprop (list): list with values that determine up to which time-step in the past backpropagate the gradients
        size (torch.Size): tensor representing the length of each dimension of the solution field tensor
        coeff (float): coefficient that multiplies the loss function L_2^A
        dim_input (list): first dimension is the channels of the solution field, second is the number of spatial dimensions
        train (bool): if true, this function was called inside the training loop, otherwise in the validation loop
        time_dependence_in_f (bool):  if true, the function f depends on time as well.

    Returns:
        torch.tensor(), torch.tensor():  the mean L_2^A,  the loss function which takes into account the field solution predicted autoregressively (only at validation)
    """    
    if start_backprop[0] == 0 or (not train):  #Encode initial condition and evolve in latent
        l_final = tc.tensor(0., device = device)
        l2_AR = tc.tensor(0., device = device)

        if start_backprop[1] == 0:
            if Auto_Encoder == 'AE':
                next_latent = conv_encoder(input_encoder[:,0,...])

            elif Auto_Encoder == 'VAE':
                _, means, log_variances = conv_encoder(input_encoder[:,0,...])
                next_latent = tc.cat((means, log_variances),dim=-1)
        else:
            with tc.no_grad():
                if Auto_Encoder == 'AE':
                    next_latent = conv_encoder(input_encoder[:,0,...])
                elif Auto_Encoder == 'VAE':
                    _, means, log_variances = conv_encoder(input_encoder[:,0,...])
                    next_latent = tc.cat((means, log_variances),dim=-1)

        if (not train):
            input_encoder = inverse_normalization_field(input_encoder, ma_mi[0], ma_mi[1], dim_input[1])
        
        step = 0

        for count in range(size[1]-1):
            if count < start_backprop[1]:
                with tc.no_grad():
                    next_latent = processor_First_Order(f, next_latent, dt[:,count,:], param[:,count,:], k, RK, ma_mi, device, time_dependence_in_f)
                    l2_AR += L2_relative(next_latent, true_latent[:,count+1,:], dim_input[1], True) 
                    step+=1
            else:
                next_latent = processor_First_Order(f, next_latent, dt[:,count,:], param[:,count,:], k, RK, ma_mi, device, time_dependence_in_f)
                l2_AR += L2_relative(next_latent, true_latent[:,count+1,:], dim_input[1], True) 
                step+=1
            if (not train):
                if Auto_Encoder == 'AE':
                    denorm_latent = inverse_normalization_field(conv_decoder(next_latent), ma_mi[0], ma_mi[1], dim_input[1])

                if Auto_Encoder == 'VAE':
                    latent_dim = int(next_latent.size(-1)/2)
                    sampled_latent = sample_from_gaussian(next_latent[:,0:latent_dim],next_latent[:, latent_dim:] )
                    denorm_latent, _ = conv_decoder(sampled_latent)
                    denorm_latent = inverse_normalization_field(denorm_latent, ma_mi[0], ma_mi[1], dim_input[1])
                l_final += L2_relative(denorm_latent, input_encoder[:,count+1,...] , dim_input[1], False) #first dimension is batch, second time, then C,W,H

        return l2_AR/step * coeff, l_final/(size[1]-1)
    
    elif start_backprop[0] == 1: #Encode ic and evolve in latent but TBPP

        place_holder = tc.zeros((size[0],size[1]-start_backprop[1], true_latent.size()[-1]), device = device)
        next_latent = conv_encoder(input_encoder[:,0,...])
        place_holder[:,0,:] = next_latent.clone()
        with tc.no_grad():
            for i in range(size[1]-start_backprop[1]-1):
                next_latent = processor_First_Order(f, next_latent, dt[:,i+1,:], param[:,i+1,:], k, RK, ma_mi, device, time_dependence_in_f)
                place_holder[:,i+1,:] = next_latent.detach().clone()

        place_holder = tc.reshape(place_holder,(size[0]*(size[1]-start_backprop[1]),true_latent.size()[-1]))
        l2_AR_1 = tc.tensor(0.0,device=device)
        next_latent = conv_encoder(input_encoder[:,0,...])
        for i in range(start_backprop[1]-1):
            next_latent = processor_First_Order(f, next_latent, dt[:,i,:], param[:,i,:], k, RK, ma_mi, device, time_dependence_in_f)
            l2_AR_1 += L2_relative(next_latent, true_latent[:,i+1,:], dim_input[1], True)

        l2_AR_2 = tc.tensor(0.0,device=device)
        for i in range(start_backprop[1]):
            place_holder = processor_First_Order(f, place_holder, dt[:,i:size[1]-start_backprop[1]+i,:].flatten().unsqueeze(-1), param[:,i:size[1]-start_backprop[1]+i,:].reshape(-1, param.size(-1)+time_dependence_in_f), k, RK, ma_mi, device, time_dependence_in_f)
        
        place_holder = tc.reshape(place_holder,(size[0],size[1]-start_backprop[1],true_latent.size()[-1]))
        l2_AR_2 += L2_relative(place_holder, true_latent[:,start_backprop[1]:,:], dim_input[1], True)

        return (l2_AR_1 * (start_backprop[1]-1) +l2_AR_2 *(size[1]-start_backprop[1]))/(size[1]-1) * coeff, tc.tensor(0.0)
        
    elif start_backprop[0] == 2: # Encode full field start_backprop[1] steps in advance and from there TBPP

        l2_AR_1 = tc.tensor(0.0,device=device)
        next_latent = conv_encoder(input_encoder[:,0,...])
        for i in range(start_backprop[1]-1):
            next_latent = processor_First_Order(f, next_latent, dt[:,i,:], param[:,i,:], k, RK, ma_mi, device, time_dependence_in_f)
            l2_AR_1 += L2_relative(next_latent, true_latent[:,i+1,:], dim_input[1], True)

        if dim_input[1] == 1:
            x_grid = size[-1]
            input_encoder = tc.reshape(input_encoder[:,0:size[1]-start_backprop[1],:],(size[0]*(size[1]-start_backprop[1]), dim_input[0], x_grid))       
        elif dim_input[1] == 2:
            x_grid = size[-1]
            y_grid = size[-2]
            input_encoder = tc.reshape(input_encoder[:,0:size[1]-start_backprop[1],:],(size[0]*(size[1]-start_backprop[1]), dim_input[0], y_grid, x_grid))
        place_holder = conv_encoder(input_encoder)

        for i in range(start_backprop[1]): 
            place_holder = processor_First_Order(f, place_holder, dt[:,i:size[1]-start_backprop[1]+i,:].flatten().unsqueeze(-1) , param[:,i:size[1]-start_backprop[1]+i,:].reshape(-1, param.size(-1)+time_dependence_in_f), k, RK, ma_mi, device, time_dependence_in_f)

        place_holder = tc.reshape(place_holder,(size[0],size[1]-start_backprop[1],true_latent.size()[-1]))
        l2_AR_2 = L2_relative(place_holder, true_latent[:,start_backprop[1]:,:], dim_input[1], True)

        return (l2_AR_1 * (start_backprop[1]-1) +l2_AR_2 *(size[1]-start_backprop[1]))/(size[1]-1) * coeff, tc.tensor(0.0)
        
def processor_First_Order(f, e1,dt, mu, k, RK, ma_mi, device, time_dependence_in_f):
    """this function implements the Runge-Kutta algorithms. First_Order refers to the fact that the ODE is a first order ODE, although higher orders would still be solved by this algorithms
    simply introducing new functions.

    Args:
        f (src.architecture.F_Latent): function f of the ODE of the latent dynamics
        e1 (torch.tensor()): tensor of dimension [B, dim_latent], where B is the batch size and dim_latent the dimension of the latent space
        dt (torch.Tensor): a tensor containing the dts used to advance each snapshot in time. It has dimensions [B, T-1], where B is the batch size and T is the length of the time series. it assumes each batch evolves accordingly to the same dts 
        mu (tc.tensor()): tensor of dimension [B, num_params] where B is the batch size and num_params the number of parameters of the system
        k (int): stage of Runge-Kutta algorithm
        RK (dict): dictionary with Butcher tablue for Runge-Kutta algorithms
        ma_mi (list): list of lists of maximum and minima of fields and parameters
        device (torch.device): device where the training and validation are done
        time_dependence_in_f (bool):  if true, the function f depends on time as well.

    Returns:
        torch.tensor(): tensor of dimension [B, dim_latent] which contains the latent vectors advanced in time from e1 of dt
    """    
    # k=1 is Euler
    ma_mi[2] = ma_mi[2].to(device)
    ma_mi[3] = ma_mi[3].to(device)
    mu = normalize_field_known_values_param(mu, ma_mi[2], ma_mi[3])
    b = tc.zeros((k, e1.size(0), e1.size(1)) , device= device)
    b[0, :,:] = f(e1, mu )
    final_sum = f(e1, mu)*RK[str(k)][-1][1]

    for i in range(k-1):
        mu_in_time = mu.clone() #avoid in place operation which messes with backprop.
        if time_dependence_in_f:
            mu_in_time[:,-1] = mu_in_time[:,-1] +  RK[str(k)][i+1][0] * dt.squeeze(-1) 
        s = tc.zeros_like(e1, device = device)

        for j in range(i+1):
            s +=  b[j] * RK[str(k)][i+1][j+1]

        b_new = f(e1 + dt * s, mu_in_time).unsqueeze(0).to(device)
        b[i+1,:,:] = b_new

        final_sum += b_new.squeeze(0) * RK[str(k)][-1][i+2]
    e2 = e1 + final_sum * dt
    return e2
