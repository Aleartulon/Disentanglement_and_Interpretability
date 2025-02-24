import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.architecture import *
from src.training_validation_functions import nn_training
from src.data_functions import *

#set type of tensors
tc.set_default_dtype(tc.float32)

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
 
    initial_information = load_config('configs/initial_information.yaml')
    model_information = load_config('configs/model_information_'+str(initial_information['AutoEncoder'])+'.yaml')
    n_halving = len(np.where(np.array(model_information['stride_enc'])==2)[0]) 


    if initial_information['dim_input'][1] == 1:
        final_reduction = int(initial_information['side_size'] /(2**n_halving))
        model_information['input_output_dfnn'] = int(final_reduction * model_information['filters_enc'][-1] ) #dimension of the vector the final linear layer of the encoder receives
    elif initial_information['dim_input'][1] == 2:
        final_reduction = int(initial_information['side_size'] /(2**n_halving))
        model_information['input_output_dfnn'] = int(final_reduction**2 * model_information['filters_enc'][-1] ) #dimension of the vector the final linear layer of the encoder receives
    #define directories and get data

    initial_information['PATH'] = initial_information['physics_model'] +'/Models/' + initial_information['description']

    #encode information in txt files and delete path where losses are saved to avoid having data from different runs
    os.makedirs(initial_information['PATH']+'/scripts/bin',exist_ok=True)
    os.makedirs(initial_information['PATH']+'/scripts/src',exist_ok=True)
    os.makedirs(initial_information['PATH']+'/scripts/configs',exist_ok=True)
    
    if os.path.exists(initial_information['PATH'] +'/runs'):
        shutil.rmtree(initial_information['PATH']+'/runs') 

    #copy scripts
    shutil.copy('bin/main.py', initial_information['PATH']+'/scripts/bin/')
    shutil.copy('src/architecture.py', initial_information['PATH']+'/scripts/src')
    shutil.copy('src/training_validation_functions.py', initial_information['PATH']+'/scripts/src')
    shutil.copy('src/data_functions.py', initial_information['PATH']+'/scripts/src')
    shutil.copy('src/method_functions.py', initial_information['PATH']+'/scripts/src')
    shutil.copy('configs/initial_information.yaml', initial_information['PATH']+'/scripts/configs')
    shutil.copy('configs/model_information_'+str(initial_information['AutoEncoder'])+'.yaml', initial_information['PATH']+'/scripts/configs')

    # go to gpu if possible
    device = tc.device(initial_information['which_device']) if tc.cuda.is_available() else tc.device("cpu")
    print(f'Selected device: {device}')

    #define the ENCODER and the Decoder 
    if initial_information['AutoEncoder'] == 'AE':
        conv_encoder = Convolutional_Encoder_AE(initial_information['dim_input'], model_information['kernel_enc'], model_information['filters_enc'], model_information['stride_enc'], model_information['input_output_dfnn'], model_information['latent_dim'], model_information['final_and_initial_activation'])
        conv_decoder = conv_decoder = Convolutional_Decoder_AE(initial_information['dim_input'], model_information['kernel_deco'], model_information['filters_deco'], model_information['stride_dec'], model_information['latent_dim'], model_information['input_output_dfnn'], final_reduction, model_information['final_and_initial_activation'], model_information['number_channels_input_cnns_deco'])
    
    if initial_information['AutoEncoder'] == 'VAE':
        conv_encoder = Convolutional_Encoder_VAE(initial_information['dim_input'], model_information['kernel_enc'], model_information['filters_enc'], model_information['stride_enc'], final_reduction**2, model_information['latent_dim'], model_information['final_and_initial_activation'])
        conv_decoder = Convolutional_Decoder_VAE(initial_information['dim_input'], model_information['kernel_deco'], model_information['filters_deco'], model_information['stride_dec'], model_information['latent_dim'], model_information['input_output_dfnn'], final_reduction, model_information['final_and_initial_activation'], model_information['number_channels_input_cnns_deco'])

    #move the models to the device
    conv_encoder.to(device)
    conv_decoder.to(device)

    #define optimizer, the pre scheduler for the warmup of the model and the scheduler
    params_to_optimize = [
        {'params': conv_encoder.parameters(), 'weight_decay': 0},
        {'params': conv_decoder.parameters(), 'weight_decay': 0}
    ]
    optim = tc.optim.Adam(params_to_optimize, lr=initial_information['learning_rate'])
    lambda1 = lambda i : i / initial_information['time_of_AE']
    pre_scheduler = tc.optim.lr_scheduler.LambdaLR(optim,lambda1) #warm up of the learning rate
    scheduler = tc.optim.lr_scheduler.ExponentialLR(optim, initial_information['gamma_lr'])

    #print information

    print('---------- INITIAL INFORMATION ----------')
    for key, value in initial_information.items():
        print(key, ' : ', value)
    print(" ")
    print('---------- MODEL INFORMATION ----------')
    for key, value in model_information.items():
        print(key, ' : ', value)
    print(" ")


    # prepare dataloaders for training and validation
    data_path = initial_information['data_path'] 
    dataset_training = CustomStarDataset_Big_Dataset(data_path + 'field_step_training.npy')
    dataset_validation = CustomStarDataset_Big_Dataset(data_path + 'field_step_validation.npy')

    training = DataLoader(dataset_training,batch_size=initial_information['batch_size'], num_workers=initial_information['num_workers'], shuffle=True,drop_last=True,pin_memory=True)
    validation = DataLoader(dataset_validation,batch_size=initial_information['batch_size'], num_workers=initial_information['num_workers'], shuffle=True,drop_last=True,pin_memory=True)

    # get max and min for training from the training dataset
    ma_mi = get_max_and_min(training, initial_information['dim_input'], initial_information['normalization_field_ma'],initial_information['normalization_field_mi']) 

    norm = {'Field':{'Max': ma_mi[0],'Min': ma_mi[1]}
                }
    with open(initial_information['PATH'] + '/Normalization.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(norm.keys())  # Write header
        writer.writerow(norm.values())  # Write data

    #training starts

    nn_training(conv_encoder, conv_decoder,training, validation, ma_mi, device, optim, initial_information['epochs'], 
        initial_information['PATH'], initial_information['loss_coefficients'], pre_scheduler,scheduler,
        initial_information['checkpoint'], initial_information['time_of_AE'], initial_information['dim_input'], initial_information['lambda_regularization_strength'], 
        initial_information['clipping'], initial_information['AutoEncoder'])
    
if __name__ == '__main__':
    main()