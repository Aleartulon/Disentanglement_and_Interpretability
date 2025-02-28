import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data_functions import *
#set type of tensors
tc.set_default_dtype(tc.float32)

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
 
    global_information = load_config('configs/global_information.yaml')
    model_information = load_config('configs/information_'+str(global_information['model'])+'.yaml')
    
    #define directories and get data

    global_information['PATH'] = global_information['physics_model'] +'/Models/' + global_information['description']

    #encode information in txt files and delete path where losses are saved to avoid having data from different runs
    os.makedirs(global_information['PATH']+'/scripts/bin',exist_ok=True)
    os.makedirs(global_information['PATH']+'/scripts/src',exist_ok=True)
    os.makedirs(global_information['PATH']+'/scripts/configs',exist_ok=True)
    
    if os.path.exists(global_information['PATH'] +'/runs'):
        shutil.rmtree(global_information['PATH']+'/runs') 

    #copy scripts
    shutil.copy('bin/main.py', global_information['PATH']+'/scripts/bin/')
    shutil.copy('../models/'+str(global_information['model'])+'/'+str(global_information['model'])+'_model.py', global_information['PATH']+'/scripts/src')
    shutil.copy('../models/'+str(global_information['model'])+'/training_validation_functions.py', global_information['PATH']+'/scripts/src')
    shutil.copy('src/data_functions.py', global_information['PATH']+'/scripts/src')
    shutil.copy('configs/global_information.yaml', global_information['PATH']+'/scripts/configs')
    shutil.copy('configs/information_'+str(global_information['model'])+'.yaml', global_information['PATH']+'/scripts/configs')

    # go to gpu if possible
    device = tc.device(global_information['device']) if tc.cuda.is_available() else tc.device("cpu")
    print(f'Selected device: {device}')

    #define the model
    module_name = 'models.'+str(global_information['model'])+'.'+str(global_information['model'])+'_model'
    class_name = global_information["class_name"]
    module = importlib.import_module(module_name)
    ClassRef = getattr(module, class_name)
    model = ClassRef(global_information, model_information)

    #move the models to the device and set optimizer
    optim, pre_scheduler, scheduler = model.send_to_device_and_set_optimizer(global_information['learning_rate'], global_information['warmup_lr'], device)
    #print information

    print('---------- INITIAL INFORMATION ----------')
    for key, value in global_information.items():
        print(key, ' : ', value)
    print(" ")
    print('---------- MODEL INFORMATION ----------')
    for key, value in model_information.items():
        print(key, ' : ', value)
    print(" ")


    # prepare dataloaders for training and validation
    data_path = global_information['data_path'] 
    dataset_training = CustomStarDataset_Big_Dataset(data_path + 'image_training.npy')
    dataset_validation = CustomStarDataset_Big_Dataset(data_path + 'image_validation.npy')

    training = DataLoader(dataset_training,batch_size=global_information['batch_size'], num_workers=global_information['num_workers'], shuffle=True,drop_last=True,pin_memory=True)
    validation = DataLoader(dataset_validation,batch_size=global_information['batch_size'], num_workers=global_information['num_workers'], shuffle=True,drop_last=True,pin_memory=True)

    # get max and min for training from the training dataset
    ma_mi = get_max_and_min(training, global_information['dim_input'], global_information['normalization_field_ma'],global_information['normalization_field_mi']) 

    norm = {'Field':{'Max': ma_mi[0],'Min': ma_mi[1]}
                }
    with open(global_information['PATH'] + '/Normalization.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(norm.keys())  # Write header
        writer.writerow(norm.values())  # Write data

    #training starts

    model.train(training, validation, ma_mi, optim, pre_scheduler, scheduler)
    
if __name__ == '__main__':
    main()
