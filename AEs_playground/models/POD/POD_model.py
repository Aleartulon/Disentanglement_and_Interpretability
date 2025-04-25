from src.data_functions import *

class POD:
    def __init__(self, global_info, model_info):
        self.latent_dimension = model_info['latent_dimension']
        self.dim_input = global_info['dim_input']
        self.PATH = global_info['PATH']
        self.device = global_info['device']
    def send_to_device_and_set_optimizer(self, learning_rate, gamma_scheduler, warmup_time, weight_decay,device):
        return 0, 0, 0

    def train(self, training, validation, ma_mi, optim, pre_scheduler, scheduler):
        print('---------------------------------------------')
        print('Starting SVD computation')
        for i in training:
            size = i.size()
            i = i.to(self.device)
            i = normalize_field_known_values_param(i, ma_mi[0].to(self.device), ma_mi[1].to(self.device))
            i = tc.reshape(i,(size[0]*size[1],-1))
            print('Shape of input SVD', i.size())
            t1 = time.time()
            self.U, self.S, self.V = tc.linalg.svd(i, full_matrices=False)
            self.U = self.U[:, :self.latent_dimension]
            self.S = self.S[:self.latent_dimension]
            self.V = self.V[:self.latent_dimension]
            t2 = time.time()
            print(f"Time for SVD computation: {t2 - t1:.4f} s")
        os.makedirs(self.PATH+'/SVD/',exist_ok=True)
        tc.save({'U': self.U , 'S': self.S , 'V': self.V}, self.PATH+'/SVD/svd.pth')


