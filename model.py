import torch.nn as nn
import torchaudio
import torch
from torch.nn.functional import sigmoid, softmax

bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
W2V = bundle.get_model(dl_kwargs={"model_dir": "./models"})

class Modelo_Fusion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # Back-End wav2vec 2.0 network
        ## Mezcal capas W2V
        self.pesos_capas_W2V = nn.Parameter(torch.rand(1,24,1,1))
        self.normalizacion = nn.InstanceNorm2d(24)

        ## Capa FT del W2V
        self.LL = nn.Linear(1024, opt.FT_size) #201,128
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Procesado Intermedio

        self.lstm = nn.LSTM(opt.FT_size, opt.hidden_size, opt.lstm_layers, batch_first=True, bidirectional=False)
        
        # Capas finales
        emb_size = opt.hidden_size

        self.biases = nn.Parameter(torch.rand(4))
        self.lineal_weights_b = nn.Sequential(nn.Linear(emb_size, int(emb_size/2)), 
                                              nn.ReLU(), 
                                              nn.Linear(int(emb_size/2), 4))
        self.lineal_weights_s = nn.Sequential(nn.Linear(emb_size, int(emb_size/2)), 
                                              nn.ReLU(), 
                                              nn.Linear(int(emb_size/2), 4))
        

    def forward(self, W2V_features, scores_redes, use_softmax=False):
        '''
        W2V_features: Tensor [batch_size, 24, Time, 1024]
        scores_redes: Tensor con los scores de la redes [batch_size, 4]
        '''
        # Union features W2V 
        tensor_norm = self.normalizacion(W2V_features)
        pesos_norm = self.pesos_capas_W2V/(self.pesos_capas_W2V.sum())
        x = ((tensor_norm * pesos_norm).sum(dim=1))

        # Capa FT W2V 
        x = self.LL(x)
        x = x.unsqueeze(dim=1)  
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1) #[bs, Time, 128]

        # procesado intermedio
        x, _ = self.lstm(x) #[bs, T, 64*2]
        embedding =  x[:, -1, :] #[bs, 64*2]

        # Capas finales
        weights_b = self.lineal_weights_b(embedding)
        weights_s = self.lineal_weights_s(embedding)

        score_bona = (weights_b*(scores_redes-self.biases)).sum(dim=1)
        score_spoof = (weights_s*(scores_redes-self.biases)).sum(dim=1)

        scores = torch.stack([score_bona, score_spoof], dim=1)
        if use_softmax and not self.training: scores = softmax(scores, dim=1)
        return scores, (weights_b, weights_s)




class Modelo_Clasificacion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # Back-End wav2vec 2.0 network
        ## Mezcal capas W2V
        self.pesos_capas_W2V = nn.Parameter(torch.rand(1,24,1,1))
        self.normalizacion = nn.InstanceNorm2d(24)

        ## Capa FT del W2V
        self.LL = nn.Linear(1024, opt.FT_size) #201,128
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Procesado Intermedio

        self.lstm = nn.LSTM(opt.FT_size, opt.hidden_size, opt.lstm_layers, batch_first=True, bidirectional=False)
        
        # Capas finales
        emb_size = opt.hidden_size
        self.lineal_class = nn.Sequential(nn.Linear(emb_size, int(emb_size/2)), 
                                              nn.ReLU(), 
                                              nn.Linear(int(emb_size/2), 2))

        

    def forward(self, W2V_features, scores_redes):
        '''
        W2V_features: Tensor [batch_size, 24, Time, 1024]
        scores_redes: Tensor con los scores de la redes [batch_size, 4]
        '''
        # Union features W2V 
        tensor_norm = self.normalizacion(W2V_features)
        x = ((tensor_norm * self.pesos_capas_W2V).sum(dim=1))

        # Capa FT W2V 
        x = self.LL(x)
        x = x.unsqueeze(dim=1)  
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1) #[bs, Time, 128]

        # procesado intermedio
        x, _ = self.lstm(x) #[bs, T, 64*2]
        embedding =  x[:, -1, :] #[bs, 64*2]

        # Capas finales
        scores = self.lineal_class(embedding)
        return scores, embedding
