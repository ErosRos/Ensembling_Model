import torchaudio
import torch
import torch.nn as nn
import sys

def produce_evaluation_file(dataloader, fusion_model, W2V, device, save_path, use_softmax=False):
    fusion_model.eval()
    W2V.eval()

    num_batch = len(dataloader)
    i=0
    fh = open(save_path, "w") #borramos si ya existe
    fh.close()

    for utt_id, waveform, scores_cm in dataloader:
        waveform = waveform.to(device)  
        waveform = waveform.float()

        W2V_features,_ = W2V.extract_features(waveform)
        W2V_features = torch.stack(W2V_features, dim = 1)
        
        batch_out, _ = fusion_model(W2V_features.to(device), scores_cm.to(device).float(), use_softmax)
        batch_score = (batch_out[:, 0]).data.cpu().numpy().ravel() 
        i+=1
        print("batch %i of %i (Memory: %.2f of %.2f GiB reserved) (evaluation)"
                  % (
                     i,
                     num_batch,
                     torch.cuda.max_memory_allocated(device) / (2 ** 30),
                     torch.cuda.max_memory_reserved(device) / (2 ** 30),
                     ),
                  end="\r",
                  )
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(utt_id, batch_score.tolist()):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()  
    print("\n",end="\r") 
    print('Scores saved to {}'.format(save_path))

def evaluate_accuracy(dev_loader, model, W2V, device):
    val_loss = 0.0
    num_total = 0.0
    correct=0
    W2V.eval()

    model.eval()
    weight = torch.FloatTensor([0.9, 0.1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    num_batch = len(dev_loader)
    i=0
    with torch.no_grad():
      for _, waveform, labels, scores_cm in dev_loader:
        batch_size = waveform.size(0)
        target = torch.LongTensor(labels).to(device)
        num_total += batch_size

        waveform = waveform.to(device)
        waveform = waveform.float()
        batch_y = labels.view(-1).type(torch.int64).to(device)
        scores_cm = scores_cm.to(device)  

        W2V_features,_ = W2V.extract_features(waveform)
        W2V_features = torch.stack(W2V_features, dim = 1)

        batch_out, _ = model(W2V_features.to(device) , scores_cm.to(device).float())
        pred = batch_out.max(1)[1] 
        correct += pred.eq(target).sum().item()
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        i=i+1
        print("batch %i of %i (Memory: %.2f of %.2f GiB reserved) (validation)"
                  % (
                     i,
                     num_batch,
                     torch.cuda.max_memory_allocated(device) / (2 ** 30),
                     torch.cuda.max_memory_reserved(device) / (2 ** 30),
                     ),
                  end="\r",
                  )
        
    val_loss /= num_total
    test_accuracy = 100. * correct / len(dev_loader.dataset)
    print("\n", end="\r")
    print('Validation accuracy: %.3f %% and loss: %.4f' % (test_accuracy, val_loss))
    return val_loss

def train_epoch(train_loader, model, W2V, optimizer, device):
    num_total = 0.0
    model.train()
    W2V.eval()

    weight = torch.FloatTensor([0.9, 0.1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    num_batch = len(train_loader)
    i=0
    for _, waveform, labels, scores_cm in train_loader:
        batch_size = waveform.size(0)
        num_total += batch_size
        
        waveform = waveform.to(device)
        waveform = waveform.float()
        batch_y = labels.view(-1).type(torch.int64).to(device)

        optimizer.zero_grad()

        W2V_features, _ = W2V.extract_features(waveform)
        W2V_features = torch.stack(W2V_features, dim = 1)
        batch_out, _ = model(W2V_features.to(device) , scores_cm.to(device).float())
        batch_loss = criterion(batch_out, batch_y)     
        
        batch_loss.backward()
        optimizer.step()
        i=i+1
        print("batch %i of %i (Memory: %.2f of %.2f GiB reserved) (training)"
                  % (
                     i,
                     num_batch,
                     torch.cuda.max_memory_allocated(device) / (2 ** 30),
                     torch.cuda.max_memory_reserved(device) / (2 ** 30),
                     ),
                  end="\r",
                  )
    sys.stdout.flush()


import argparse
import os
from utils import reproducibility, extract_eer
from model import Modelo_Fusion, Modelo_Clasificacion
from dataset import fusion_dataset_eval, fusion_dataset_training
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fusion model')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/home/ros/datasets_local', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %  |- LA
    %  |   |- ASVspoof2021_LA_eval/wav
    %  |   |- ASVspoof2019_LA_dev/wav
    %  |- DF
    %      |- ASVspoof2021_DF_eval/wav
    '''
    parser.add_argument('--protocols_path', type=str, default='protocols', help='Change with path to user\'s LA database protocols directory address')
    parser.add_argument("--model_name", "-name", default=None,
                           help="Use other name and not the default one")

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--patience', type=int, default=5, help="Number of periods allowed without improvement after which the learning rate will be reduced (by half).")
    parser.add_argument('--Sc_norm', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']), help="Whether to use the normalized scores, of the antispoofing models, (True) or the unnormalized scores (False)")
    
    

    #model parameters
    parser.add_argument('--FT_size', type=int, default=128, metavar='N',
                    help='embedding size') 
    parser.add_argument('--hidden_size', type=int, default=80, metavar='N',
                    help='hidden_size for the lstm')
    parser.add_argument('--lstm_layers', type=int, default=2, metavar='N',
                    help='Number of lstm layers')
    parser.add_argument('--clasificacion', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to use clasification model or ensembling model')
    parser.add_argument('--train_2019', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='When using the clasification model if we train and val with 2019 or the same datasets as the Ensembling model')
    
    # model save path
    parser.add_argument('--seed', type=int, default=12345678, 
                        help='random seed (default: 1)')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--comment_eval', type=str, default=None,
                        help='Comment to describe the saved scores')
    parser.add_argument('--remake_scores', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='When set to False, an error is returned if the score file already exists and a regeneration attempt is made.')
    
    #Train
    parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model, if the model has been previusly trained and the weights already exist can be set to False')
    
    #Eval
    parser.add_argument('--n_mejores_loss', type=int, default=7, help='save the n-best models')
    parser.add_argument('--average_model', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether average the weight of the n_average_model epochs')
    parser.add_argument('--use_softmax', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether use softmax to normalice scores in evaluation')
    parser.add_argument('--n_average_model', default=5, type=int)

    ############################ Código ############################

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    print(args)
 
    #make experiment reproducible
    reproducibility(args.seed, args)
    
    track = 'LA'
    n_mejores=args.n_mejores_loss
    assert args.n_average_model<args.n_mejores_loss+1, 'average models must be smaller or equal to number of saved epochs'
    if args.use_softmax: 
        if args.comment_eval: args.comment_eval = args.comment_eval + '_softmax'
        else: args.comment_eval = 'softmax'

    # Model name
    if (args.model_name is None):
        # We generate a default name from the parameters if no name has been provided.
        arg_dict = vars(args).copy()
        name = "ModeloFusion_"
        if args.clasificacion: name = "ModeloClasificacion_"
        arg_dict.pop("database_path")
        arg_dict.pop("protocols_path")
        arg_dict.pop("comment_eval")
        arg_dict.pop("use_softmax")
        arg_dict.pop("average_model")
        arg_dict.pop("train")
        arg_dict.pop("remake_scores")
        
        for arg in arg_dict:
            if arg_dict[arg] is not None and arg_dict[arg] != parser.get_default(arg):
                # If the argument has a value other than the default (and is not None) it is appended to the name
                words_arg = arg.split("_")  # We have eliminated the _ 
                capitalized_words = ([words_arg[0]]
                                         + [word.capitalize() for word in words_arg[1:]]
                                         )  # We transform the argument names from arg_red to argRed
                capitalized_words = "".join(capitalized_words)
                name += (capitalized_words[:6] + "-" + str(arg_dict[arg]) + "_")  # we add their value
        args.model_name = name[:-1]  # Delete the last '_'


    model_tag = args.model_name
    model_save_path = os.path.join('models', model_tag)
    print('Model tag: '+ model_tag)

    # Model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    best_save_path = os.path.join(model_save_path, 'best')
    if not os.path.exists(best_save_path):
        os.mkdir(best_save_path)
    
    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    # Model
    model = Modelo_Fusion(args)
    if args.clasificacion: model = Modelo_Clasificacion(args)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters() if param.requires_grad])
    model =model.to(device)
    model.float()
    print('nb_params:',nb_params)

    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
    W2V = bundle.get_model(dl_kwargs={"model_dir": "./models"})
    W2V = W2V.to(device)
    for param in W2V.parameters():
        param.requires_grad = False

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=0.5, verbose=True)
     
    # Datasets 
    ## Training
    if args.train:
        train_set = fusion_dataset_training(dir_databases = args.database_path, name_db = 'ASVspoof2019', track=track, sub_set='dev', file_format='wav', dir_protocols=args.protocols_path, column_names=1, column_labels=4, scores_folder='Scores', norm=args.Sc_norm)
        if args.clasificacion and args.train_2019: train_set = fusion_dataset_training(dir_databases = args.database_path, name_db = 'ASVspoof2019', track=track, sub_set='train', file_format='wav', dir_protocols=args.protocols_path, column_names=1, column_labels=4, scores_folder=None, norm=args.Sc_norm)
        print('no. of training trials', train_set.__len__())
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = 10, shuffle=True,drop_last = True) 
        del train_set  
        ## Validation
        dev_set = fusion_dataset_training(dir_databases = args.database_path, name_db = 'ASVspoof2021', track=track, sub_set='eval', file_format='wav', dir_protocols=args.protocols_path, column_names=1, column_labels=5, scores_folder='Scores', norm=args.Sc_norm)
        if args.clasificacion and args.train_2019: dev_set = fusion_dataset_training(dir_databases = args.database_path, name_db = 'ASVspoof2019', track=track, sub_set='dev', file_format='wav', dir_protocols=args.protocols_path, column_names=1, column_labels=4, scores_folder=None, norm=args.Sc_norm)
        print('no. of validation trials', dev_set.__len__())
        dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=10, shuffle=False)
        del dev_set
    
        ##################### Training and validation #####################
        not_improving=0
        bests=np.ones(n_mejores,dtype=float)*float('inf')
        best_loss=float('inf')
    
        for i in range(n_mejores):
            np.savetxt( os.path.join(best_save_path, 'best_{}.pth'.format(i)), np.array((0,0)))
        for epoch in range(args.epochs):
            print('######## Epoca {} ########'.format(epoch+1))
            train_epoch(train_loader, model, W2V, optimizer, device)
            val_loss = evaluate_accuracy(dev_loader, model, W2V, device)
            scheduler.step(val_loss)
            if val_loss<best_loss:
                best_loss=val_loss
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
                print('New best epoch')
                not_improving=0
            else:
                not_improving+=1
                if not_improving>args.early_stop: break
            for i in range(n_mejores):
                if bests[i]>val_loss:
                    for t in range(n_mejores-1,i,-1):
                        bests[t]=bests[t-1]
                        os.system('mv {}/best_{}.pth {}/best_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                    bests[i]=val_loss
                    torch.save(model.state_dict(), os.path.join(best_save_path, 'best_{}.pth'.format(i)))
                    print('n-best loss:', bests)
                    break
            #torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
        print('Total epochs: ' + str(epoch+1) +'\n')


    print('######## Eval ########')
    if args.average_model:
        sdl=[]
        model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        sd = model.state_dict()
        for i in range(1,args.n_average_model):
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            sd2 = model.state_dict()
            for key in sd:
                sd[key]=(sd[key]+sd2[key])
        for key in sd:
            sd[key]=(sd[key])/args.n_average_model
        model.load_state_dict(sd)
        print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
    else:
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
        print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))

    eval_tracks=['LA', 'DF']
    if args.comment_eval:
        model_tag = model_tag + '_{}'.format(args.comment_eval)

    for tracks in eval_tracks:
        if (not os.path.exists('Scores/{}/{}.txt'.format(tracks, model_tag))) or (args.remake_scores):
            eval_set=fusion_dataset_eval(dir_databases = args.database_path, name_db = 'ASVspoof2021', track=tracks, sub_set='eval', file_format='wav', dir_protocols=args.protocols_path, column_names_eval=1, scores_folder='Scores', norm=args.Sc_norm)
            print('no. of validation trials', eval_set.__len__())
            eval_loader = DataLoader(eval_set, batch_size=64, num_workers=5, shuffle=False)
            produce_evaluation_file(eval_loader, model, W2V, device, 'Scores/{}/{}.txt'.format(tracks, model_tag), use_softmax=args.use_softmax)
        else:
            print('Score file already exists')
    
    for tracks in eval_tracks:
        eer,ts = extract_eer(
            score_file='Scores/{}/{}.txt'.format(tracks, model_tag), 
            metadata='Metadata/ASVspoof2021_{}_eval/trial_metadata.txt'.format(tracks),
            col_names=1,
            col_labels=5,
            phase='eval', 
            col_phase=7
            )
        eer_p,ts = extract_eer(
            score_file='Scores/{}/{}.txt'.format(tracks, model_tag), 
            metadata='Metadata/ASVspoof2021_{}_eval/trial_metadata.txt'.format(tracks),
            col_names=1,
            col_labels=5,
            phase='progress', 
            col_phase=7
            )
        save_path='resultados_fin.txt'
        with open(save_path, 'a+') as fh:
            fh.write('Model: {} track: {} EER: {:.3f} EER_progress: {:.3f}\n'.format(model_tag, tracks, eer*100, eer_p*100))
        fh.close()  
        print('Model: {} track: {} EER: {:.3f} EER_progress: {:.3f}\n'.format(model_tag, tracks, eer*100, eer_p*100))
        torch.cuda.empty_cache()
