import argparse
import sys
import os
import random
import data_utils_LA
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import torch
from torch import nn
from model import RawNet
from tensorboardX import SummaryWriter


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(path, model, optimizer, epoch, seed, metrics=None):
    checkpoint = {
        "epoch": epoch,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics or {}
    }
    torch.save(checkpoint, path)




def keras_lr_decay(step, decay = 0.0001):
	return 1./(1.+decay*step)

def pad(x, max_len=64600):
    
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    
    return padded_x 

def init_weights(m):
    #print(m)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0001)
    elif isinstance(m, nn.BatchNorm1d):
        pass
    else:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, a=0.01)
        else:		
            print('no weight',m)


def evaluate_accuracy(data_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_meta in data_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x,batch_y)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    fname_list = []
    key_list = []
    sys_id_list = []
    key_list = []
    score_list = []
    with torch.no_grad():
        for batch_x, batch_y, batch_meta in data_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x,batch_y,is_test=True)
            batch_score = (batch_out[:, 1]
                           ).data.cpu().numpy().ravel()

            # add outputs
            fname_list.extend(list(batch_meta[1]))
            key_list.extend(
              ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
            sys_id_list.extend([dataset.sysid_dict_inv[s.item()]
                                for s in list(batch_meta[3])])
            score_list.extend(batch_score.tolist())
        
    with open(save_path, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            if dataset.is_eval:
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))
            else:
                fh.write('{} {}\n'.format(f, cm))
    print('Result saved to {}'.format(save_path))


def train_epoch(data_loader, model, lr, optim, device, epoch, seed, model_save_path=None):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()
    weight = torch.FloatTensor([1.0, 9.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y, batch_meta in data_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x,batch_y)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        
        if ii % 10 == 0:
            avg_loss = running_loss / num_total
            accuracy = (num_correct / num_total) * 100
            print(f"Batch [{ii}/{len(data_loader)}] | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%", flush=True)
            
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        # Frequent periodic checkpoint (every 100 batches)
        if model_save_path and ii % 100 == 0:
            ckpt_path = os.path.join(model_save_path, 'latest_mid_epoch.pth')
            try:
                save_checkpoint(ckpt_path, model, optim, epoch, seed)
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Failed to verify checkpoint at {ckpt_path}")
                else:
                    print("Checkpoint saved successfully")
            except Exception as e:
                print(f"\n❌ CRITICAL ERROR: Could not save checkpoint! ({e})")
                print("Stopping training to prevent data loss. Please check your Google Drive connection.")
                sys.exit(1) # Exit with error code to stop the loop
       
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy




if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASVSpoof2019  model')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
   
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/ASVspoof_database/', help='change this to user\'s full directory address of LA database')
    parser.add_argument('--protocols_path', type=str, default='database/ASVspoof2019_LA_cm_protocols/', help='Change with path to user\'s LA database protocols directory address')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved mdoel')
    parser.add_argument('--track', type=str, default='logical')
    parser.add_argument('--features', type=str, default='Raw_audio')
    parser.add_argument('--is_eval', action='store_true', default=False)
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for model/training')
    parser.add_argument('--data_seed', type=int, default=None,
                        help='seed for data order/randomization (default: same as --seed)')
    parser.add_argument('--checkpoint_interval_epochs', type=int, default=1,
                        help='periodic checkpoint interval in epochs')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='number of samples to use from the dataset (for testing)')
    

    dir_yaml = os.path.splitext('model_config_RawNet2')[0] + '.yaml'

    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)

    
    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    run_seed = args.seed if args.seed is not None else int(parser1['seed'])
    data_seed = args.data_seed if args.data_seed is not None else run_seed
    seed_everything(run_seed)
    print('Run seed:', run_seed)
    print('Data seed:', data_seed)

    # LA and PA
    track = args.track
    
    #Creat Model
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    
    is_logical = (track == 'logical')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    
    transforms = transforms.Compose([
        
        lambda x: pad(x),
        lambda x: Tensor(x)
        
    ])

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  # cuda-0
    
    

    # Dataloader
    dev_set = data_utils_LA.ASVDataset(database_path=args.database_path,protocols_path=args.protocols_path,is_train=False, is_logical=is_logical,
                                    transform=transforms,
                                    feature_name=args.features, is_eval=args.is_eval, eval_part=args.eval_part,
                                    sample_size=args.sample_size)
    dev_gen = torch.Generator()
    dev_gen.manual_seed(data_seed)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, generator=dev_gen, num_workers=2)
    
    #torch.backends.cudnn.enabled = False
    
    # Model Initialization
    if bool(parser1['mg']):
            model_1gpu = RawNet(parser1['model'], device)
            nb_params = sum([param.view(-1).size()[0] for param in model_1gpu.parameters()])
            model =(model_1gpu).to(device)
    else:
        model = RawNet(parser1['model'], device).to(device)
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    
 

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    
    start_epoch = 0
    if args.model_path:
        loaded_obj = torch.load(args.model_path, map_location=device)
        if isinstance(loaded_obj, dict) and 'model_state_dict' in loaded_obj:
            model.load_state_dict(loaded_obj['model_state_dict'])
            if 'optimizer_state_dict' in loaded_obj:
                optimizer.load_state_dict(loaded_obj['optimizer_state_dict'])
            if 'epoch' in loaded_obj:
                start_epoch = loaded_obj['epoch'] + 1
        else:
            model.load_state_dict(loaded_obj)
        print('Model loaded : {}'.format(args.model_path))
        print('Starting from epoch {}'.format(start_epoch))

    if args.eval:
        produce_evaluation_file(dev_set, model, device, args.eval_output)
        sys.exit(0)

    # Dataloader
    train_set = data_utils_LA.ASVDataset(database_path=args.database_path,protocols_path=args.protocols_path,is_train=True, is_logical=is_logical, transform=transforms,
                                      feature_name=args.features, sample_size=args.sample_size)
    train_gen = torch.Generator()
    train_gen.manual_seed(data_seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=train_gen, num_workers=2)

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_acc = 0.0
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nStart training epoch {epoch}")
        train_loss, train_acc = train_epoch(train_loader, model, args.lr, optimizer, device, epoch, run_seed, model_save_path)
        valid_acc = evaluate_accuracy(dev_loader, model, device)
        
        writer.add_scalar('train_accuracy', train_acc, epoch)
        writer.add_scalar('valid_accuracy', valid_acc, epoch)
        writer.add_scalar('loss', train_loss, epoch)
        
        print('\nEpoch {} - Loss: {:.4f} - Train Acc: {:.2f}% - Valid Acc: {:.2f}%'.format(
            epoch, train_loss, train_acc, valid_acc))
        
        if valid_acc > best_acc:
            print('--- Best model found at epoch {} ---'.format(epoch))
            best_acc = valid_acc
            
        print('*'*50)
        
        # Save epoch checkpoint
        try:
            epoch_ckpt_path = os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch))
            save_checkpoint(epoch_ckpt_path, model, optimizer, epoch, run_seed, {
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'valid_accuracy': valid_acc,
                'best_accuracy': best_acc,
            })
            if not os.path.exists(epoch_ckpt_path):
                raise FileNotFoundError(f"Failed to verify epoch checkpoint at {epoch_ckpt_path}")
            print(f"Epoch {epoch} checkpoint saved successfully")
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: Could not save epoch checkpoint! ({e})")
            sys.exit(1)

        if (epoch + 1) % args.checkpoint_interval_epochs == 0:
            try:
                periodic_ckpt_path = os.path.join(model_save_path, 'checkpoint_ep{:03d}.pth'.format(epoch + 1))
                save_checkpoint(periodic_ckpt_path, model, optimizer, epoch, run_seed, {
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'valid_accuracy': valid_acc,
                    'best_accuracy': best_acc,
                })
                if not os.path.exists(periodic_ckpt_path):
                    raise FileNotFoundError(f"Failed to verify periodic checkpoint at {periodic_ckpt_path}")
                print(f"Periodic checkpoint for epoch {epoch+1} saved successfully")
            except Exception as e:
                print(f"\n❌ CRITICAL ERROR: Could not save periodic checkpoint! ({e})")
                sys.exit(1)
