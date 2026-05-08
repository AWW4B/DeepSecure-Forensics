import argparse
import sys
import os
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


def keras_lr_decay(step, decay=0.0001):
    return 1. / (1. + decay * step)


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0001)
    elif isinstance(m, nn.BatchNorm1d):
        pass
    else:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, a=0.01)
        else:
            print('no weight', m)


def print_system_info(device):
    print('=' * 50)
    print('SYSTEM INFO')
    print('=' * 50)
    print('Python  : {}'.format(sys.version.split()[0]))
    print('PyTorch : {}'.format(torch.__version__))
    print('Device  : {}'.format(device.upper()))
    if torch.cuda.is_available():
        print('GPU     : {}'.format(torch.cuda.get_device_name(0)))
        print('VRAM    : {:.1f} GB'.format(
            torch.cuda.get_device_properties(0).total_memory / 1024**3))
        print('CUDA    : {}'.format(torch.version.cuda))
    else:
        print('GPU     : Not available (running on CPU)')
    print('=' * 50)


def evaluate_accuracy(data_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    fname_list = []
    key_list = []
    sys_id_list = []
    key_list = []
    score_list = []
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, batch_y, is_test=True)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

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


def train_epoch(train_loader, model, lr, optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()
    weight = torch.FloatTensor([1.0, 9.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y, batch_meta in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, batch_y)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct / num_total) * 100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy


def save_checkpoint(epoch, model, optimizer, running_loss, train_accuracy,
                    valid_accuracy, save_path):
    """Save a full training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
        'train_accuracy': train_accuracy,
        'valid_accuracy': valid_accuracy,
    }
    torch.save(checkpoint, save_path)
    print('Checkpoint saved to {}'.format(save_path))


def load_checkpoint(model_path, model, optimizer, device):
    """
    Load a checkpoint. Supports both:
      - Full checkpoints (dict with 'model_state_dict')
      - Legacy weights-only .pth files
    Returns the epoch to resume from (0 if legacy/not found).
    """
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print('Full checkpoint loaded from epoch {}'.format(checkpoint['epoch']))
        print('  -> train_acc: {:.2f}  valid_acc: {:.2f}  loss: {:.4f}'.format(
            checkpoint.get('train_accuracy', 0),
            checkpoint.get('valid_accuracy', 0),
            checkpoint.get('loss', 0)))
    else:
        # Legacy: plain state_dict
        model.load_state_dict(checkpoint)
        start_epoch = 0
        print('Legacy weights-only checkpoint loaded.')
    return start_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASVSpoof2019 model')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model checkpoint (.pth) to load')
    parser.add_argument('--database_path', type=str,
                        default='D:/i230079/dataset/LA/LA/',
                        help='Full path to ASVspoof LA database directory')
    parser.add_argument('--protocols_path', type=str,
                        default='D:/i230079/dataset/LA/LA/ASVspoof2019_LA_cm_protocols/',
                        help='Path to LA database protocols directory')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--track', type=str, default='logical')
    parser.add_argument('--features', type=str, default='Raw_audio')
    parser.add_argument('--is_eval', action='store_true', default=False)
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from --model_path checkpoint')

    dir_yaml = os.path.splitext('model_config_RawNet2')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)

    np.random.seed(parser1['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists('models'):
        os.mkdir('models')

    args = parser.parse_args()

    track = args.track

    # Build model save directory
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    is_logical = (track == 'logical')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    print('Checkpoints will be saved to: {}'.format(
        os.path.abspath(model_save_path)))

    # Transforms
    data_transforms = transforms.Compose([
        lambda x: pad(x),
        lambda x: Tensor(x)
    ])

    # Device + system info (printed first)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_system_info(device)

    # Dev/eval dataloader
    dev_set = data_utils_LA.ASVDataset(
        database_path=args.database_path,
        protocols_path=args.protocols_path,
        is_train=False,
        is_logical=is_logical,
        transform=data_transforms,
        feature_name=args.features,
        is_eval=args.is_eval,
        eval_part=args.eval_part)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)

    # Model
    if bool(parser1['mg']):
        model_1gpu = RawNet(parser1['model'], device)
        nb_params = sum([param.view(-1).size()[0]
                         for param in model_1gpu.parameters()])
        model = model_1gpu.to(device)
    else:
        model = RawNet(parser1['model'], device).to(device)
        nb_params = sum([param.view(-1).size()[0]
                         for param in model.parameters()])

    print('nb_params: {}'.format(nb_params))

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load checkpoint if provided
    start_epoch = 0
    if args.model_path:
        start_epoch = load_checkpoint(
            args.model_path, model, optimizer, device)
        print('Model loaded: {}'.format(args.model_path))
        if not args.resume:
            start_epoch = 0  # reset epoch counter unless --resume is set

    # Evaluation-only mode
    if args.eval:
        produce_evaluation_file(dev_set, model, device, args.eval_output)
        sys.exit(0)

    # Training dataloader
    train_set = data_utils_LA.ASVDataset(
        database_path=args.database_path,
        protocols_path=args.protocols_path,
        is_train=True,
        is_logical=is_logical,
        transform=data_transforms,
        feature_name=args.features)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)

    # TensorBoard writer
    writer = SummaryWriter('logs/{}'.format(model_tag))

    best_acc = 0.0

    for epoch in range(start_epoch, args.num_epochs):
        running_loss, train_accuracy = train_epoch(
            train_loader, model, args.lr, optimizer, device)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)

        # TensorBoard logging
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)

        print('\nEpoch {} | Loss: {:.4f} | Train Acc: {:.2f}% | Val Acc: {:.2f}%'.format(
            epoch, running_loss, train_accuracy, valid_accuracy))

        # ── Save full checkpoint every epoch ──────────────────────────────
        ckpt_path = os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch))
        save_checkpoint(epoch, model, optimizer, running_loss,
                        train_accuracy, valid_accuracy, ckpt_path)

        # ── Save best model separately ────────────────────────────────────
        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            best_ckpt_path = os.path.join(model_save_path, 'best_model.pth')
            save_checkpoint(epoch, model, optimizer, running_loss,
                            train_accuracy, valid_accuracy, best_ckpt_path)
            print('*** Best model updated at epoch {} (val acc: {:.2f}%) ***'.format(
                epoch, valid_accuracy))

        print('*' * 50)
        print('Current best val acc: {:.2f}%'.format(best_acc))
        print('*' * 50)

    writer.close()
    print('Training complete. Best val acc: {:.2f}%'.format(best_acc))
    print('Checkpoints saved in: {}'.format(os.path.abspath(model_save_path)))