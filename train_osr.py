import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm

import evaluation
from utils import seed_torch, dataset_info, get_dataloader, create_known_classes_mapping


class Model(nn.Module):
    def __init__(self, num_classes, num_features, num_channels, im_size, dropout, learnt_centers, device):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Dropout2d(dropout)
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),
                nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),
                nn.Conv2d(128, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),
                nn.Dropout2d(dropout),
                nn.Flatten(),
                nn.Linear(2 * im_size * im_size, num_features)
            ) for _ in range(num_classes)
        ])

        self.classifiers = nn.ModuleList([nn.Linear(num_features, num_classes) for _ in range(num_classes)])

        self.centers = nn.Parameter(torch.zeros(num_classes, num_features).double().to(device),
                                    requires_grad=learnt_centers)

    def forward(self, x):
        encoded = self.encoder(x)  # batch_size x encoded_size
        heads = torch.concat([head(encoded).unsqueeze(0) for head in self.heads], dim=0).permute(1, 0, 2)  # batch_size, num_heads x num_features
        logits = torch.concat([classifier(heads[:, i]).unsqueeze(0) for i, classifier in enumerate(self.classifiers)], dim=0).permute(1, 0, 2)  # batch_size, num_heads x num_classes
        distances = torch.sqrt(torch.square(heads.double() - self.centers.expand(len(x), *self.centers.shape)).sum(dim=2))  # batch_size, num_classes_or_heads

        return heads, logits, distances


def loss_function(logits, distances, targets, alw, olw):
    num_classes = len(logits[0])
    ce_losses = []

    for i in range(num_classes):  # num_heads
        ce_losses.append(F.cross_entropy(logits[:, i], targets))

    ce_loss = sum(ce_losses) / len(ce_losses)
    anchor_loss = alw * torch.sqrt(distances[range(len(targets)), targets]).mean()
    non_gt = torch.Tensor(
        [[i for i in range(num_classes) if targets[x] != i] for x in range(len(distances))]).long().cuda()
    others = torch.gather(distances, 1, non_gt)

    origin_loss = olw * torch.log(1 + torch.exp(-others)).mean()
    return ce_loss + anchor_loss + origin_loss


def train(model, train_loader, optimizer, scheduler, mapping, loss_function, loss_additional_args, num_epochs, device):
    model.train()

    total = 0
    correct = 0
    total_loss = 0

    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(total=len(train_loader))
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            targets = torch.Tensor([mapping[x] for x in targets]).long().to(device)
            heads, logits, distances = model(images)
            loss = loss_function(logits, distances, targets, **loss_additional_args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(torch.mean(torch.softmax(logits, dim=2), dim=1) * torch.softmax(-distances, dim=1), dim=1)

            correct += predictions.eq(targets).sum().detach().item()
            total += images.size(0)

            total_loss += loss.detach().item() * images.size(0)

            pbar.update(1)

        accuracy = 100 * correct / total
        training_loss = total_loss / total

        pbar.set_description('LR: %.4f | Epoch: %d | Training accuracy: %.2f | Training Loss: %.3f'
                             % (scheduler.get_last_lr()[0], epoch, accuracy, training_loss))
        pbar.close()

        scheduler.step()


def evaluate(model, test_known_loader, test_unknown_loader, mapping, device):
    if type(model) is str:
        state_dict = torch.load(model)
        model = Model(**state_dict['model_args'], device=device).to(device)
        model.load_state_dict(state_dict['model'])
    model.eval()
    with torch.no_grad():
        open_set_scores = []
        correct_flag = []

        for images, targets in test_known_loader:
            images, targets = images.to(device), targets.to(device)
            targets = torch.Tensor([mapping[x] for x in targets]).long().to(device)
            heads, logits, distances = model(images)
            predicted_classes = torch.argmax(torch.mean(torch.softmax(logits, dim=2), dim=1) * torch.softmax(-distances, dim=1), dim=1)
            correct_flag.append(predicted_classes.eq(targets))

            open_set_scores.append((1 - torch.softmax(-distances, dim=1)).min(dim=1)[0])

        open_set_scores_known_set = torch.concat(open_set_scores).cpu().numpy()
        correct_flag = torch.concat(correct_flag).cpu().numpy()

        open_set_scores = []

        for images, _ in test_unknown_loader:
            heads, logits, distances = model(images.to(device))
            open_set_scores.append((1 - torch.softmax(-distances, dim=1)).min(dim=1)[0])

        open_set_scores_unknown_set = torch.concat(open_set_scores).cpu().numpy()

    return evaluation.metrics(-open_set_scores_known_set, -open_set_scores_unknown_set, correct_flag)['Bas']


def main(args, trials=None):
    print(args)
    seed = 1000

    dataset_name = args['dataset']
    base_path = './runs'
    if not os.path.isdir(base_path): os.mkdir(base_path)
    base_path += '/' + ';'.join([f"{k}={v}" for k, v in args.items() if k not in ['dataset', 'eval', 'device']])
    if not os.path.isdir(base_path): os.mkdir(base_path)

    dataset_details = dataset_info[dataset_name]
    torch.cuda.set_device(args['device'])
    device = torch.cuda.current_device()

    loss_additional_args = {'alw': args['alw'], 'olw': args['olw']}

    results = []

    for trial_num in trials or range(len(dataset_details['known_classes'])):

        seed_torch(seed)

        train_loader = get_dataloader(dataset_name, dataset_details, trial_num, split='train',classes='known')
        test_known_loader = get_dataloader(dataset_name, dataset_details, trial_num, split='test', classes='known')
        test_unknown_loader = get_dataloader(dataset_name, dataset_details, trial_num, split='test', classes='unknown')

        mapping = create_known_classes_mapping(dataset_details, trial_num)

        model_path = base_path + '/' + ('cifar100' if 'cifar100-' in dataset_name else dataset_name) + '_trial_' + str(trial_num) + '.pt'
        num_epochs = dataset_details['num_epochs']

        if not args['eval']:
            model_args = dict(
                num_classes=dataset_details['num_known_classes'],
                num_features=args['nf'],
                num_channels=dataset_details['im_channels'],
                im_size=dataset_details['im_size'],
                dropout=dataset_details['dropout'],
                learnt_centers=not args['ac']
            )

            model = Model(**model_args, device=device).to(device)

            print('Num parameters: ', sum([p.numel() for p in model.parameters()]))

            if dataset_name == "tiny_imagenet":
                optimizer = torch.optim.Adam(model.parameters(), lr=dataset_details['learning_rate'],
                                             weight_decay=dataset_details['weight_decay'])
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=dataset_details['learning_rate'],
                                            momentum=dataset_details['momentum'],
                                            weight_decay=dataset_details['weight_decay'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=dataset_details['step_size'],
                                                        gamma=dataset_details['step_gamma'])

            train(model, train_loader, optimizer, scheduler, mapping, loss_function, loss_additional_args, num_epochs, device)
            torch.save({'model': model.state_dict(), 'model_args': model_args}, model_path)

        results.append(evaluate(model_path, test_known_loader, test_unknown_loader, mapping, device))

    df = pd.DataFrame(results)
    mean_results = df.mean()
    mean_results.name = 'mean'
    df = df.append(mean_results)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv(base_path + '/' + dataset_name + '.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training")

    parser.add_argument('--dataset', type=str, default='cifar10', help="mnist | svhn | cifar10 | cifar100-10 | cifar100-50 | tiny_imagenet | flir2 | outdoor")
    parser.add_argument('--nf', type=int, default=12, help="number of features in latent space")
    parser.add_argument('--ac', action='store_true', help="anchored centers")
    parser.add_argument('--olw', type=float, default=2, help="weight for origin loss")
    parser.add_argument('--alw', type=float, default=0.1, help="weight for anchor loss")
    parser.add_argument('--eval', action='store_true', help="perform only evaluation")
    parser.add_argument('--device', type=int, default=0, help="0 .. n")

    args = vars(parser.parse_args())

    main(args)
