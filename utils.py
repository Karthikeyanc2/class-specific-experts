from splits import splits_2020 as splits

import random
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as tf


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


dataset_info = {
    'mnist': {
        'mean': [
            [0.13591310381889343],
            [0.14117178320884705],
            [0.13846933841705322],
            [0.13845889270305634],
            [0.13719801604747772]
        ],
        'std': [
            [0.2922218441963196],
            [0.2987239360809326],
            [0.29500940442085266],
            [0.29548293352127075],
            [0.29417088627815247]
        ],
        'path': 'datasets/MNIST',
        'im_size': 32,
        'im_channels': 1,
        'flip': 0,
        'rotate': 0,
        'scale_min': 0.7,
        'batch_size': 128,
        'num_total_classes': 10,
        'num_known_classes': 6,
        'num_unknown_classes': 4,
        'known_classes': [sorted(s) for s in splits['mnist']],
        'unknown_classes': [sorted(list(set(range(10)) - set(split))) for split in splits['mnist']],
        'learning_rate': 0.01,
        'step_size': 35,
        'step_gamma': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-6,
        'dropout': 0.2,
        'num_epochs': 70,
    },
    'svhn': {
        'mean': [
            [0.43773889541625977, 0.4439214766025543, 0.47282674908638],
            [0.43773889541625977, 0.4439214766025543, 0.47282674908638],
            [0.43757709860801697, 0.4435049295425415, 0.4727678894996643],
            [0.4392078220844269, 0.4453091621398926, 0.4738246500492096],
            [0.4378757178783417, 0.4438391327857971, 0.4729028642177582]
        ],
        'std': [
            [0.19755372405052185, 0.20053085684776306, 0.19663006067276],
            [0.19755372405052185, 0.20053085684776306, 0.19663006067276],
            [0.1986754834651947, 0.20168840885162354, 0.19715414941310883],
            [0.19690780341625214, 0.200160413980484, 0.19706986844539642],
            [0.1983148157596588, 0.20132160186767578, 0.19709119200706482]
        ],
        'path': 'datasets/SVHN',
        'im_size': 32,
        'im_channels': 3,
        'flip': 0,
        'rotate': 0,
        'scale_min': 0.7,
        'batch_size': 128,
        'num_total_classes': 10,
        'num_known_classes': 6,
        'num_unknown_classes': 4,
        'known_classes': [sorted(s) for s in splits['svhn']],
        'unknown_classes': [sorted(list(set(range(10)) - set(split))) for split in splits['svhn']],
        'learning_rate': 0.01,
        'step_size': 50,
        'step_gamma': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-6,
        'dropout': 0.2,
        'num_epochs': 75,
    },
    'cifar10': {
        'mean': [
            [0.48985224962234497, 0.4806106388568878, 0.44241583347320557],
            [0.48985224962234497, 0.4806106388568878, 0.44241583347320557],
            [0.4898015856742859, 0.46766477823257446, 0.42540884017944336],
            [0.49295806884765625, 0.49064207077026367, 0.471828818321228],
            [0.48378467559814453, 0.46913525462150574, 0.414938747882843]
        ],
        'std': [
            [0.24751833081245422, 0.24498571455478668, 0.2642909288406372],
            [0.24751833081245422, 0.24498571455478668, 0.2642909288406372],
            [0.2515733242034912, 0.2480219155550003, 0.25724688172340393],
            [0.25241565704345703, 0.25051620602607727, 0.2710384726524353],
            [0.24325338006019592, 0.23940759897232056, 0.252301961183548]
        ],
        'path': 'datasets/CIFAR10',
        'im_size': 32,
        'im_channels': 3,
        'flip': 0.5,
        'rotate': 10,
        'scale_min': 0.8,
        'batch_size': 128,
        'num_total_classes': 10,
        'num_known_classes': 6,
        'num_unknown_classes': 4,
        'known_classes': [sorted(s) for s in splits['cifar10']],
        'unknown_classes': [sorted(list(set(range(10)) - set(split))) for split in splits['cifar10']],
        'learning_rate': 0.01,
        'step_size': 150,
        'step_gamma': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-6,
        'dropout': 0.2,
        'num_epochs': 200,
    },
    'cifar100-10': {
        'mean': [
            [0.48585033416748047, 0.47123515605926514, 0.4300920367240906],
            [0.4854668974876404, 0.4645322263240814, 0.42184510827064514],
            [0.4854668974876404, 0.4645322263240814, 0.42184510827064514],
            [0.4778749644756317, 0.46086767315864563, 0.4121752083301544],
            [0.496433287858963, 0.5063987970352173, 0.5172167420387268]
        ],
        'std': [
            [0.25042781233787537, 0.24791552126407623, 0.25896766781806946],
            [0.253011018037796, 0.2509117126464844, 0.26277944445610046],
            [0.253011018037796, 0.2509117126464844, 0.26277944445610046],
            [0.24693456292152405, 0.24221321940422058, 0.2545713186264038],
            [0.25988835096359253, 0.25759416818618774, 0.27456164360046387]
        ],
        'path': 'datasets/CIFAR100',
        'im_size': 32,
        'im_channels': 3,
        'flip': 0.5,
        'rotate': 10,
        'scale_min': 0.8,
        'batch_size': 128,
        'num_total_classes': 100,
        'num_known_classes': 4,
        'num_unknown_classes': 10,
        'known_classes': [sorted(s) for s in splits['cifar100']],
        'unknown_classes': [sorted(s) for s in splits['cifar100-10']],
        'learning_rate': 0.01,
        'step_size': 150,
        'step_gamma': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-6,
        'dropout': 0.2,
        'num_epochs': 200,
    },
    'cifar100-50': {
        'mean': [
            [0.48585033416748047, 0.47123515605926514, 0.4300920367240906],
            [0.4854668974876404, 0.4645322263240814, 0.42184510827064514],
            [0.4854668974876404, 0.4645322263240814, 0.42184510827064514],
            [0.4778749644756317, 0.46086767315864563, 0.4121752083301544],
            [0.496433287858963, 0.5063987970352173, 0.5172167420387268]
        ],
        'std': [
            [0.25042781233787537, 0.24791552126407623, 0.25896766781806946],
            [0.253011018037796, 0.2509117126464844, 0.26277944445610046],
            [0.253011018037796, 0.2509117126464844, 0.26277944445610046],
            [0.24693456292152405, 0.24221321940422058, 0.2545713186264038],
            [0.25988835096359253, 0.25759416818618774, 0.27456164360046387]
        ],
        'path': 'datasets/CIFAR100',
        'im_size': 32,
        'im_channels': 3,
        'flip': 0.5,
        'rotate': 10,
        'scale_min': 0.8,
        'batch_size': 128,
        'num_total_classes': 100,
        'num_known_classes': 4,
        'num_unknown_classes': 50,
        'known_classes': [sorted(s) for s in splits['cifar100']],
        'unknown_classes': [sorted(s) for s in splits['cifar100-50']],
        'learning_rate': 0.01,
        'step_size': 150,
        'step_gamma': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-6,
        'dropout': 0.2,
        'num_epochs': 200,
    },
    'tiny_imagenet': {
        'mean': [
            [0.4600285291671753, 0.44064581394195557, 0.3940916061401367],
            [0.48554331064224243, 0.4397341310977936, 0.37665948271751404],
            [0.4799080193042755, 0.44145429134368896, 0.3857525587081909],
            [0.4711034595966339, 0.43554022908210754, 0.38351908326148987],
            [0.47205010056495667, 0.44500303268432617, 0.39388570189476013]
        ],
        'std': [
            [0.27523180842399597, 0.2688269019126892, 0.28100213408470154],
            [0.274737685918808, 0.260355144739151, 0.2754135727882385],
            [0.2722342014312744, 0.2620832920074463, 0.26948025822639465],
            [0.2780453562736511, 0.26721030473709106, 0.28439322113990784],
            [0.2758927345275879, 0.26615774631500244, 0.2808636426925659]
        ],
        'path': 'datasets/tiny-imagenet-200',
        'im_size': 64,
        'im_channels': 3,
        'flip': 0.5,
        'rotate': 20,
        'scale_min': 0.7,
        'batch_size': 128,
        'num_total_classes': 200,
        'num_known_classes': 20,
        'num_unknown_classes': 180,
        'known_classes': [sorted(s) for s in splits['tiny_imagenet']],
        'unknown_classes': [sorted(list(set(range(200)) - set(split))) for split in splits['tiny_imagenet']],
        'learning_rate': 0.0001,
        'step_size': 500,
        'step_gamma': 0.1,
        'momentum': 0.9,
        'weight_decay': 10e-4,
        'dropout': 0.3,
        'num_epochs': 800,
    },
    'flir2': {
        'mean': [
            [0.5849854946136475, 0.5849854946136475, 0.5849854946136475],
            [0.5450783371925354, 0.5450783371925354, 0.5450783371925354],
            [0.5828297138214111, 0.5828297138214111, 0.5828297138214111],
            [0.6080410480499268, 0.6080410480499268, 0.6080410480499268],
            [0.5798013806343079, 0.5798013806343079, 0.5798013806343079]
        ],
        'std': [
            [0.18316762149333954, 0.18316762149333954, 0.18316762149333954],
            [0.17467668652534485, 0.17467668652534485, 0.17467668652534485],
            [0.18367813527584076, 0.18367813527584076, 0.18367813527584076],
            [0.17494623363018036, 0.17494623363018036, 0.17494623363018036],
            [0.18296436965465546, 0.18296436965465546, 0.18296436965465546]
        ],
        'path': 'datasets/FLIR2',
        'im_size': 32,
        'im_channels': 3,
        'flip': 0.5,
        'rotate': 10,
        'scale_min': 0.8,
        'batch_size': 128,
        'num_total_classes': 12,
        'num_known_classes': 4,
        'num_unknown_classes': 8,
        'known_classes': [sorted(s) for s in splits['flir2']],
        'unknown_classes': [sorted(list(set(range(12)) - set(split))) for split in splits['flir2']],
        'learning_rate': 0.01,
        'step_size': 150,
        'step_gamma': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-6,
        'dropout': 0.2,
        'num_epochs': 200,
    },
    'outdoor': {
        'mean': [[0.5220552682876587, 0.5220552682876587, 0.5220552682876587]],
        'std': [[0.23996512591838837, 0.23996512591838837, 0.23996512591838837]],
        'path': 'datasets/OUTDOOR',
        'im_size': 32,
        'im_channels': 3,
        'flip': 0.5,
        'rotate': 10,
        'scale_min': 0.8,
        'batch_size': 128,
        'num_total_classes': 6,
        'num_known_classes': 3,
        'num_unknown_classes': 3,
        'known_classes': [sorted(s) for s in splits['outdoor']],
        'unknown_classes': [sorted(list(set(range(6)) - set(split))) for split in splits['outdoor']],
        'learning_rate': 0.01,
        'step_size': 150,
        'step_gamma': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-6,
        'dropout': 0.2,
        'num_epochs': 200,
    }
}


def create_known_classes_mapping(details, trial_num):
    mapping = [None for _ in range(details["num_total_classes"])]
    known_classes = np.asarray(details['known_classes'][trial_num])
    known_classes.sort()
    for i, num in enumerate(known_classes):
        mapping[num] = i
    return mapping


def get_dataloader(dataset_name, details, trial_num, split='train', classes='known'):
    train = split == 'train'
    if train:
        transform = tf.Compose([
            tf.Resize((details["im_size"], details["im_size"])),
            tf.RandomResizedCrop(details["im_size"], scale=(details["scale_min"], 1.0)),
            tf.RandomHorizontalFlip(details["flip"]),
            tf.RandomRotation(details["rotate"]),
            tf.ToTensor(),
            tf.Normalize(details["mean"][trial_num], details["std"][trial_num])
        ])
    else:
        transform = tf.Compose([
            tf.Resize((details["im_size"], details["im_size"])),
            tf.ToTensor(),
            tf.Normalize(details["mean"][trial_num], details["std"][trial_num])
        ])

    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root=details["path"],
            train=train,
            transform=transform,
            download=True
        )
    elif dataset_name == 'svhn':
        dataset = torchvision.datasets.SVHN(
            root=details["path"],
            split='train' if train else 'test',
            transform=transform,
            download=True
        )
    elif dataset_name == 'cifar10' or \
            ('cifar100' in dataset_name and train) or \
            ('cifar100' in dataset_name and not train and classes == 'known'):
        dataset = torchvision.datasets.CIFAR10(
            root=details["path"],
            train=train,
            transform=transform,
            download=True
        )
    elif 'cifar100' in dataset_name:
        dataset = torchvision.datasets.CIFAR100(
            root=details["path"],
            train=train,
            transform=transform,
            download=True
        )
    elif dataset_name == 'tiny_imagenet' or dataset_name == 'flir2' or dataset_name == 'outdoor':
        dataset = torchvision.datasets.ImageFolder(
            root=details["path"] + '/train' if train else details["path"] + '/val',
            transform=transform
        )
    else:
        raise AttributeError('Sorry unimplemented dataset requested')

    all_targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
    classes = details['known_classes'][trial_num] if classes == 'known' else details['unknown_classes'][trial_num]
    indices = list({x for i in classes for x in list(np.where(np.asarray(all_targets) == i)[0])})
    current_dataset = torch.utils.data.Subset(dataset, indices)

    loader = torch.utils.data.DataLoader(
        current_dataset,
        batch_size=details['batch_size'],
        shuffle=train,
        num_workers=8,
        drop_last=train,
        generator=None if train else torch.Generator()
    )

    return loader
