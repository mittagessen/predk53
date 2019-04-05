import os
import click

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage, Loss
from ignite.handlers import ModelCheckpoint, TerminateOnNan
from ignite.contrib.handlers import ProgressBar

from predk53.model import Darknet53

@click.group()
def cli():
    pass

@cli.command()
@click.option('-n', '--name', default=None, help='prefix for checkpoint file names')
@click.option('-l', '--lrate', default=1e-4, show_default=True, help='initial learning rate')
@click.option('-e', '--weight-decay', show_default=True, default=0.0, help='Weight decay')
@click.option('-w', '--workers', default=0, show_default=True, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', show_default=True, help='pytorch device')
@click.option('-t', '--train', default='train', show_default=True, help='training set location')
@click.option('-v', '--validation', default='val', show_default=True, help='validation set location')
@click.option('-b', '--batch', default=256, show_default=True, help='minibatch size')
def train(name, lrate, weight_decay, workers, device, train, validation, batch):

    print('model output name: {}'.format(name))

    model = Darknet53()
    optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)

    traindir = os.path.join(data, 'train')
    valdir = os.path.join(data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([transforms.RandomResizedCrop(224),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             normalize]))


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device, non_blocking=True)
    metrics = {
        'accuracy': CategoricalAccuracy(),
        'precision': Precision(),
        'recall': Recall(),
        'nll': Loss(criterion)
    }
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    ckpt_handler = ModelCheckpoint('.', name, save_interval=1, n_saved=10, require_empty=False)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'nll')

    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(trainer, ['nll'])

    evaluator.add_event_handler(Events.COMPLETED, est_handler)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=ckpt_handler, to_save={'net': model})
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=TerminateOnNan())

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_data_loader)
        metrics = evaluator.state.metrics
        progress_bar.log_message('eval results - epoch {} loss: {:.4f} accuracy: {:.4f} recall: {:.4f} precision {:.4f}'.format(engine.state.epoch,
                                                                                                                   metrics['nll'],
                                                                                                                   metrics['accuracy'],
                                                                                                                   metrics['recall'],
                                                                                                                   metrics['precision']))
    trainer.run(train_data_loader, max_epochs=1000)
