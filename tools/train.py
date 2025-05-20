import torch
import datetime, argparse, yaml, os
from models.yolo import YOLOV1
from dataset.voc import VOCDataset
from torch.utils.data import DataLoader, random_split
#from torch.utils.data.dataloader import DataLoader, random_split
from loss.yolov1_loss import YOLOV1Loss
from tqdm import tqdm



def load_config(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def collate_function(data):
    return list(zip(*data))


def validation_epoch(validation_loader, device, use_sigmoid, model, loss_fn):
    running_vloss = 0.0
    with torch.no_grad():
        for idx, (ims, targets, _) in enumerate(tqdm(validation_loader, leave=False)):  # tqdm(
            yolo_targets = torch.cat([target['yolo_targets'].unsqueeze(0).float().to(device) for target in targets], dim=0)
            im = torch.cat([im.unsqueeze(0).float().to(device) for im in ims], dim=0)
            yolo_preds = model(im)
            loss = loss_fn(yolo_preds, yolo_targets, use_sigmoid)
            running_vloss += loss.item()

    return running_vloss


def train_epoch(train_loader, device, use_sigmoid, model, optimizer, loss_fn):
    loss_train = 0.0
    for idx, (ims, targets, _) in enumerate(tqdm(train_loader, leave=False)):  # tqdm(
        # if not load_to_memory:
        yolo_targets = torch.cat([target['yolo_targets'].unsqueeze(0).float().to(device) for target in targets], dim=0)
        im = torch.cat([im.unsqueeze(0).float().to(device) for im in ims], dim=0)
        # else:
        #    yolo_targets = targets
        #    im = ims
        yolo_preds = model(im)
        loss = loss_fn(yolo_preds, yolo_targets, use_sigmoid)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    return loss_train


def training_loop(n_epochs, optimizer, scheduler, model, loss_fn, dataset, save_path, use_sigmoid, device, load_to_memory,
                  batch, num_workers):
    best_vloss = float('inf')
    train_size = int(0.90 * len(dataset))  # 90% for training
    val_size = len(dataset) - train_size  # 10% for validation



    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if load_to_memory:
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                   collate_fn=collate_function)
        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True,
                                   collate_fn=collate_function)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                   collate_fn=collate_function,
                                   num_workers=num_workers, persistent_workers=True)

        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True,
                                   collate_fn=collate_function,
                                   num_workers=num_workers, persistent_workers=True)

    for epoch in range(1, n_epochs + 1):
        model.train(True)
        loss_train = train_epoch(train_loader, device, use_sigmoid, model, optimizer, loss_fn) / len(train_loader)
        scheduler.step()
        model.eval()
        loss_validation = validation_epoch(val_loader, device, use_sigmoid, model, loss_fn) / len(val_loader)
        if epoch % 3 == 0:
            print(f'{datetime.datetime.now()} Epoch {epoch}, Training loss {loss_train}, Validation loss {loss_validation}')
        if loss_validation < best_vloss:
            torch.save(model.state_dict(), save_path)
            best_vloss = loss_validation



def main(args):
    #torch.multiprocessing.set_start_method('fork')
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")
    config = load_config(args)
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']


    model = YOLOV1(im_size=dataset_config['im_size'], num_classes=dataset_config['num_classes'],
                        B=model_config['B'], S=model_config['S'], im_channels=model_config['im_channels'],
                        use_conv=model_config['use_conv'], shrink_network=model_config['shrink']).to(device=device)

    save_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(save_path):
        print('Loading checkpoint as one exists')
        model.load_state_dict(torch.load(save_path, map_location=device))

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    voc = VOCDataset(split='train', im_sets=dataset_config['train_im_sets'], labels=dataset_config['labels'],
                     im_size=dataset_config['im_size'], S=model_config['S'], B=model_config['B'],
                     C=dataset_config['num_classes'], im_channels=model_config['im_channels'], load_to_memory=dataset_config['load_to_memory'])


    criterion = YOLOV1Loss(C=dataset_config['num_classes'], B=model_config['B'], S=model_config['S']).to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=train_config['lr'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['num_epochs'])


    training_loop(n_epochs=train_config['num_epochs'], optimizer=optimizer, scheduler=scheduler, model=model,
                  loss_fn=criterion, dataset=voc, save_path=save_path, use_sigmoid=model_config['use_sigmoid'],
                  device=device, load_to_memory=dataset_config['load_to_memory'], batch=train_config['batch_size'],
                  num_workers=dataset_config['num_workers'])


    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for yolov1 training')
    parser.add_argument('--config', dest='config_path',
                        default='config\\voc.yaml', type=str)
    args = parser.parse_args()
    main(args)