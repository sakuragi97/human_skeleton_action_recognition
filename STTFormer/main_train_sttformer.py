# python main_train_sttformer.py --num_epoch 3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import sys
import os
from pathlib import Path
import datetime
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tensorboardX import SummaryWriter
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from utils.ntu_dataset import NTUDataset
from utils.base_dataset import BaseDataset
from utils.ntu_pipeline import NTU2Feeder
from model.sttformer import Model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', default='/home/yas50454/datasets/NTU_Data/NTU_60/ntu60_3danno.pkl', type=str, help='import annotation pickle file')
    parser.add_argument('--split', default='xsub', type=str, help='x_sub/x_view/xset')
    parser.add_argument('--output', default=f"exp_{timestamp}", type=str, help='choose directory for outputs')

    # Model    
    parser.add_argument('--device', default=0, type=int, help='GPU device (int)')
    parser.add_argument('--seed', default=None, type=int, help='choose manual seed')
    parser.add_argument('--model_path', default=None, type=str, help='model path')

    # NTUDataset
    parser.add_argument('--window_size', default=120, type=int, help='sequence lenght')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='number worker for the DataLoader')
    
    # optim
    parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0004, nargs='+', help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay for optimizer')
    parser.add_argument('--step', type=int, default=[60,80], help='step for optimizer')
    parser.add_argument('--warm_up_epoch', type=int, default=5, help='warm up epoch for optimizer')
    parser.add_argument('--num_epoch', type=int, default=6, help='number of epochs for training')
    

    return parser

def load_model(device: int, model_path: str=None):
    output_device = device # GPU device

    config = [[64,  64,  16], [64,  64,  16], 
            [64,  128, 32], [128, 128, 32],
            [128, 256, 64], [256, 256, 64], 
            [256, 256, 64], [256, 256, 64]]
    
    sttformer_model = Model(len_parts=6,
              num_frames=120,
              num_joints=25,
              num_classes=60,
              num_heads=3,
              num_channels=3,
              num_persons=2,
              kernel_size=[3,5],
              use_pes=True,
              config=config).cuda(output_device)

    sttformer_model = nn.DataParallel(sttformer_model,
                                      device_ids=[output_device,output_device+1,output_device+2,output_device+3])

    if model_path:
        sttformer_model.load_state_dict(torch.load(f=model_path, map_location='cuda:0'))
        print(f"**** Model loaded successfully from {model_path} **** ")
    return sttformer_model

def load_dataset(ann_file: str,
                 split: str='xsub',
                 window_size: int=120):
    ann_file = ann_file
    split_type = f"{split}_train"
    ntu2feeder = NTU2Feeder(window_size=120)
    pipeline = torchvision.transforms.Compose([ntu2feeder])
    ntu_feeder_train_dataset = NTUDataset(ann_file,
                                        pipeline=pipeline,
                                        split=split_type,
                                        num_classes=60,
                                        multi_class=True,
                                        augmentation=None)
    split_type = f"{split}_val"
    ntu_feeder_test_dataset = NTUDataset(ann_file,
                                        pipeline=pipeline,
                                        split=split_type,
                                        num_classes=60,
                                        multi_class=True,
                                        augmentation=None)
    
    return ntu_feeder_train_dataset, ntu_feeder_test_dataset

def print_model_summary(model: torch.nn.Module,
                        input_size):
    summary(model, input_size)
    
def adjust_learning_rate(epoch, optimizer, base_lr, warm_up_epoch, lr_decay_rate,step):
    # print(f"adjust learning rate, using warm up, epoch: {warm_up_epoch}")
    if epoch < warm_up_epoch:
        lr = base_lr * (epoch + 1) / warm_up_epoch
    else:
        lr = base_lr * ( lr_decay_rate ** np.sum(epoch >= np.array(step)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


### Training step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):
    train_loss, train_acc = 0, 0

    model.train()
    
    for batch, batch_item in enumerate(dataloader):
        X = batch_item['keypoint']
        y = batch_item['label']
        with torch.no_grad():
            X = X.float().cuda(device)
            y = y.float().cuda(device)
        
        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=0), dim=1)
        y_argmax = torch.argmax(y, dim=1)
        train_acc += (y_pred_class==y_argmax).sum().item()/len(y_pred)
    
    # Adjust metrics to get average los and accuracy 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc    

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device):
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, batch_item in enumerate(dataloader):
            X = batch_item['keypoint']
            y = batch_item['label']
            X, y = X.float().cuda(device), y.float().cuda(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=0), dim=1)
            y_argmax = torch.argmax(y, dim=1)
            test_acc += (y_pred_class==y_argmax).sum().item()/len(y_pred)
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc    

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device=None):
    
    results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}
    global train_writer
    global global_step

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: adjust_learning_rate(epoch,
                                                                                              optimizer,
                                                                                               args.learning_rate,
                                                                                               args.warm_up_epoch,
                                                                                               args.lr_decay_rate,
                                                                                               args.step))

    for epoch in tqdm(range(epochs)):
        train_writer.add_scalar('epoch', epoch, global_step)
        global_step += 1

        train_loss, train_acc = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f} | lr: {lr:.4f}")

        train_writer.add_scalar('train_loss', train_loss, global_step)
        train_writer.add_scalar('train_acc', train_acc, global_step)
        train_writer.add_scalar('learning_rate', lr, global_step)
        train_writer.add_scalar('test_loss', test_loss, global_step)
        train_writer.add_scalar('test_acc', test_acc, global_step)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


    return results

def save_model(model: torch.nn.Module,
               output):
    # Create model save
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_name = f"sttformer_{timestamp}.pth"
    model_save_path = os.path.join('exp',output,model_name)

    #save model state dict
    print(f"Saving model to:{model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def plot_loss_curves(results: Dict[str, List[float]], output: str):
    """
    Plots training curves of a results doctionary.
    """
    # Get the loss valurs of the results dict
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15,7))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs,loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1,2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    # Save the plot as a PNG file
    fig_save_name = "loss_curves.png"
    output_path = os.path.join('exp',args.output,fig_save_name)
    plt.savefig(output_path)
    plt.show()
    print(f"**** Plotting loss curves successfully! ****")
  


def get_random_test_samples_labels(test_dataset,
                                   seed=None):
    if seed:
        random.seed(seed)
    test_samples = []
    test_labels = []
    for sample in random.sample(list(test_dataset), k=9):
        _, sample_label, sample_keypoint, _, _, _ = sample.values()
        test_samples.append(sample_keypoint)
        test_labels.append(sample_label)
    
    test_labels = [int(i.argmax(dim=0).numpy()) for i in test_labels]
    return test_samples, test_labels

def make_predictions(model: torch.nn.Module,
                    data: List[torch.Tensor],
                    device: int):
    pred_probs = []
    model.cuda(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = sample.unsqueeze(0).cuda(device)
            pred = model(sample)
            pred_prob = torch.softmax(pred.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    
    pred_probs = torch.stack(pred_probs)
    pred_classes = list(pred_probs.argmax(dim=1).numpy())

    return pred_classes

# Plot results
def plot_predictions(test_samples,
                    pred_classes,
                    test_labels,
                    action_classes,
                    all_subjects=True,
                    output=None):
    plt.figure(figsize=(10,10))
    nrows, ncols = 3, 3
    for i, sample in enumerate(test_samples):
        plt.subplot(nrows, ncols, i+1)
        ntu_pairs = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4),
                    (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
                    (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16),
                    (18, 17), (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11)]
        
        random_frame_idx = random.choice(range(50))

        tensor_reshaped = sample.permute(3, 1, 2, 0)
        pred_label = action_classes[pred_classes[i]]
        truth_label = action_classes[test_labels[i]]

        title_text = f"Pred: {pred_label} \n Truth: {truth_label}"
        title_text += f"\nSample total lengh: {tensor_reshaped.shape[1]} frame: {random_frame_idx}"
        if pred_label == truth_label:
            plt.title(title_text, fontsize=9, c='g')
        else:
            plt.title(title_text, fontsize=9, c='r')

        if all_subjects:
            x_keypoint_0 = tensor_reshaped[0][random_frame_idx][:,0] 
            y_keypoint_0 = tensor_reshaped[0][random_frame_idx][:,1]

            x_keypoint_1 = tensor_reshaped[1][random_frame_idx][:,0] 
            y_keypoint_1 = tensor_reshaped[1][random_frame_idx][:,1]

            for i,j in ntu_pairs:
                plt.plot([x_keypoint_0[i], x_keypoint_0[j]], [y_keypoint_0[i],y_keypoint_0[j]], c='b', marker='o')
                plt.plot([x_keypoint_1[i], x_keypoint_1[j]], [y_keypoint_1[i],y_keypoint_1[j]], c='r', marker='o')
            
            plt.xlim([np.amin(tensor_reshaped[:,:,:,0].numpy()), np.amax(tensor_reshaped[:,:,:,0].numpy())])
            plt.ylim([np.amin(tensor_reshaped[:,:,:,1].numpy()), np.amax(tensor_reshaped[:,:,:,1].numpy())])
            plt.axis(True)
        else:
            x_keypoint = tensor_reshaped[0][random_frame_idx][:,0] 
            y_keypoint = tensor_reshaped[0][random_frame_idx][:,1]
            for i,j in ntu_pairs:
                plt.plot([x_keypoint[i], x_keypoint[j]], [y_keypoint[i],y_keypoint[j]], c='b', marker='o')
            
            plt.xlim([np.amin(tensor_reshaped[0,:,:,0].numpy()), np.amax(tensor_reshaped[0,:,:,0].numpy())])
            plt.ylim([np.amin(tensor_reshaped[0,:,:,1].numpy()), np.amax(tensor_reshaped[0,:,:,1].numpy())])
            plt.axis(True)
    plt.subplots_adjust(hspace=0.35)
    # Save the plot as a PNG file
    fig_save_name = "predictions.png"
    output_path = os.path.join('exp',args.output,fig_save_name)
    plt.savefig(output_path)
    plt.show()
    print(f"**** Plotting predictions successfully! ****")

def main(parser, action_classes):
    
    exp_folder = Path("exp")
    exp_folder.mkdir(parents=True, exist_ok=True)

    model_path = Path(os.path.join(exp_folder,args.output))
    model_path.mkdir(parents=True, exist_ok=True)

    
    model = load_model(args.device, args.model_path)
    train_dataset, test_dataset = load_dataset(ann_file=args.annotations,
                                               split=args.split,
                                               window_size=args.window_size)
    
    print_model_summary(model,input_size=[args.batch_size, 3, 120, 25, 2])

    ntu_train_dataloader = DataLoader(dataset=train_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               drop_last=True,
                               num_workers=args.num_workers)
    ntu_test_dataloader = DataLoader(dataset=test_dataset,
                               batch_size=args.batch_size,
                               shuffle=False,
                               drop_last=True,
                               num_workers=args.num_workers)
    
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    loss_fn = nn.CrossEntropyLoss().cuda(args.device)
    optimizer = optim.SGD(model.parameters(),
                        lr=learning_rate,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=weight_decay)
    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.model_path is None:
        num_epochs = args.num_epoch

        start_time = timer()

        # Train model
        model_results = train(model=model,
                                train_dataloader=ntu_train_dataloader,
                                test_dataloader=ntu_test_dataloader,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                epochs=num_epochs,
                                device=args.device)
    
        # End the timer and print out how long it took
        end_time = timer()
        total_time = end_time-start_time
        total_time_minutes = total_time // 60
        total_time_minutes_sec = total_time % 60
        print(f"**** Total training time: {total_time_minutes}Mn {total_time_minutes_sec:.3f} sec ****")

        save_model(model=model, output=args.output)

        plot_loss_curves(results=model_results,
                        output=args.output)
        
    test_samples, test_labels = get_random_test_samples_labels(test_dataset=test_dataset,
                                                                seed=args.seed)

    pred_classes = make_predictions(model=model,
                                    data= test_samples,
                                    device=args.device)
    
    plot_predictions(test_samples=test_samples,
                pred_classes=pred_classes,
                test_labels=test_labels,
                all_subjects=True,
                action_classes=action_classes,
                output=args.output)
    
    train_writer.close()


if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    parser = get_parser()
    args = parser.parse_args()
    global_step = 0
    train_writer = SummaryWriter(os.path.join('exp',args.output))

    action_classes = [
    'drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup', 'throw', 'sitting down',
    'standing up (from sitting position)', 'clapping', 'reading', 'writing', 'tear up paper', 'wear jacket',
    'take off jacket', 'wear a shoe', 'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat/cap',
    'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something', 'reach into pocket', 'hopping (one foot jumping)',
    'jump up', 'make a phone call/answer phone', 'playing with phone/tablet', 'typing on a keyboard',
    'pointing to something with finger', 'taking a selfie', 'check time (from watch)', 'rub two hands together',
    'nod head/bow', 'shake head', 'wipe face', 'salute', 'put palms together', 'cross hands in front',
    'sneeze/cough', 'staggering', 'falling', 'touch head (headache)', 'touch chest (stomachache/heart pain)',
    'touch back (backache)', 'touch neck (neckache)', 'nausea or vomiting condition', 'use a fan (with hand or paper)/feeling warm',
    'punching/slapping other person', 'kicking other person', 'pushing other person', 'pat on back of other person',
    'point finger at the other person', 'hugging other person', 'giving something to other person', 'touch other person\'s pocket',
    'handshaking', 'walking towards each other', 'walking apart from each other'
    ]

    idx_to_class = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 47, 48, 49, 50,  51, 52, 53, 54, 55, 56, 57, 58, 59, 60
    ]

    ntu_pairs = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4),
            (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
            (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16),
            (18, 17), (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11)]

    main(args,action_classes)
    print("********* END PROCESSING *********")