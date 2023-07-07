#  "python": "/home/yas50454/miniconda3/envs/torch_1_131/bin/python"

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import math
import sys
import os
from pathlib import Path
import datetime
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts



from tensorboardX import SummaryWriter
from utils.ntu_dataset import NTUDataset
import utils.augmentations as augmentations
from utils.augmentations import Normalize3D
from utils.base_dataset import BaseDataset
from model.sttformer import Model
from utils.ntu_pipeline import NTU2Feeder
from utils.Dino_utils import DinoMLP
from utils.Dino_utils import MultiCropWrapper
from utils.Dino_utils import init_distributed_mode
from utils.Dino_utils import DINOLoss
from utils.Dino_utils import get_params_groups
from utils.Dino_utils import LARS
from utils.Dino_utils import cosine_scheduler, get_world_size
from utils.Dino_utils import CustomKNeighborsClassifier
from utils.Dino_utils import LARS

from tqdm.auto import tqdm
from timeit import default_timer as timer




def plot_tensor_skeleton(tensor: torch.Tensor, set_frame: int=None, all_subjects: bool=False):
        """
        Take as input a tensor of shape (C, T, V, M)
        """
        ntu_pairs = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4),
                (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
                (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16),
                (18, 17), (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11)]
        print(tensor.shape)
        
        if set_frame is not None:
            if set_frame > tensor.shape[1]:
                raise ValueError(f"Frame value to high, total frame {tensor.shape[1]}")
            else:
                random_frame_idx = set_frame
        else:
            random_frame_idx = random.choice(range(tensor.shape[1]))

        tensor_reshaped = tensor.permute(3, 1, 2, 0)

        if all_subjects:
            x_keypoint_0 = tensor_reshaped[0][random_frame_idx][:,0] 
            y_keypoint_0 = tensor_reshaped[0][random_frame_idx][:,2]

            x_keypoint_1 = tensor_reshaped[1][random_frame_idx][:,0] 
            y_keypoint_1 = tensor_reshaped[1][random_frame_idx][:,2]

            plt.figure(figsize=(4,6))
            for i,j in ntu_pairs:
                plt.plot([x_keypoint_0[i], x_keypoint_0[j]], [y_keypoint_0[i],y_keypoint_0[j]], c='b', marker='o')
                plt.plot([x_keypoint_1[i], x_keypoint_1[j]], [y_keypoint_1[i],y_keypoint_1[j]], c='r', marker='o')
            
            plt.xlim([np.amin(tensor_reshaped[:,:,:,0].numpy()), np.amax(tensor_reshaped[:,:,:,0].numpy())])
            plt.ylim([np.amin(tensor_reshaped[:,:,:,2].numpy()), np.amax(tensor_reshaped[:,:,:,2].numpy())])

            title = f"Sample total lengh: {tensor_reshaped.shape[1]} frame: {random_frame_idx}"
            plt.title(title)
        else:
            x_keypoint = tensor_reshaped[0][random_frame_idx][:,0] 
            y_keypoint = tensor_reshaped[0][random_frame_idx][:,2]
            plt.figure(figsize=(4,6))
            for i,j in ntu_pairs:
                plt.plot([x_keypoint[i], x_keypoint[j]], [y_keypoint[i],y_keypoint[j]], c='b', marker='o')
            
            plt.xlim([np.amin(tensor_reshaped[0,:,:,0].numpy()), np.amax(tensor_reshaped[0,:,:,0].numpy())])
            plt.ylim([np.amin(tensor_reshaped[0,:,:,2].numpy()), np.amax(tensor_reshaped[0,:,:,2].numpy())])

            title = f"Sample total lengh: {tensor_reshaped.shape[1]} frame: {random_frame_idx}"
            plt.title(title)

def classify(embeddings, labels, classifier=None):

    if classifier:

        return classifier, classifier(embeddings)

    classifier = CustomKNeighborsClassifier(embedding=embeddings, label=labels)

    return classifier, classifier(embeddings)

def train_step(student: torch.nn.Module,
               teacher: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               normal_dataloader: torch.utils.data.DataLoader,
               val_dataloader: torch.utils.data.DataLoader,
               dino_loss: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               momentum_schedule,
               epoch,
               lr_schedule=None,
               wd_schedule=None,
               device=0):
    train_loss, cos_sum = 0, 0
    
    lr_update = optimizer.param_groups[0]['lr']
    weight_decay_update = optimizer.param_groups[0]['weight_decay']
         
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    

    for it, batch_item in enumerate(train_dataloader):
        # update weight decay and learning rate according to their schedule
        it = (len(train_dataloader) * epoch) + it  # global training iteration
        # for i, param_group in enumerate(optimizer.param_groups):
        #         param_group["lr"] = lr_schedule[it]
        #         if i == 0:  # only the first group is regularized
        #             param_group["weight_decay"] = wd_schedule[it]
        
        X_views = batch_item['samples']

        
        X_views = [view.cuda(device) for view in X_views]

        # Forward pass
        teacher_output = teacher(X_views[:1]) # The two first views are global views

        student_output = student(X_views)
        
        loss = dino_loss(student_output, teacher_output, epoch)
        
        train_loss += loss.item()
        
        cos_output = cos(teacher_output.detach().chunk(1)[0], student_output.detach().chunk(2)[0])
        cos_sum += torch.sum(cos_output)/len(cos_output)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        ## TODO: feed the knn the output student model
        # run two dataloaders (one for traning, other for validation)
        # evaluation in the last batch of the first epoch.
        # 
         
        # Backward
        # student update
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher.module.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # lr_update = lr_schedule[it]
        # weight_decay_update = wd_schedule[it]  
    
    if epoch==0 or epoch % 3==0:
        labels=[]
        val_labels=[]
        train_embeds=[]
        val_embeds=[]
        student.eval()
        with torch.inference_mode():
            for it, batch_item in enumerate(normal_dataloader):
                label = torch.argmax(batch_item['label'], dim=-1) 
                embed = student(batch_item['keypoint']).detach().cpu().numpy()
                labels.append(label)
                train_embeds.append(embed)
            
            labels = np.concatenate(labels, axis=0)
            train_embeds = np.concatenate(train_embeds, axis=0)

            for it, batch_item in enumerate(val_dataloader):
                label = torch.argmax(batch_item['label'], dim=-1) 
                embed = student(batch_item['keypoint']).detach().cpu().numpy()
                val_labels.append(label)
                val_embeds.append(embed)
            
            val_labels = np.concatenate(val_labels, axis=0)
            val_embeds = np.concatenate(val_embeds, axis=0)

            knn_classifier, knn_pred = classify(train_embeds, labels, classifier=None)
            knn_val_pred = knn_classifier(val_embeds)
            knn_acc = (knn_val_pred == val_labels).sum() / len(knn_val_pred)
    else:
        knn_acc = 0.
              
    lr_schedule.step()
    
    # Adjust metrics to get average loss and accuracy 
    train_loss = train_loss / len(train_dataloader)
    cos_sim = cos_sum / len(train_dataloader)
   
    return train_loss, lr_update, cos_sim, knn_acc

def plot_loss_curves(results: Dict[str, List[float]]):
  """
  Plots training curves of a results doctionary.
  """
  # Get the loss valurs of the results dict
  loss = results["train_loss"]
  lr = results["lr"]


  # Figure out how many epochs there were
  epochs = range(len(results["train_loss"]))

  # Setup a plot
  plt.figure(figsize=(18,8))

  # Plot the loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs,loss, label="train_loss", marker='o')

  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend();

  # Plot the accuracy
  plt.subplot(1,2, 2)
  plt.plot(epochs, lr, label="learning rate", marker='o')
  plt.title("learning rate")
  plt.xlabel("Epochs")
  plt.legend();

  plt.savefig(exp_subfolder/f"plot_loss_curves.png")


def main():
    
    
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

    init_distributed_mode()

    ann_file = '/home/yas50454/datasets/NTU_Data/NTU_60/ntu60_3danno.pkl'
    ntu2feeder = NTU2Feeder(window_size=60)
    pipeline = torchvision.transforms.Compose([augmentations.PreNormalize3D(),ntu2feeder])
    augmentation = torchvision.transforms.Compose([augmentations.RandomRot(theta=0.2), augmentations.RandomScale(scale=0.2), augmentations.RandomGaussianNoise(sigma=0.005)])
    strong_augmentation = torchvision.transforms.Compose([augmentations.RandomRot(theta=0.6), augmentations.RandomScale(scale=0.6), augmentations.RandomGaussianNoise(sigma=0.04)])
    
    ntu_augmented_dataset = NTUDataset(ann_file, pipeline=pipeline, split='xsub_train', num_classes=60, multi_class=True, augmentation=augmentation, strong_augmentation=strong_augmentation)

    ntu_normal_dataset = NTUDataset(ann_file, pipeline=pipeline, split='xsub_train', num_classes=60, multi_class=True, augmentation=None, strong_augmentation=None)
    ntu_val_dataset = NTUDataset(ann_file, pipeline=pipeline, split='xsub_val', num_classes=60, multi_class=True, augmentation=None, strong_augmentation=None)

    
    ### Preparing models student, teacher
    output_device = 0 # GPU device

    config = [[64,  64,  16], [64,  64,  16], 
            [64,  128, 32], [128, 128, 32],
            [128, 256, 64], [256, 256, 64], 
            [256, 256, 64], [256, 256, 64]]

    sttformer_student = Model(len_parts=6,
                num_frames=60,
                num_joints=25,
                num_classes=256,
                num_heads=3,
                num_channels=3,
                num_persons=2,
                kernel_size=[3,5],
                use_pes=True,
                config=config)

    sttformer_teacher = Model(len_parts=6,
                num_frames=60,
                num_joints=25,
                num_classes=256,
                num_heads=3,
                num_channels=3,
                num_persons=2,
                kernel_size=[3,5],
                use_pes=True,
                config=config)

    in_channels = 256
    hidden_channels=512
    bottleneck_channels=128 #Had no effect on training
    out_channels=128

    dino_mlp = DinoMLP(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                bottleneck_channels=bottleneck_channels
    )

    sttformer_student = MultiCropWrapper(sttformer_student, dino_mlp)
    sttformer_teacher = MultiCropWrapper(sttformer_teacher, dino_mlp)

    sttformer_student = nn.DataParallel(sttformer_student, device_ids=[output_device,output_device+1,output_device+2,output_device+3])
    sttformer_teacher = nn.DataParallel(sttformer_teacher, device_ids=[output_device,output_device+1,output_device+2,output_device+3])

    sttformer_student, sttformer_teacher = sttformer_student.cuda(), sttformer_teacher.cuda() 

    # teacher and student start with the same weights
    sttformer_teacher.module.load_state_dict(sttformer_student.module.state_dict())

    # there is no backpropagation through the teacher, so no need for gradients
    for p in sttformer_teacher.parameters():
        p.requires_grad = False


    ### Preparing Loss
    epochs = 51

    dino_loss = DINOLoss(out_dim=128,
                        ncrops=2,
                        warmup_teacher_temp=0.03,
                        teacher_temp=0.03,
                        warmup_teacher_temp_epochs=0,
                        student_temp=0.1,
                        center_momentum=0.9,
                        nepochs=epochs).cuda()
    
    ### Preparing Optimizer

    params_group = get_params_groups(sttformer_student)

    learning_rate = 0.3
    weight_decay = 0.0004
    # optimizer = optim.SGD(params_group,
    #                     lr=learning_rate,
    #                     momentum=0.9,
    #                     nesterov=True,
    #                     weight_decay=weight_decay
    #                     )
    
    optimizer = LARS(params_group,
                     lr=learning_rate,
                     weight_decay=weight_decay,
                     momentum=0.9)
    # optimizer = optim.Adam(params_group,
    #                     lr=learning_rate,
    #                     weight_decay=weight_decay
    #                     )


    batch_size_per_gpu = 128

    ntu_train_dataloader = DataLoader(dataset=ntu_augmented_dataset,
                                batch_size=batch_size_per_gpu,
                                shuffle=True,
                                drop_last=True,
                                num_workers=os.cpu_count())
    
    ntu_normal_dataloader = DataLoader(dataset=ntu_normal_dataset,
                                batch_size=batch_size_per_gpu,
                                shuffle=True,
                                drop_last=True,
                                num_workers=os.cpu_count())
    
    ntu_val_dataloader = DataLoader(dataset=ntu_val_dataset,
                                batch_size=batch_size_per_gpu,
                                shuffle=True,
                                drop_last=True,
                                num_workers=os.cpu_count())
    
    ### Preparing Schedulers

    # lr = 0.01
    # min_lr = 1e-2
    # warmup_epochs = 1
    # weight_decay = 0.04
    # weight_decay_end = 0.4
    momentum_teacher = 0.9995

    # lr_schedule = cosine_scheduler(
    #             lr * (batch_size_per_gpu * get_world_size()) / 256.,  # linear scaling rule
    #             min_lr,
    #             epochs, len(ntu_train_dataloader),
    #             warmup_epochs=warmup_epochs
    #         )
    # wd_schedule = cosine_scheduler(
    #             weight_decay,
    #             weight_decay_end,
    #             epochs, len(ntu_train_dataloader)
    #             )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(momentum_teacher, 1,
                                        epochs, len(ntu_train_dataloader))
    print(f"Loss, optimizer and schedulers ready.")

    # lr_schedule = CosineAnnealingLR(optimizer,
    #                             T_max = epochs, # Maximum number of iterations.
    #                             eta_min = 1e-3) # Minimum learning rate.
    lr_schedule = CosineAnnealingLR(optimizer,
                              T_max = 100, # Maximum number of iterations.
                             eta_min = 1e-3) # Minimum learning rate.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    start_time = timer()
    ### Start training 

    results = {"train_loss": [],
                "lr": [],
                "weight_decay": []}

    global_step = 0

    for epoch in tqdm(range(epochs)):
        train_writer.add_scalar('epoch', epoch, global_step)

    # for epoch in range(epochs):
        train_loss, lr_update, cos_sim, knn_acc = train_step(student=sttformer_student,
                                                                teacher=sttformer_teacher,
                                                                train_dataloader=ntu_train_dataloader,
                                                                normal_dataloader=ntu_normal_dataloader,
                                                                val_dataloader=ntu_val_dataloader,
                                                                dino_loss=dino_loss,
                                                                optimizer=optimizer,
                                                                lr_schedule=lr_schedule,
                                                                momentum_schedule=momentum_schedule,
                                                                epoch=epoch,
                                                                device=output_device)
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Cos similarity: {cos_sim:.4f} | Knn acc: {knn_acc:.4f} | lr update: {lr_update:.4f}")
        train_writer.add_scalar('train_loss', train_loss, global_step)
        train_writer.add_scalar('learning_rate', lr_update, global_step)
        train_writer.add_scalar('cos_sim', cos_sim, global_step)
        train_writer.add_scalar('knn_acc', knn_acc, global_step)


        global_step += 1


        results["train_loss"].append(train_loss)
        results["lr"].append(lr_update)

    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

    
    
    # Create model save
    student_model_name = f"dino_student_sttformer_{timestamp}.pth"
    teacher_model_name = f"dino_teacher_sttformer_{timestamp}.pth"
    model_student_save_path = exp_subfolder/student_model_name
    model_teacher_save_path = exp_subfolder/teacher_model_name


    #save model state dict
    print(f"Saving model to:{model_student_save_path}")
    torch.save(obj=sttformer_student.state_dict(), f=model_student_save_path)

    print(f"Saving model to:{model_teacher_save_path}")
    torch.save(obj=sttformer_teacher.state_dict(), f=model_teacher_save_path)
    print(f"Models saved with success!")

    
    plot_loss_curves(results)
    train_writer.close()



if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    exp_folder = Path("dino_exp")
    exp_folder.mkdir(parents=True, exist_ok=True)

    exp_subfolder = Path(exp_folder/f"exp_{timestamp}")
    exp_subfolder.mkdir(parents=True, exist_ok=True)
    train_writer = SummaryWriter(exp_subfolder)
    main()
    print("********* END PROCESSING *********")

