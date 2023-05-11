import numpy as np
import matplotlib.pyplot as plt
import random
import os
import torch

class NTU2Feeder:
    """
    Class for preprocessing NTU Dataset as STTFormer feeder format
    """
    def feeder_tensor_reshape(self, tensor: torch.Tensor):
        # tmp_padded = add_padding(tensor, 75) ## ERROR not working well
        # tensor = tensor.unsqueeze(0)
        tmp_reshaped = tensor.permute(3,1,2,0)
        return tmp_reshaped
    
    @staticmethod
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
            y_keypoint_0 = tensor_reshaped[0][random_frame_idx][:,1]

            x_keypoint_1 = tensor_reshaped[1][random_frame_idx][:,0] 
            y_keypoint_1 = tensor_reshaped[1][random_frame_idx][:,1]

            plt.figure(figsize=(4,6))
            for i,j in ntu_pairs:
                plt.plot([x_keypoint_0[i], x_keypoint_0[j]], [y_keypoint_0[i],y_keypoint_0[j]], c='b', marker='o')
                plt.plot([x_keypoint_1[i], x_keypoint_1[j]], [y_keypoint_1[i],y_keypoint_1[j]], c='r', marker='o')
            
            plt.xlim([np.amin(tensor_reshaped[:,:,:,0].numpy()), np.amax(tensor_reshaped[:,:,:,0].numpy())])
            plt.ylim([np.amin(tensor_reshaped[:,:,:,1].numpy()), np.amax(tensor_reshaped[:,:,:,1].numpy())])

            title = f"Sample total lengh: {tensor_reshaped.shape[1]} frame: {random_frame_idx}"
            plt.title(title)
        else:
            x_keypoint = tensor_reshaped[0][random_frame_idx][:,0] 
            y_keypoint = tensor_reshaped[0][random_frame_idx][:,1]
            plt.figure(figsize=(4,6))
            for i,j in ntu_pairs:
                plt.plot([x_keypoint[i], x_keypoint[j]], [y_keypoint[i],y_keypoint[j]], c='b', marker='o')
            
            plt.xlim([np.amin(tensor_reshaped[0,:,:,0].numpy()), np.amax(tensor_reshaped[0,:,:,0].numpy())])
            plt.ylim([np.amin(tensor_reshaped[0,:,:,1].numpy()), np.amax(tensor_reshaped[0,:,:,1].numpy())])

            title = f"Sample total lengh: {tensor_reshaped.shape[1]} frame: {random_frame_idx}"
            plt.title(title)
            
    def crop_frames(self, tensor: torch.Tensor, window_size):
        C, T, V, M = tensor.shape
        if M == 2:
            if T < window_size:
                # If the window size is longer than the actual frame length,
                # we pad the tensor with zeros along the T dimension
                padded_tensor = torch.zeros((C, window_size, V, M))
                padded_tensor[:, :T, :, :] = tensor
                return padded_tensor
            
            # If the crop size is shorter, we don't need to pad the tensor
            tensor_permuted = tensor.permute(3,2,0,1)
            tensor_cropped = tensor_permuted[:,:,:,:window_size]
            M, V, C, T = tensor_cropped.shape
            cropped_tensor = tensor_cropped.permute(2,3,1,0)
            return cropped_tensor
        if T < window_size:
            # If the window size is longer than the actual frame length,
            # we pad the tensor with zeros along the T dimension
            padded_tensor = torch.zeros((C, window_size, V, M))
            padded_tensor[:, :T, :, :] = tensor
            padded_tensor = padded_tensor.permute(3,1,2,0)
            padded_tensor_added = torch.zeros((2, window_size, V, C))
            padded_tensor_added[:1,:,:,:] = padded_tensor
            padded_tensor_added = padded_tensor_added.permute(3,1,2,0)
            return padded_tensor_added
        
        tensor_permuted = tensor.permute(3,2,0,1)
        tensor_cropped = tensor_permuted[:,:,:,:window_size]
        padded_tensor = tensor_cropped.permute(0,3,1,2)
        M, T, V, C = padded_tensor.shape
        padded_tensor_added = torch.zeros((2, T, V, C))
        padded_tensor_added[:1,:,:,:] = padded_tensor
        padded_tensor_added = padded_tensor_added.permute(3,1,2,0)
        return padded_tensor_added
    
    def __init__(self, window_size: int=120):
        self.window_size = window_size

    def __call__(self, dataset_item):
        skeleton = dataset_item['keypoint']
        tensor = torch.tensor(skeleton)
        tensor = self.feeder_tensor_reshape(tensor)
        tensor_cropped = self.crop_frames(tensor, self.window_size)
        dataset_item['keypoint'] = tensor_cropped.numpy()
        dataset_item['input'] = tensor_cropped.numpy() #copy keypoint content to input (to solve problem of shape for Dataloader)

        return dataset_item
                