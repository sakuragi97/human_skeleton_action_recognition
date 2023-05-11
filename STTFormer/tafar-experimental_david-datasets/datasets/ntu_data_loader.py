import math
import numpy as np
import pickle
import torch

from ntu_dataset import NTUDataset
# from models import transformer


# source: https://github.com/shlizee/Predict-Cluster


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def mini_batch(data, seq_len, input_size, batch_size, hidden_dim):
    encoder_inputs = torch.zeros((batch_size, seq_len, input_size))
    autoregressive_inputs = torch.zeros((batch_size, 1, input_size))
    seq_len_enc = torch.zeros((batch_size,))
    decoder_inputs = torch.zeros((batch_size, seq_len, hidden_dim))
    data_len = len(data)
    for i in range(batch_size):
        index = np.random.choice(data_len)
        data_sel = data[index]
        data_sel_auto = data[index][0]
        encoder_inputs[i, :data_sel.shape[0], :] = data_sel.detach().clone()
        autoregressive_inputs[i, 0, :] = data_sel_auto.detach().clone()
        seq_len_enc[i] = data_sel.shape[0]
    return encoder_inputs, autoregressive_inputs, decoder_inputs, seq_len_enc


def normalize_video(video):
    video = np.reshape(video, (video.shape[0], -1))  ### ERROR EDITED
    max_75 = np.amax(video, axis=0)
    min_75 = np.amin(video, axis=0)
    max_x = np.max([max_75[i] for i in range(0,75,3)])
    max_y = np.max([max_75[i] for i in range(1,75,3)])
    max_z = np.max([max_75[i] for i in range(2,75,3)])
    min_x = np.min([min_75[i] for i in range(0,75,3)])
    min_y = np.min([min_75[i] for i in range(1,75,3)])
    min_z = np.min([min_75[i] for i in range(2,75,3)])
    norm = np.zeros_like(video)
    for i in range(0,75,3):
        norm[:,i] = 2*(video[:,i]-min_x)/(max_x-min_x)-1
        norm[:,i+1] = 2*(video[:,i+1]-min_y)/(max_y-min_y)-1
        norm[:,i+2] = 2*(video[:,i+2]-min_z)/(max_z-min_z)-1
    return norm


def downsample(data, batch_size=64):
    dsamp_data = []
    for i in range(len(data)):
        val = torch.Tensor(data[i]['input'])
        if val.shape[0] > batch_size:
            new_val = torch.zeros((batch_size, 75))
            diff = math.floor(val.shape[0] / batch_size)
            idx = 0
            for i in range(0, val.shape[0], diff):
                new_val[idx, :] = val[i, :]
                idx += 1
                if idx >= batch_size:
                    break
            dsamp_data.append(new_val)
        else:
            dsamp_data.append(val)
    return dsamp_data


def get_train_lab(dsamp_train, emb_size, fea_size=75, samp_len=50):
    fea = []
    lab = []
    seq_len_new = []
    for idx, data in enumerate(dsamp_train):
        label = data["label"]
        val = np.asarray(data["input"])
        raw_len = val.shape[0]
        if raw_len > samp_len:
            seq_len_new.append(samp_len)
            fea.append(dsamp_train[idx])
        else:
            seq_len_new.append(raw_len)
            pad_data = np.zeros((samp_len, fea_size))
            pad_data[:raw_len, :] = dsamp_train[idx]
            fea.append(pad_data)
        one_hot_label = np.zeros((emb_size,))
        one_hot_label[label] = 1.
        lab.append(one_hot_label)
    return lab


def get_test_lab(dsamp_test, emb_size, fea_size=75, samp_len=50):
    test_fea = []
    test_lab = []
    test_seq_len_new = []
    for idx, data in enumerate(dsamp_test):
        label = data["label"]
        val = np.asarray(data["input"])
        raw_len = val.shape[0]
        if raw_len > samp_len:
            test_seq_len_new.append(samp_len)
            test_fea.append(dsamp_test[idx])
        else:
            test_seq_len_new.append(raw_len)
            pad_data = np.zeros((samp_len, fea_size))
            pad_data[:raw_len, :] = dsamp_test[idx]
            test_fea.append(pad_data)
        one_hot_label = np.zeros((emb_size,))
        one_hot_label[label] = 1.
        test_lab.append(one_hot_label)
    return test_lab


def mini_batch_classify(feature_xyz, labels, seq_len, batch_size):
    for start in range(0, len(feature_xyz), batch_size):
        end = min(start+batch_size, len(feature_xyz))
        yield feature_xyz[start:end], labels[start:end], seq_len[start:end]


if __name__ == '__main__':
    path = '/home/yas50454/datasets/NTU_Data/NTU_60/'
    train_data = load_data(path + 'NTU_60_cross_subject_data_transform.pkl')
    test_data = load_data(path + 'NTU_60_cross_subject_data_val_transform.pkl')
    dataset_len = len(train_data)
    
    for i in range(len(train_data)):
        train_data[i]['input'] = normalize_video(np.asarray(train_data[i]['input'])) ### ERROR
    for i in range(len(test_data)):
        test_data[i]['input'] = normalize_video(np.asarray(test_data[i]['input']))
    print(train_data[0].keys())
    dsamp_train = downsample(train_data)
    # dsamp_test = downsample(test_data)
    sequence_length = dsamp_train[0].shape[1]
    enc_in, _, dec_in, enc_in_len = mini_batch(dsamp_train, sequence_length, dsamp_train[0].shape[-1],
                                               dsamp_train[0].shape[0])

    print(f"END SCRIPT!")
    # model = transformer.TransformerModel(orig_dim=enc_in.shape[-1], hidden_dim=128, depth=2, num_heads=4)
    # tmp2 = model(enc_in)
    # print(tmp2)
    # transformer.TransformerModel()
