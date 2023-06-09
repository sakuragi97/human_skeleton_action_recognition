import os
import os.path as osp
import sys
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import json
import torchvision


import augmentations

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from ntu_dataset import NTUDataset



def save_statistics(ske_name, setup_file, camera_file, performer_file, replication_file, label_file):
    left, action = ske_name.split('A')
    left, replication = left.split('R')
    left, performer = left.split('P')
    left, camera = left.split('C')
    left, setup = left.split('S')
    with open(setup_file, 'a') as sf:
        sf.write(setup + '\n')
    with open(camera_file, 'a') as cf:
        cf.write(camera + '\n')
    with open(performer_file, 'a') as pf:
        pf.write(setup + '\n')
    with open(replication_file, 'a') as rf:
        rf.write(setup + '\n')
    with open(label_file, 'a') as lf:
        lf.write(action + '\n')


def get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger, setup_file, camera_file,
                        performer_file, replication_file, label_file):
    """
    Get raw bodies data from a skeleton sequence.

    Each body's data is a dict that contains the following keys:
      - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
      - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
      - interval: a list which stores the frame indices of this body.
      - motion: motion amount (only for the sequence with 2 or more bodyIDs).

    Return:
      a dict for a skeleton sequence with 3 key-value pairs:
        - name: the skeleton filename.
        - data: a dict which stores raw data of each body.
        - num_frames: the number of valid frames.
    """
    # for sk_path in skes_path:
    ske_file = osp.join(skes_path, ske_name + '.skeleton')
    save_statistics(ske_name, setup_file, camera_file, performer_file, replication_file, label_file)
    assert osp.exists(ske_file), 'Error: Skeleton file %s not found' % ske_file
    # Read all data from .skeleton file into a list (in string format)
    # print('Reading data from %s' % ske_file[-29:])
    with open(ske_file, 'r') as fr:
        str_data = fr.readlines()

    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop = []
    bodies_data = dict()
    valid_frames = -1  # 0-based index
    current_line = 1

    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 1

        if num_bodies == 0:  # no data in this frame, drop it
            frames_drop.append(f)  # 0-based index
            continue

        valid_frames += 1
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
        colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)

        for b in range(num_bodies):
            bodyID = str_data[current_line].strip('\r\n').split()[0]
            current_line += 1
            num_joints = int(str_data[current_line].strip('\r\n'))  # 25 joints
            current_line += 1

            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
                current_line += 1

            if bodyID not in bodies_data:  # Add a new body's data
                body_data = dict()
                body_data['joints'] = joints[b]  # ndarray: (25, 3)
                body_data['colors'] = colors[b, np.newaxis]  # ndarray: (1, 25, 2)
                body_data['interval'] = [valid_frames]  # the index of the first frame
            else:  # Update an already existed body's data
                body_data = bodies_data[bodyID]
                # Stack each body's data of each frame along the frame order
                body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                pre_frame_idx = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_idx + 1)  # add a new frame index

            bodies_data[bodyID] = body_data  # Update bodies_data

    num_frames_drop = len(frames_drop)
    # if num_frames_drop == num_frames:
    #     with open('/home/lerch-iosb/devel/IOSB/PROTOTYPES/Predict-Cluster/preprocess/statistics/exclude.txt', 'a') as sf:
    #         sf.write(ske_name + '\n')
    # else:
    assert num_frames_drop < num_frames, \
        'Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name)
    if num_frames_drop > 0:
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=int)
        frames_drop_logger.info('{}: {} frames missed: {}\n'.format(ske_name, num_frames_drop,
                                                                    frames_drop))

    # Calculate motion (only for the sequence with 2 or more bodyIDs)
    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


def get_raw_skes_data(setup_file, camera_file, performer_file, replication_file, label_file, skes_name_file, skes_path,
                      frames_drop_skes, frames_drop_logger, save_data_pkl, save_path, frames_drop_pkl):
    # # save_path = './data'
    # # skes_path = '/data/pengfei/NTU/nturgb+d_skeletons/'
    # stat_path = osp.join(save_path, 'statistics')
    #
    # skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    # save_data_pkl = osp.join(save_path, 'raw_skes_data.pkl')
    # frames_drop_pkl = osp.join(save_path, 'frames_drop_skes.pkl')
    #
    # frames_drop_logger = logging.getLogger('frames_drop')
    # frames_drop_logger.setLevel(logging.INFO)
    # frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'frames_drop.log')))
    # frames_drop_skes = dict()
    ske_name_list = list(os.listdir(skes_path))
    num_files = len(ske_name_list)
    print('Found %d available skeleton files.' % num_files)
    with open('/home/dav86141/devel/skeleton_smoothing/Predict_Cluster/preprocess/statistics/exclude.txt', 'r') as f:
        exclude = f.read().split('\n')
    exclude = [file_name.split('.')[0] for file_name in ske_name_list if file_name.split('.')[0] in exclude]
    raw_skes_data = []
    frames_cnt = np.zeros(num_files-len(exclude), dtype=int)
    idx = 0
    idx_ref = 0
    for file_name in ske_name_list:
        ske_name = file_name.split('.')[0]
        if ske_name not in exclude:
            bodies_data = get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger, setup_file,
                                              camera_file, performer_file, replication_file, label_file)
            raw_skes_data.append(bodies_data)
            frames_cnt[idx] = bodies_data['num_frames']
            idx += 1
            if (idx + 1) % 10000 == 0:
                print('Processed: %.2f%% (%d / %d)' % \
                      (100.0 * (idx + 1) / len(frames_cnt), idx + 1, len(frames_cnt)))
            if idx >= len(frames_cnt):
                print('Should finish now... %d' % idx_ref)
        idx_ref += 1

    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw bodies data into %s' % save_data_pkl)
    print('Total frames: %d' % np.sum(frames_cnt))

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)


def denoising_by_length(ske_name, bodies_data, noise_len_thres, noise_len_logger):
    """
    Denoising data based on the frame length for each bodyID.
    Filter out the bodyID which length is less or equal than the predefined threshold.

    """
    noise_info = str()
    new_bodies_data = bodies_data.copy()
    for (bodyID, body_data) in new_bodies_data.items():
        length = len(body_data['interval'])
        if length <= noise_len_thres:
            noise_info += 'Filter out: %s, %d (length).\n' % (bodyID, length)
            noise_len_logger.info('{}\t{}\t{:.6f}\t{:^6d}'.format(ske_name, bodyID,
                                                                  body_data['motion'], length))
            del bodies_data[bodyID]
    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info


def get_valid_frames_by_spread(points, noise_spr_thres1):
    """
    Find the valid (or reasonable) frames (index) based on the spread of X and Y.

    :param points: joints or colors
    """
    num_frames = points.shape[0]
    valid_frames = []
    for i in range(num_frames):
        x = points[i, :, 0]
        y = points[i, :, 1]
        if (x.max() - x.min()) <= noise_spr_thres1 * (y.max() - y.min()):  # 0.8
            valid_frames.append(i)
    return valid_frames


def denoising_by_spread(ske_name, bodies_data, noise_spr_thres2, noise_spr_logger, noise_spr_thres1):
    """
    Denoising data based on the spread of Y value and X value.
    Filter out the bodyID which the ratio of noisy frames is higher than the predefined
    threshold.

    bodies_data: contains at least 2 bodyIDs
    """
    noise_info = str()
    denoised_by_spr = False  # mark if this sequence has been processed by spread.

    new_bodies_data = bodies_data.copy()
    # for (bodyID, body_data) in bodies_data.items():
    for (bodyID, body_data) in new_bodies_data.items():
        if len(bodies_data) == 1:
            break
        valid_frames = get_valid_frames_by_spread(body_data['joints'].reshape(-1, 25, 3), noise_spr_thres1)
        num_frames = len(body_data['interval'])
        num_noise = num_frames - len(valid_frames)
        if num_noise == 0:
            continue

        ratio = num_noise / float(num_frames)
        motion = body_data['motion']
        if ratio >= noise_spr_thres2:  # 0.69754
            del bodies_data[bodyID]
            denoised_by_spr = True
            noise_info += 'Filter out: %s (spread rate >= %.2f).\n' % (bodyID, noise_spr_thres2)
            noise_spr_logger.info('%s\t%s\t%.6f\t%.6f' % (ske_name, bodyID, motion, ratio))
        else:  # Update motion
            joints = body_data['joints'].reshape(-1, 25, 3)[valid_frames]
            body_data['motion'] = min(motion, np.sum(np.var(joints.reshape(-1, 3), axis=0)))
            noise_info += '%s: motion %.6f -> %.6f\n' % (bodyID, motion, body_data['motion'])
            # TODO: Consider removing noisy frames for each bodyID

    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info, denoised_by_spr


def denoising_by_motion(ske_name, bodies_data, bodies_motion, noise_mot_thres_hi, noise_mot_thres_lo, noise_mot_logger):
    """
    Filter out the bodyID which motion is out of the range of predefined interval

    """
    # Sort bodies based on the motion, return a list of tuples
    # bodies_motion = sorted(bodies_motion.items(), key=lambda x, y: cmp(x[1], y[1]), reverse=True)
    bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)

    # Reserve the body data with the largest motion
    denoised_bodies_data = [(bodies_motion[0][0], bodies_data[bodies_motion[0][0]])]
    noise_info = str()

    for (bodyID, motion) in bodies_motion[1:]:
        if (motion < noise_mot_thres_lo) or (motion > noise_mot_thres_hi):
            noise_info += 'Filter out: %s, %.6f (motion).\n' % (bodyID, motion)
            noise_mot_logger.info('{}\t{}\t{:.6f}'.format(ske_name, bodyID, motion))
        else:
            denoised_bodies_data.append((bodyID, bodies_data[bodyID]))
    if noise_info != '':
        noise_info += '\n'

    return denoised_bodies_data, noise_info


def denoising_bodies_data(bodies_data, noise_len_thres, noise_len_logger, noise_spr_thres2, noise_spr_logger,
                          noise_spr_thres1):
    """
    Denoising data based on some heuristic methods, not necessarily correct for all samples.

    Return:
      denoised_bodies_data (list): tuple: (bodyID, body_data).
    """
    ske_name = bodies_data['name']
    bodies_data = bodies_data['data']

    # Step 1: Denoising based on frame length.
    bodies_data, noise_info_len = denoising_by_length(ske_name, bodies_data, noise_len_thres, noise_len_logger)

    if len(bodies_data) == 1:  # only has one bodyID left after step 1
        return bodies_data.items(), noise_info_len

    # Step 2: Denoising based on spread.
    bodies_data, noise_info_spr, denoised_by_spr = denoising_by_spread(ske_name, bodies_data, noise_spr_thres2,
                                                                       noise_spr_logger, noise_spr_thres1)

    if len(bodies_data) == 1:
        return bodies_data.items(), noise_info_len + noise_info_spr

    bodies_motion = dict()  # get body motion
    for (bodyID, body_data) in bodies_data.items():
        bodies_motion[bodyID] = body_data['motion']
    # Sort bodies based on the motion
    # bodies_motion = sorted(bodies_motion.items(), key=lambda x, y: cmp(x[1], y[1]), reverse=True)
    bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)
    denoised_bodies_data = list()
    for (bodyID, _) in bodies_motion:
        denoised_bodies_data.append((bodyID, bodies_data[bodyID]))

    return denoised_bodies_data, noise_info_len + noise_info_spr

    # TODO: Consider denoising further by integrating motion method

    # if denoised_by_spr:  # this sequence has been denoised by spread
    #     bodies_motion = sorted(bodies_motion.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    #     denoised_bodies_data = list()
    #     for (bodyID, _) in bodies_motion:
    #         denoised_bodies_data.append((bodyID, bodies_data[bodyID]))
    #     return denoised_bodies_data, noise_info

    # Step 3: Denoising based on motion
    # bodies_data, noise_info = denoising_by_motion(ske_name, bodies_data, bodies_motion)

    # return bodies_data, noise_info


def get_one_actor_points(body_data, num_frames):
    """
    Get joints and colors for only one actor.
    For joints, each frame contains 75 X-Y-Z coordinates.
    For colors, each frame contains 25 x 2 (X, Y) coordinates.
    """
    joints = np.zeros((num_frames, 75), dtype=np.float32)
    colors = np.ones((num_frames, 1, 25, 2), dtype=np.float32) * np.nan
    start, end = body_data['interval'][0], body_data['interval'][-1]
    joints[start:end + 1] = body_data['joints'].reshape(-1, 75)
    colors[start:end + 1, 0] = body_data['colors']

    return joints, colors


def remove_missing_frames(ske_name, joints, colors, missing_skes_logger1, missing_skes_logger2, missing_skes_logger,
                          missing_count):
    """
    Cut off missing frames which all joints positions are 0s

    For the sequence with 2 actors' data, also record the number of missing frames for
    actor1 and actor2, respectively (for debug).
    """
    num_frames = joints.shape[0]
    num_bodies = colors.shape[1]  # 1 or 2

    if num_bodies == 2:  # DEBUG
        missing_indices_1 = np.where(joints[:, :75].sum(axis=1) == 0)[0]
        missing_indices_2 = np.where(joints[:, 75:].sum(axis=1) == 0)[0]
        cnt1 = len(missing_indices_1)
        cnt2 = len(missing_indices_2)

        start = 1 if 0 in missing_indices_1 else 0
        end = 1 if num_frames - 1 in missing_indices_1 else 0
        if max(cnt1, cnt2) > 0:
            if cnt1 > cnt2:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}\t{:^5d}\t{:^3d}'.format(ske_name, num_frames,
                                                                            cnt1, cnt2, start, end)
                missing_skes_logger1.info(info)
            else:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}'.format(ske_name, num_frames, cnt1, cnt2)
                missing_skes_logger2.info(info)

    # Find valid frame indices that the data is not missing or lost
    # For two-subjects action, this means both data of actor1 and actor2 is missing.
    valid_indices = np.where(joints.sum(axis=1) != 0)[0]  # 0-based index
    missing_indices = np.where(joints.sum(axis=1) == 0)[0]
    num_missing = len(missing_indices)

    if num_missing > 0:  # Update joints and colors
        joints = joints[valid_indices]
        colors[missing_indices] = np.nan
        missing_count += 1
        missing_skes_logger.info('{}\t{:^10d}\t{:^11d}'.format(ske_name, num_frames, num_missing))

    return joints, colors


def get_bodies_info(bodies_data):
    bodies_info = '{:^17}\t{}\t{:^8}\n'.format('bodyID', 'Interval', 'Motion')
    for (bodyID, body_data) in bodies_data.items():
        start, end = body_data['interval'][0], body_data['interval'][-1]
        bodies_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), body_data['motion'])

    return bodies_info + '\n'


def get_two_actors_points(bodies_data, fail_logger_1, fail_logger_2, actors_info_dir, noise_len_thres,
                          noise_len_logger, noise_spr_thres2, noise_spr_logger, noise_spr_thres1):
    """
    Get the first and second actor's joints positions and colors locations.

    # Arguments:
        bodies_data (dict): 3 key-value pairs: 'name', 'data', 'num_frames'.
        bodies_data['data'] is also a dict, while the key is bodyID, the value is
        the corresponding body_data which is also a dict with 4 keys:
          - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
          - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
          - interval: a list which records the frame indices.
          - motion: motion amount

    # Return:
        joints, colors.
    """
    joints = None
    colors = None
    ske_name = bodies_data['name']
    label = int(ske_name[-2:])
    num_frames = bodies_data['num_frames']
    bodies_info = get_bodies_info(bodies_data['data'])

    bodies_data, noise_info = denoising_bodies_data(bodies_data, noise_len_thres, noise_len_logger, noise_spr_thres2,
                                                    noise_spr_logger, noise_spr_thres1)  # Denoising data
    bodies_info += noise_info

    bodies_data = list(bodies_data)
    if len(bodies_data) == 1:  # Only left one actor after denoising
        if label >= 50:  # DEBUG: Denoising failed for two-subjects action
            fail_logger_2.info(ske_name)
        # print(len(bodies_data))
        bodyID, body_data = bodies_data[0]
        joints, colors = get_one_actor_points(body_data, num_frames)
        bodies_info += 'Main actor: %s' % bodyID
        with open(osp.join(actors_info_dir, ske_name + '.txt'), 'w') as fw:
            fw.write(bodies_info + '\n')
    elif len(bodies_data) > 1:
        if label < 50:  # DEBUG: Denoising failed for one-subject action
            fail_logger_1.info(ske_name)

        joints = np.zeros((num_frames, 75), dtype=np.float32)
        colors = np.ones((num_frames, 2, 25, 2), dtype=np.float32) * np.nan
        # print(len(bodies_data))
        bodyID, actor1 = bodies_data[0]  # the 1st actor with largest motion
        start1, end1 = actor1['interval'][0], actor1['interval'][-1]
        joints[start1:end1 + 1, :] = actor1['joints'].reshape(-1, 75)
        colors[start1:end1 + 1, 0] = actor1['colors']
        actor1_info = '{:^17}\t{}\t{:^8}\n'.format('Actor1', 'Interval', 'Motion') + \
                      '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start1, end1]), actor1['motion'])
        del bodies_data[0]

        # actor2_info = '{:^17}\t{}\t{:^8}\n'.format('Actor2', 'Interval', 'Motion')
        # start2, end2 = [0, 0]  # initial interval for actor2 (virtual)

        # while len(bodies_data) > 0:
        #     bodyID, actor = bodies_data[0]
        #     start, end = actor['interval'][0], actor['interval'][-1]
        #     if min(end1, end) - max(start1, start) <= 0:  # no overlap with actor1
        #         joints[start:end + 1, :75] = actor['joints'].reshape(-1, 75)
        #         colors[start:end + 1, 0] = actor['colors']
        #         actor1_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
        #         # Update the interval of actor1
        #         start1 = min(start, start1)
        #         end1 = max(end, end1)
        #     elif min(end2, end) - max(start2, start) <= 0:  # no overlap with actor2
        #         joints[start:end + 1, 75:] = actor['joints'].reshape(-1, 75)
        #         colors[start:end + 1, 1] = actor['colors']
        #         actor2_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
        #         # Update the interval of actor2
        #         start2 = min(start, start2)
        #         end2 = max(end, end2)
        #     del bodies_data[0]

        bodies_info += ('\n' + actor1_info) # + '\n' + actor2_info)

        with open(osp.join(actors_info_dir, ske_name + '.txt'), 'w') as fw:
            fw.write(bodies_info + '\n')

    return joints, colors


def get_raw_denoised_data(raw_data_file, save_path, fail_logger_1, fail_logger_2, actors_info_dir, noise_len_thres,
                          noise_len_logger, noise_spr_thres2, noise_spr_logger, noise_spr_thres1, missing_skes_logger1,
                          missing_skes_logger2, missing_skes_logger, missing_count, num_class):
    """
    Get denoised data (joints positions and color locations) from raw skeleton sequences.

    For each frame of a skeleton sequence, an actor's 3D positions of 25 joints represented
    by an 2D array (shape: 25 x 3) is reshaped into a 75-dim vector by concatenating each
    3-dim (x, y, z) coordinates along the row dimension in joint order. Each frame contains
    two actor's joints positions constituting a 150-dim vector. If there is only one actor,
    then the last 75 values are filled with zeros. Otherwise, select the main actor and the
    second actor based on the motion amount. Each 150-dim vector as a row vector is put into
    a 2D numpy array where the number of rows equals the number of valid frames. All such
    2D arrays are put into a list and finally the list is serialized into a cPickle file.

    For the skeleton sequence which contains two or more actors (mostly corresponds to the
    last 11 classes), the filename and actors' information are recorded into log files.
    For better understanding, also generate RGB+skeleton videos for visualization.
    """

    with open(raw_data_file, 'rb') as fr:  # load raw skeletons data
        raw_skes_data = pickle.load(fr)

    num_skes = len(raw_skes_data)
    print('Found %d available skeleton sequences.' % num_skes)

    raw_denoised_joints = []
    raw_denoised_colors = []
    frames_cnt = []

    for (idx, bodies_data) in enumerate(raw_skes_data):
        ske_name = bodies_data['name']
        # print('Processing %s' % ske_name)
        num_bodies = len(bodies_data['data'])

        if num_bodies == 1:  # only 1 actor
            num_frames = bodies_data['num_frames']
            body_data = list(bodies_data['data'].values())[0]
            joints, colors = get_one_actor_points(body_data, num_frames)
            raw_denoised_joints.append(joints)
            raw_denoised_colors.append(colors)
            frames_cnt.append(num_frames)
        elif num_bodies > 1:  # more than 1 actor, select two main actors
            joints, colors = get_two_actors_points(bodies_data, fail_logger_1, fail_logger_2, actors_info_dir,
                                                   noise_len_thres, noise_len_logger, noise_spr_thres2,
                                                   noise_spr_logger, noise_spr_thres1)
            if joints is not None:
                # Remove missing frames
                joints, colors = remove_missing_frames(ske_name, joints, colors, missing_skes_logger1, missing_skes_logger2,
                                                       missing_skes_logger, missing_count)
                num_frames = joints.shape[0]  # Update
                # Visualize selected actors' skeletons on RGB videos.
                raw_denoised_joints.append(joints)
                raw_denoised_colors.append(colors)
                frames_cnt.append(num_frames)

        if (idx + 1) % 10000 == 0:
            print('Processed: %.2f%% (%d / %d), ' % \
                  (100.0 * (idx + 1) / num_skes, idx + 1, num_skes) + \
                  'Missing count: %d' % missing_count)

    raw_skes_joints_pkl = osp.join(save_path, 'raw_denoised_joints_' + num_class + '.pkl')
    with open(raw_skes_joints_pkl, 'wb') as f:
        pickle.dump(raw_denoised_joints, f, pickle.HIGHEST_PROTOCOL)

    raw_skes_colors_pkl = osp.join(save_path, 'raw_denoised_colors_' + num_class + '.pkl')
    with open(raw_skes_colors_pkl, 'wb') as f:
        pickle.dump(raw_denoised_colors, f, pickle.HIGHEST_PROTOCOL)

    frames_cnt = np.array(frames_cnt, dtype=int)
    np.savetxt(osp.join(save_path, 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw denoised positions of {} frames into {}'.format(np.sum(frames_cnt),
                                                                     raw_skes_joints_pkl))
    print('Found %d files that have missing data' % missing_count)


def load_data(path):
    """

    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_data(data, lengths):
    num_scene = 0
    for scene in data:
        it = 0
        for frame in scene:
            if it == 0:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                xs = frame[:, 0]
                ys = frame[:, 1]
                zs = frame[:, 2]
                colors = ['g', 'g', 'g', 'g', 'b', 'b', 'b', 'b', 'r', 'r', 'r', 'r', 'c', 'c', 'c', 'c', 'm', 'm',
                          'm', 'm', 'g', 'b', 'b', 'r', 'r']
                labels = [str(i) for i in range(len(colors))]
                for i in range(len(xs)):
                    ax.scatter(xs[i], ys[i], zs[i], color=colors[i], label=labels[i])
            else:
                xs = frame[:, 0]
                ys = frame[:, 1]
                zs = frame[:, 2]
                ax.scatter(xs, ys, zs, color=colors)
            if it == lengths[num_scene]:
                ax.set_xlim(np.amin(scene), np.amax(scene))
                ax.set_ylim(np.amin(scene), np.amax(scene))
                ax.set_zlim(np.amin(scene), np.amax(scene))
                plt.legend()
                plt.show()
                break
            it += 1
            print(it, '/', len(scene))
        num_scene += 1


def main():
    

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conf', action='append')
    args = parser.parse_args()
    args_json = []
    if args.conf is not None:
        for conf_fname in args.conf:
            with open(conf_fname, 'r') as f:
                d = json.load(f)
            for key, value in d.items():
                parser.add_argument(key)
                args_json.extend((key, value))
        # Reload arguments to override config file values with command line values
        args = parser.parse_args(args_json)
    data = load_data(args.train_pkl_path)

    normalizer = augmentations.Normalize3D()
    masker = augmentations.RandomMasking(augmentations={'frames': 0.5, 'joints': 0.5})
    noiser = augmentations.RandomAdditiveNoise(dist='NORMAL', prob=0.5, std=0.01) ### ERROR EDITED
    augment = torchvision.transforms.Compose([normalizer, masker, noiser])
    lengths = []
    for video in data['annotations']:
        key_points = video['keypoint'][0]
        data['annotations'][len(lengths)]['keypoint'] = video['keypoint'][0]
        lengths.append(key_points.shape[0])
    x = np.zeros((len(data['annotations']), max(lengths), 25, 3))
    i = 0
    for video in data['annotations']:
        augment(video)
        x[i, :video['keypoint'][0].shape[0]] = video['keypoint'][0]
        i += 1

    plot_data(x, lengths)


if __name__ == '__main__':
    main()
