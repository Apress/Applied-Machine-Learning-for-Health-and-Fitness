import os
import sys

import numpy as np
import matplotlib.pyplot as plt

def split_num(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return head, tail

def files_in_order(folderpath):
    npy_files = os.listdir(folderpath)

    no_extensions = [os.path.splitext(npy_file)[0] for npy_file in npy_files]

    splitted = [split_num(s) for s in no_extensions]

    splitted = np.array(splitted)

    indices = np.lexsort((splitted[:, 1].astype(int), splitted[:, 0]))

    npy_files = np.array(npy_files)
    return npy_files[indices]

# Generates binary labels (good=1, bad=0) given an array-like of filenames
def get_labels(array):
    labels = [1 if "good" in i else 0 for i in array]
    return np.array(labels)

# Generates binary labels (expert=1, beginner=0) given an array-like of filenames
def get_labels_by_level(array):
    labels = [1 if "expert" in i else 0 for i in array]
    return np.array(labels)


def get_side(poses):
    right_present = [1 for pose in poses 
            if pose.rshoulder.exists and pose.relbow.exists and pose.rwrist.exists]
    left_present = [1 for pose in poses
            if pose.lshoulder.exists and pose.lelbow.exists and pose.lwrist.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    side = 'right' if right_count > left_count else 'left'
    
    return side

def get_joints(poses):
    joints = [joint for joint in poses if all(part.exists for part in joint)]
    joints = np.array(joints)
    return joints

def get_normalized_joint_vector(joints, p1, p2):
    v = np.array([(joint[p1].x - joint[p2].x, joint[p1].y - joint[p2].y) for joint in joints])
    return v / np.expand_dims(np.linalg.norm(v, axis=1), axis=1)

def get_angle(v1,v2):
    return np.degrees(np.arccos(np.clip(np.sum(np.multiply(v1, v2), axis=1), -1.0, 1.0)))

def chart(v, name, ylabel, c):
    plt.scatter(np.arange(v.shape[0]), v, alpha=0.5, color=c)
    plt.title(name)
    plt.xlabel('Frames')
    plt.ylabel(ylabel)
    # Set range on y-axis so the plots are consistent
    plt.ylim(0,120) 
    plt.show()
    #plt.savefig('image_'+name+'.png', dpi=300)

def filter_joints(joints):
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    joints = np.array(joints)

# Compute Dynamic Time Warp Distance of two sequences
# http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])