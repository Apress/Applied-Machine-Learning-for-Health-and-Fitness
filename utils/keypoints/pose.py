import numpy as np
import glob
import json
import os

class PoseSequence:
    def __init__(self, sequence):
        self.poses = []
        for parts in sequence:
            self.poses.append(Pose(parts))
        
        # normalize poses based on the average torso pixel length
        torso_lengths = np.array([Part.dist(pose.neck, pose.lhip) for pose in self.poses if pose.neck.exists and pose.lhip.exists] +
                                 [Part.dist(pose.neck, pose.rhip) for pose in self.poses if pose.neck.exists and pose.rhip.exists])
        mean_torso = np.mean(torso_lengths)
        

        for pose in self.poses:
            for attr, part in pose:
                setattr(pose, attr, part / mean_torso)

    @staticmethod
    def load(filename):
        return PoseSequence(np.load(filename))

    @staticmethod
    def parse_sequence(num_keypoints, json_folder, output_folder):
        json_files = glob.glob(os.path.join(json_folder, '*.json'))
        json_files = sorted(json_files)

        num_frames = len(json_files)
        keypoints = np.zeros((num_frames, num_keypoints, 3))
        for i in range(num_frames):
            with open(json_files[i]) as f:
                json_obj = json.load(f)
                pose_keypoints_2d = np.array(json_obj['people'][0]['pose_keypoints_2d'])
                keypoints[i] = pose_keypoints_2d.reshape((num_keypoints, 3))
        
        output_dir = os.path.join(output_folder, os.path.basename(json_folder))
        np.save(output_dir, keypoints)



class Pose:
    PART_NAMES = ['nose', 'neck',  'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear']

    def __init__(self, parts):
        """Construct a pose for one frame, given an array of parts

        Arguments:
            parts - 18 * 3 ndarray of x, y, confidence values
        """
        for name, vals in zip(self.PART_NAMES, parts):
            setattr(self, name, Part(vals))
    
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value
    
    def __str__(self):
        out = ""
        for name in self.PART_NAMES:
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ +"\n"
        return out
    
    def print(self, parts):
        out = ""
        for name in parts:
            if not name in self.PART_NAMES:
                raise NameError(name)
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ +"\n"
        return out

class Part:
    def __init__(self, vals):
        self.x = vals[0]
        self.y = vals[1]
        self.c = vals[2]
        self.exists = self.c != 0.0

    def __floordiv__(self, scalar):
        __truediv__(self, scalar)

    def __truediv__(self, scalar):
        return Part([self.x / scalar, self.y / scalar, self.c])

    @staticmethod
    def dist(part1, part2):
        return np.sqrt(np.square(part1.x - part2.x) + np.square(part1.y - part2.y))