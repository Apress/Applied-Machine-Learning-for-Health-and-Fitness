import gym
import pybullet as p
import pybullet_data as pd
from pybulletgym.tests.roboschool.agents.policies import SmallReactivePolicy
import pybulletgym.tests.roboschool.agents.HumanoidPyBulletEnv_v0_2017may as HumanoidWeights
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from numpngw import write_apng
from IPython.display import Image

def load_humanoid(trained = False):
    env_name = 'HumanoidPyBulletEnv-v0'
    env = gym.make(env_name)
    
    if trained:
        weights = {
             'HumanoidPyBulletEnv-v0': [[HumanoidWeights.weights_dense1_w,
                                         HumanoidWeights.weights_dense2_w,
                                        HumanoidWeights.weights_final_w],
                                        [HumanoidWeights.weights_dense1_b,
                                        HumanoidWeights.weights_dense2_b,
                                        HumanoidWeights.weights_final_b]]
        }

        env.seed(7)   # Fix random seed to achieve determinism
        agent = SmallReactivePolicy(env.observation_space, env.action_space, 
                                    weights[env_name][0],  # weights
                                    weights[env_name][1])  # biases
        return env, agent
    else:
        return env, None
    

def plot_humanoid():
    camTargetPos = [0, 0, 0]
    cameraUp = [0, 0, 1]
    cameraPos = [1, 1, 1]
    p.setGravity(0, 0, -10)
    pitch = -10.0
    roll = 0
    upAxisIndex = 2
    camDistance = 3
    pixelWidth = 1080
    pixelHeight = 790
    nearPlane = 0.01
    farPlane = 100
    fov = 90
    cubeStartPos = [0,0,1]
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    yaw = 0
    
    for r in range(2):
        for c in range(2):
            yaw += 60
            pylab.figure(figsize=(10, 5))
            viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                                          roll, upAxisIndex)
            aspect = pixelWidth / pixelHeight
            projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
            img_arr = p.getCameraImage(pixelWidth,pixelHeight,viewMatrix,projectionMatrix)
            w = img_arr[0]  
            h = img_arr[1]  
            rgb = img_arr[2] 
            dep = img_arr[3] 
            np_img_arr = np.reshape(rgb, (h, w, 4))
            np_img_arr = np_img_arr * (1. / 255.)
            pylab.imshow(np_img_arr, interpolation='none', animated=True)

def animated_humanoid(fn):
    camTargetPos = [0, 0, 0]
    cameraUp = [0, 0, 1]
    cameraPos = [1, 1, 1]
    p.setGravity(0, 0, -10)
    pitch = -10.0
    roll = 0
    upAxisIndex = 2
    camDistance = 3
    pixelWidth = 1080
    pixelHeight = 790
    nearPlane = 0.01
    farPlane = 100
    fov = 90
    cubeStartPos = [0,0,1]
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    yaw = 0
    frames=[] 
    print("Creating animated png, please wait about 5 seconds")
    for r in range(60):
        yaw += 6
        pitch = -10.0
        roll = 0
        upAxisIndex = 2
        camDistance = 4
        pixelWidth = 320
        pixelHeight = 200
        nearPlane = 0.01
        farPlane = 100
        fov = 60
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                                    roll, upAxisIndex)
        aspect = pixelWidth / pixelHeight
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

        img_arr = p.getCameraImage(pixelWidth,pixelHeight,viewMatrix,projectionMatrix)
        w = img_arr[0]  
        h = img_arr[1]  
        rgb = img_arr[2]  
        dep = img_arr[3]  
        np_img_arr = np.reshape(rgb, (h, w, 4))
        frame = np_img_arr[:, :, :3]
        frames.append(frame)
    write_apng(fn, frames, delay=100)
    

def get_camera_image():
    w = 1080
    h = 720
    fov = 60
    viewMatrix = p.computeViewMatrixFromYawPitchRoll([0,0,0], 4, fov, -10, 0, 2)
    projectionMatrix = p.computeProjectionMatrixFOV(fov, w/h, 0.01, 100);
    return p.getCameraImage(w, h, viewMatrix,projectionMatrix)
    

