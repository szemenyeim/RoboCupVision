import torch
import os.path as osp
from torch.utils import data
import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import re
import os

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def default_loader(path):
    return Image.open(path)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def convert(root, split="train"):

    labels = []
    preds = []

    data_dir = osp.join(root, split)
    lab_dir = osp.join(data_dir,"labels")

    for file in sorted(glob.glob1(lab_dir, "*.png"),key=alphanum_key):
        labels.append(file)

    cntr = 0

    for file in labels:
        path = osp.join(lab_dir,file)
        label = cv2.imread(path,0)

        pred = [file]

        # Detect balls
        balls = np.array(label == 1, dtype=np.uint8)
        _,cont,_ = cv2.findContours(balls,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        areas = []
        minArea = 25
        for candidate in cont:
            area = cv2.contourArea(candidate)
            if area > minArea:
                candidates.append( cv2.boundingRect(candidate))
                areas.append(  area )

        if len(areas) > 0:
            ball = []
            maxArea = max(areas)
            for cand,area in zip(candidates,areas):
                if area >= maxArea:
                    ball.append(cand)

            pred.append([1,ball[0]])
        else:
            pred.append([0,(0,0,0,0)])

        # Detect robots
        robots = np.array(label == 2, dtype=np.uint8)
        _, cont, _ = cv2.findContours(robots, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        areas = []
        minArea = 200
        for candidate in cont:
            area = cv2.contourArea(candidate)
            if area > minArea:
                candidates.append(cv2.boundingRect(candidate))
                areas.append(area)

        robot = []
        maxArea = max(areas) if len(areas) > 0 else 0
        for area, cand in sorted(zip(areas, candidates)):
            if area >= maxArea*0.2 and len(robot) < 5:
                robot.append(cand)
                pred.append([1, cand])

        for i in range(len(robot),5):
            pred.append([0, (0, 0, 0, 0)])

        # Detect goals
        goals = np.array(label == 3, dtype=np.uint8)
        _, cont, _ = cv2.findContours(goals, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        areas = []
        minArea = 30
        for candidate in cont:
            area = cv2.contourArea(candidate)
            if area > minArea:
                candidates.append(cv2.boundingRect(candidate))
                areas.append(area)

        goal = []
        maxArea = max(areas) if len(areas) > 0 else 0
        for area, cand in sorted(zip(areas, candidates)):
            if area >= maxArea*0.25 and len(goal) < 2:
                goal.append(cand)
                pred.append([1, cand])

        cntr += len(goal)
        for i in range(len(goal),2):
            pred.append([0, (0, 0, 0, 0)])

        lines = np.array(label == 4, dtype=np.uint8)
        img = np.zeros(lines.shape, dtype=np.uint8)
        edges = np.zeros(lines.shape, dtype=np.uint8)
        _,cont,_ = cv2.findContours(lines, mode=cv2.RETR_LIST , method=cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edges,cont,-1,255,thickness=1)
        hough = cv2.HoughLinesP(edges,2,np.pi/180,50,20,20)
        if hough is not None:
            for h in hough:
                for x1, y1, x2, y2 in h:
                    cv2.line(img, (x1, y1), (x2, y2), (255), 2)
            cv2.imshow("line",edges)
            cv2.imshow("hough",img)
            cv2.waitKey(0)
        else:
            print('None')




        #print(pred)


        preds.append(pred)

    print(cntr)
    print(len(labels))

if __name__ == '__main__':
    convert(root='./data/FinetuneHorizon/')