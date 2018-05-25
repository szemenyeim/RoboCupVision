import os.path as osp
import glob
import cv2
import numpy as np
import re
import os
from sklearn.cluster import KMeans
import pickle

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

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def convert(root, split="train"):

    labels = []
    images = []
    preds = []

    data_dir = osp.join(root, split)
    lab_dir = osp.join(data_dir,"labels")
    img_dir = osp.join(data_dir,"images")

    for file in sorted(glob.glob1(lab_dir, "*.png"),key=alphanum_key):
        labels.append(file)
    for file in sorted(glob.glob1(img_dir, "*.png"),key=alphanum_key):
        images.append(file)

    for file, img in zip(labels,images):
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

        ball = []
        maxArea = max(areas) if len(areas) > 0 else 0
        for area, cand in sorted(zip(areas, candidates)):
            if area >= maxArea*0.05 and len(ball) < 6:
                ball.append(cand)
                pred.append([1, np.asarray(cand)])

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
            if area >= maxArea*0.05 and len(robot) < 5:
                robot.append(cand)
                pred.append([2, np.asarray(cand)])

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
            if area >= maxArea*0.2 and len(goal) < 2:
                goal.append(cand)
                pred.append([3, np.asarray(cand)])


        '''orig_file = osp.join(img_dir,img)
        img = cv2.imread(orig_file)
        for i, elem in enumerate(pred):
            if i >= 1 and elem[0] > 0:
                color = (0,0,255) if elem[0] == 1 else ((0,255,0) if elem[0] == 2 else (255,0,0))
                rect = list(elem[1])
                pt1 = rect[0:2]
                pt2 = (rect[0] + rect[2], rect[1] + rect[3])
                img = cv2.rectangle(img,tuple(pt1),tuple(pt2),color,3)
        cv2.imshow('img',img)
        cv2.waitKey(0)'''


        preds.append(pred)

    # do clustering
    ballRects = np.empty((0,4))
    goalRects = np.empty((0,4))
    robotRects = np.empty((0,4))

    for pred in preds:
        for i, elem in enumerate(pred):
            if i > 0 and elem[0] > 0:
                if elem[0] == 1:
                    ballRects = np.append(ballRects,[elem[1]],axis=0)
                elif elem[0] == 2:
                    robotRects =np.append(robotRects,[elem[1]],axis=0)
                else:
                    goalRects = np.append(goalRects,[elem[1]],axis=0)

    bMean = np.mean(ballRects,axis=0)
    kmr = KMeans(5).fit(robotRects)
    kmg = KMeans(2).fit(goalRects)

    np.save(osp.join(data_dir,'bMean.npy'),bMean)
    np.save(osp.join(data_dir,'rMean.npy'),kmr)
    np.save(osp.join(data_dir,'gMean.npy'),kmg)

    with open(osp.join(data_dir,'preds.pickle'), 'wb') as f:
        pickle.dump(preds, f)

if __name__ == '__main__':
    convert(root='./data/')