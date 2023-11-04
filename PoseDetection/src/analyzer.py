import cv2
import mediapipe as mp
import numpy as np
import os
import itertools
import csv
import sys
import numpy as np

sys.path.insert(1, '../../GolfDB')

from GolfDB.detectPoses import detectEvents

import pandas as pd

event_names = {
        0: 'Address',
        1: 'Toe-up',
        2: 'Mid-backswing (arm parallel)',
        3: 'Top',
        4: 'Mid-downswing (arm parallel)',
        5: 'Impact',
        6: 'Mid-follow-through (shaft parallel)',
        7: 'Finish'
    }

def analyzeEvents(input_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(input_path)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates
                landmark_points = []

                for index, landmark in enumerate(landmarks):
                    landmark_points.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

                cv2.imshow('Mediapipe Feed', image)

                return landmark_points

            except Exception as e:
                print(f'Except: {e}')
                pass

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def writeData():
    # traverse root directory, and list directories as dirs and files as files
    frontDirs = [dir[0] for dir in os.walk("../proEvents/front")]
    frontDirs = frontDirs[1:]
    # print(frontDirs)

    sideDirs = [dir[0] for dir in os.walk("../proEvents/side")]
    sideDirs = sideDirs[1:]

    allDirs = [frontDirs, sideDirs]
    isFront = True

    print('Writing data...')

    for sfDir in allDirs:

        dataDict = {}

        for dir in sfDir:
            print(f'Writing for Dir: {dir}')
            for filename in os.listdir(dir):
                f = os.path.join(dir, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    event = filename.split('.mp4_')[-1].split('.jpg')[0]
                    filen = filename.split('.mp4_')[0]

                    points = analyzeEvents(f)

                    if filen not in dataDict:
                        dataDict[filen] = []

                    dataDict[filen] = points

        if isFront:
            with open('../output/frontData.csv', mode='w+', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for key, val in dataDict.items():
                    flatList = list(itertools.chain.from_iterable(val))
                    print(f'key: {key} flat: {flatList}')
                    flatList.insert(0, key)
                    csv_writer.writerow(flatList)

        else:
            with open('../output/sideData.csv', mode='w+', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for key, val in dataDict.items():
                    flatList = list(itertools.chain.from_iterable(val))
                    print(f'key: {key} flat: {flatList}')
                    flatList.insert(0, key)
                    csv_writer.writerow(flatList)

        isFront = False
    print('Done writing data')

def findMinDistance(inputVideo, perspective):
    """
    Find the minimum distance between the landmarks in the inputFile and the landmarks for the pro golfers.
    For each event, the euclidean distance between analogous landmarks is calculated.
    The sum of the distances over all events against a player is the distance between the input and the player.
    The player with the minimum distance is the closest match.
    :param inputVideo: path to the input video
    :param perspective: "Front" or "Side"
    :return: player name, distance
    """

    inputList = []
    compList = []

    processInput(inputVideo)

    with open('../input/inputData.csv', mode='r') as f:
        inputList = list(csv.reader(f, delimiter=','))

    if perspective == 'Front':
        with open('../output/frontData.csv', mode='r') as f:
            compList = list(csv.reader(f, delimiter=','))
    else:
        with open('../output/sideData.csv', mode='r') as f:
            compList = list(csv.reader(f, delimiter=','))

    inputName = inputList[0][0]
    inputData = inputList[0][1:]

    inputChunks = [inputData[i:i+4] for i in range(0, 132, 4)]

    compDict = {}

    for row in compList:
        data = row[1:]
        name = row[0]
        chunks = [data[i:i+4] for i in range(0, 132, 4)]

        compDict[name] = chunks

    minDist = float('inf')
    minName = ''

    for key, val in compDict.items():
        dist = 0
        for i in range(len(val)):
            ix, iy, iz, iv = inputChunks[i]
            cx, cy, cz, cv = val[i]

            # Make sure the landmark is visible
            if cv > 0:
                tempDist = np.sqrt((float(ix) - float(cx))**2 + (float(iy) - float(cy))**2 + (float(iz) - float(cz))**2)
                dist += tempDist

        print(f'Name: {key} Dist: {dist}')

        if dist < minDist:
            minDist = dist
            minName = key

    return minName, minDist

def addProEvents(input, perspective):
    """
    Add pro events to the database. Calls GolfDB.detectPoses.detectEvents and adds events to the proEvents/perspective folder.
    Make sure to create a new folder following the convention "FirstName-LastName_Club_Perspective#" and add the generated events to it.
    :param input: path to the input video
    :param perspective: "Front" or "Side"
    :return:
    """
    if perspective == 'Front':
        outDir = '../proEvents/front'
    else:
        outDir = '../proEvents/side'

    if os.path.isfile(input):
        print('File exists')
        detectEvents(input, outDir)
    else:
        print('File does not exist')

def viewLandmarks(input_path):
    """
    View the mediapipe landmarks for a given video
    :param input_path: video path
    :return:
    """

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(input_path)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates
                landmark_points = []

                for index, landmark in enumerate(landmarks):
                    landmark_points.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                # cv2.imwrite('../output' + filename, image)

                cv2.imshow('Mediapipe Feed', image)

            except Exception as e:
                print(f'Except: {e}')
                pass

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def processInput(input_path):
    """
    Process the input video and generate a csv file with the landmarks.
    Generates events and places them in input/inputEvents.
    Generates landmarks and places the data in a csv file in input/inputData.csv
    :param input_path: path to the input video
    :return:
    """
    if os.path.isfile(input_path):
        print('File exists')
        print('Detecting events...')
        detectEvents(input_path, '../input/inputEvents')
        print('Done detecting events')
        print('Generating csv...')

        dataDict = {}

        for filename in os.listdir('../input/inputEvents'):
            f = os.path.join('../input/inputEvents', filename)
            # checking if it is a file
            if os.path.isfile(f):
                event = filename.split('.mp4_')[-1].split('.jpg')[0]
                filen = filename.split('.mp4_')[0]

                points = analyzeEvents(f)

                if filen not in dataDict:
                    dataDict[filen] = []

                dataDict[filen] = points

        with open('../input/inputData.csv', mode='w+', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for key, val in dataDict.items():
                flatList = list(itertools.chain.from_iterable(val))
                print(f'key: {key} flat: {flatList}')
                flatList.insert(0, key)
                csv_writer.writerow(flatList)

        print('Done generating csv')

    else:
        print('File does not exist')


if __name__ == "__main__":

    # input = '../videos/HidekiMatsuyama/Hideki-Matsuyama_LongIrons_Side1.mp4'

    # addProEvents(input, 'Side')


    # test1 = '../proEvents/front/Adam-Scott_LongIrons_Front1/Adam-Scott_LongIrons_Front1.mp4_Address.jpg'
    # testDir = '../proEvents/front/Adam-Scott_LongIrons_Front1'
    #
    writeData()

    # viewLandmarks('../videos/ColinMorikawa/Colin-Morikawa_LongIrons_Side1.mp4')

    # processInput('../input/test_video.mp4')

    # minName, minDist = findMinDistance('../output/sample.csv', 'Side')
    # print(f'MinName: {minName} MinDist: {minDist}')






