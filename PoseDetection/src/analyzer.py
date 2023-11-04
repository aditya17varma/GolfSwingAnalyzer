import cv2
import mediapipe as mp
import numpy as np
import os
import itertools
import csv
import sys
import numpy as np

from submodule.GolfDB.detectPoses import detectEvents

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
    :return: player name, minimum distance, distance dictionary
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

    distanceDict = {}

    for key, val in compDict.items():
        dist = 0
        for i in range(len(val)):
            ix, iy, iz, iv = inputChunks[i]
            cx, cy, cz, cv = val[i]

            # Make sure the landmark is visible
            if float(cv) and float(iv) > 0:
                tempDist = np.sqrt((float(ix) - float(cx))**2 + (float(iy) - float(cy))**2 + (float(iz) - float(cz))**2)
                dist += tempDist

        distanceDict[key] = dist

        if dist < minDist:
            minDist = dist
            minName = key

    return minName, minDist, distanceDict

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
    View the mediapipe landmarks for a given video file.
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
                # print(f'key: {key} flat: {flatList}')
                flatList.insert(0, key)
                csv_writer.writerow(flatList)

        print('Done generating csv')

    else:
        print('File does not exist')

def createLandmarkImage(filename, outputFolder):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(filename)

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

                cv2.imshow('Mediapipe Feed', image)

                outputPath = os.path.join(outputFolder, filename.split('/')[-1])
                # print(f'Writing to {outputPath}')
                cv2.imwrite(outputPath, image)

                cap.release()
                cv2.destroyAllWindows()

                return

            except Exception as e:
                print(f'Except: {e}')
                pass

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def createComparisonEvents(playerFile, perspective):
    """
    Create events for the given player.
    :param playerFile: path to the player file
    :return:
    """
    if perspective == 'Front':
        playerEventFolder = '../proEvents/front/' + playerFile
    else:
        playerEventFolder = '../proEvents/side/' + playerFile


    print()
    print('-------------------')

    print(f'Creating events for {playerFile}')
    print(f'Player event folder: {playerEventFolder}')

    for filename in os.listdir(playerEventFolder):
        f = os.path.join(playerEventFolder, filename)
        # checking if it is a file
        if os.path.isfile(f):
            createLandmarkImage(f, '../output/compPlayer')

    for filename in os.listdir('../input/inputEvents'):
        f = os.path.join('../input/inputEvents', filename)
        # checking if it is a file
        if os.path.isfile(f):
            createLandmarkImage(f, '../output/compInput')

def orderImages(folder1_images, folder2_images):
    """
    Order the images in the two folders in order of golf swing.
    :param folder1_images: folder 1 images
    :param folder2_images: folder 2 images
    :return:
    """

    folder1_ordered = [None] * len(folder1_images)
    folder2_ordered = [None] * len(folder2_images)

    # 0: 'Address',
    # 1: 'Toe-up',
    # 2: 'Mid-backswing (arm parallel)',
    # 3: 'Top',
    # 4: 'Mid-downswing (arm parallel)',
    # 5: 'Impact',
    # 6: 'Mid-follow-through (shaft parallel)',
    # 7: 'Finish'
    for f1 in folder1_images:
        if f1.endswith('Address.jpg'):
            folder1_ordered[0] = f1
        elif f1.endswith('Toe-up.jpg'):
            folder1_ordered[1] = f1
        elif f1.endswith('Mid-backswing (arm parallel).jpg'):
            folder1_ordered[2] = f1
        elif f1.endswith('Top.jpg'):
            folder1_ordered[3] = f1
        elif f1.endswith('Mid-downswing (arm parallel).jpg'):
            folder1_ordered[4] = f1
        elif f1.endswith('Impact.jpg'):
            folder1_ordered[5] = f1
        elif f1.endswith('Mid-follow-through (shaft parallel).jpg'):
            folder1_ordered[6] = f1
        elif f1.endswith('Finish.jpg'):
            folder1_ordered[7] = f1
        else:
            print('Error')
            break

    for f2 in folder2_images:
        if f2.endswith('Address.jpg'):
            folder2_ordered[0] = f2
        elif f2.endswith('Toe-up.jpg'):
            folder2_ordered[1] = f2
        elif f2.endswith('Mid-backswing (arm parallel).jpg'):
            folder2_ordered[2] = f2
        elif f2.endswith('Top.jpg'):
            folder2_ordered[3] = f2
        elif f2.endswith('Mid-downswing (arm parallel).jpg'):
            folder2_ordered[4] = f2
        elif f2.endswith('Impact.jpg'):
            folder2_ordered[5] = f2
        elif f2.endswith('Mid-follow-through (shaft parallel).jpg'):
            folder2_ordered[6] = f2
        elif f2.endswith('Finish.jpg'):
            folder2_ordered[7] = f2
        else:
            print('Error')
            break

    return folder1_ordered, folder2_ordered

def displayComparisons():
    # Path to the two image folders
    folder1_path = '../output/compPlayer'
    folder2_path = '../output/compInput'

    # List all the image files in each folder
    folder1_images = [f for f in os.listdir(folder1_path) if f.endswith(('.jpg', '.png'))]
    folder2_images = [f for f in os.listdir(folder2_path) if f.endswith(('.jpg', '.png'))]

    # Sort the image files to ensure they are in the same order
    folder1_images, folder2_images = orderImages(folder1_images, folder2_images)


    # Iterate through the image files and display them side by side
    for img1_name, img2_name in zip(folder1_images, folder2_images):
        # Load the images
        image1 = cv2.imread(os.path.join(folder1_path, img1_name))
        image2 = cv2.imread(os.path.join(folder2_path, img2_name))

        # Check if the images have the same dimensions
        if image1.shape[:2] == image2.shape[:2]:
            # Images have the same dimensions, concatenate them side by side
            concatenated_image = cv2.hconcat([image1, image2])

            # Display the concatenated image
            cv2.imshow('Concatenated Images', concatenated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # Resize one of the images to match the dimensions of the other
            image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

            # Concatenate the resized image with the first image
            concatenated_image = cv2.hconcat([image1, image2_resized])

            # Display the concatenated image
            cv2.imshow('Concatenated Images', concatenated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

def main(videoInput, perspective='Front'):
    """
    Main function for the analyzer.
    :param videoInput: path to user input video
    :return:
    """
    clearFolders('../input/inputEvents')
    clearFolders('../output/compPlayer')
    clearFolders('../output/compInput')

    minName, minDist, distDict = findMinDistance(videoInput, perspective)
    minNameSplit = minName.split('_')
    playerName = minNameSplit[0]
    club = minNameSplit[1]

    print()
    print('-------------------')
    print('Distances: ')
    for key, dist in distDict.items():
        name = key.split('_')[0]
        print(f'{name:30} | {dist:30}')

    print()
    print(f'Closest Match to: {playerName} with {club} minDist: {minDist}')

    createComparisonEvents(minName, 'Side')

    displayComparisons()

def clearFolders(folder_path):
    """
    Clear all files from the inputEvents, output/compPlayer, and output/compInput folders.
    :param folder_path:
    :return:
    """
    # List all files in the folder
    files = os.listdir(folder_path)
    print(f'Clearing folder: {folder_path}')

    # Iterate through the files and delete them
    for file in files:
        file_path = os.path.join(folder_path, file)

        # Check if the path is a file (not a subdirectory)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                # print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":

    # for directory in sys.path:
    #     print(directory)

    # input = '../videos/HidekiMatsuyama/Hideki-Matsuyama_LongIrons_Side1.mp4'
    # detectEvents(input, '../output')

    # addProEvents(input, 'Side')


    # test1 = '../proEvents/front/Adam-Scott_LongIrons_Front1/Adam-Scott_LongIrons_Front1.mp4_Address.jpg'
    # testDir = '../proEvents/front/Adam-Scott_LongIrons_Front1'
    #
    # writeData()

    # viewLandmarks('../input/Shriya_Side1.mov')

    # processInput('../input/test_video.mp4')

    # minName, minDist = findMinDistance('../input/test_video.mp4', 'Side')
    # print()
    # print(f'MinName: {minName} MinDist: {minDist}')
    #
    # # createLandmarkImage('../proEvents/front/Adam-Scott_LongIrons_Front1/Adam-Scott_LongIrons_Front1.mp4_Address.jpg', '../output/compPlayer')
    #
    # createComparisonEvents(minName, 'Side')
    #
    # displayComparisons()


    main('../input/Shriya_Side1.mov', 'Side')
    # clearFolders('../input/inputEvents')







