from tkinter.tix import IMAGETEXT
from cmu_112_graphics import *
import cv2
import mediapipe as mp
import time
import os
#from BlazePose import BodyPose
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import math
from playsound import playsound
import copy

#Note: Citations will be done on this file as the other two files are just
#where I have been playing around with the individual detection software
#and here is where I have actually put them together
#https://google.github.io/mediapip
#https://github.com/Practical-CV/Color-Based-Ball-Tracking-With-OpenCV/blob/master/ball_tracking_mine.py




####################################
# Helper Functions:
####################################
def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def calculate_angle(a,b,c):
    
    BA = np.array(a) - np.array(b)
    BC = np.array(c) - np.array(b)


    BA_abs = math.sqrt(BA[0]**2 + BA[1]**2)
    BC_abs = math.sqrt(BC[0]**2 + BC[1]**2)

    numerator = np.array(BA) * np.array(BC)

    costheta = ((numerator[0] + numerator[1])/ (BA_abs * BC_abs))


    return np.arccos(costheta) * 180/np.pi

def FeedBack(L1, L2, juggles, points, tricks):
    
    img = np.zeros((800,450,3), np.uint8)
    img = imutils.resize(img, width=800, height = 450)
    for i in L1:
        if type(i[0]) == int:
            cx = i[0]
            cy = i[1]
            r = 4
            cv2.circle(img,(i[0],i[1]), r, (0,0,255), -1)

    for j in L2:
        cx = j[0]
        cy = j[1]
        r = 3
        cv2.circle(img,(int(j[0]),int(j[1])), r, (255,0,0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .9
    color = (0, 0, 255)
    thickness = 2


    cv2.putText(img, f'Juggles: {juggles}', (200, 50), font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.putText(img, f'Points: {int(points)}', (400, 50), font, fontScale, color, thickness, cv2.LINE_AA)
    
    cv2.putText(img, f'Tricks: {tricks}', (300, 100), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Canvas", img)
    cv2.waitKey(0)

    

####################################
# Juggling App:
####################################


def BodyPose(overlaps = 0, Juggles = 0, differences = 0, points = 0, offscreentime = 0):
    #30 - 33 GitHub
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")

    args = vars(ap.parse_args())

    #36 - 43 Based on external resources
    lower_red = np.array([161, 155, 84])
    upper_red = np.array([179, 255, 255])
    pts = deque(maxlen=args["buffer"])


    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic


    camera = cv2.VideoCapture(0)

    #For Video input:
    #53 - 56 GitHub

    currloc = (400,0) #Represents Ball Starting the top of screen
    ball_locs = []
    ball_intercepts = []
    TricksDone = ''
    QuitMode = False

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
            
                

        #57 - 181 Mostly me including slight manipulations of the cited code so they work together. 
            success, image = camera.read()
            if not success:
                break

            #Recolor image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
  

            #Make Detection
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #List Joints and Connections
            #Extract Coords and Compare to Ball Detection
            image = imutils.resize(image, width=800, height = 450)
            try:
                landmarks = results.pose_landmarks.landmark

                leftfootcoords = (landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])

                rightfootcoords = (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])


                currleftfoot = ((leftfootcoords.x)*800, (leftfootcoords.y)*450)
                currrightfoot = ((rightfootcoords.x)*800, (rightfootcoords.y)*450)

                
            except:
                pass 

            #Draw Joints and Connections
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                mp_drawing.DrawingSpec(color = (0,0,255), thickness=3, circle_radius=2),
                mp_drawing.DrawingSpec(color = (0,255,0), thickness=3, circle_radius=2))


            blurred = cv2.GaussianBlur(image, (11,11), 0) 
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv,lower_red,upper_red) #Change to red
            mask = cv2.erode(mask,None, iterations = 2)
            mask = cv2.dilate(mask, None, iterations  = 2) #Possibly Dilate 

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cnts = imutils.grab_contours(cnts)

            center = None

            #120 - 134 https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
            

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((x,y), radius) = cv2.minEnclosingCircle(c)


                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M['m01'] / M["m00"]))
                
                ball_locs.append(center) #DATA

                if radius > 10:
                    cv2.circle(image, (int(x), int(y)), int(radius), (0,255,255), 2)

                    cv2.circle(image,center,5, (0,0,255), -1)


        
            pts.appendleft(center)

            prevloc = currloc

            currloc = center

            try:
                if distance(center[0],center[1],currrightfoot[0],currrightfoot[1]) < radius+25:
                    contact = True
                    ball_intercepts.append((currrightfoot[0],currrightfoot[1]))
                
                if (currloc[1] - prevloc[1]) < 0:
                    differences += (currloc[1] - prevloc[1])

                elif (currloc[1] - prevloc[1]) > 0:
                    differences = 0 

                if (abs(differences) >= 60) :
                    up = True


                if (up == True) and (differences == 0) and (contact == True):
                    Juggles += 1
                    points += 1 * (Juggles/10)
                    up = False
                    differences = 0
                    contact = False


            except:
                pass
                
            try:
                if distance(center[0],center[1],currleftfoot[0],currleftfoot[1]) < radius+25:
                    contact = True
                    ball_intercepts.append((currleftfoot[0],currleftfoot[1]))
                
                if (currloc[1] - prevloc[1]) < 0:
                    differences += (currloc[1] - prevloc[1])

                elif (currloc[1] - prevloc[1]) > 0:
                    differences = 0 

                if (abs(differences) >= 60) :
                    up = True


                if (up == True) and (differences == 0) and (contact == True):
                    Juggles += 1 
                    points += 1 * (Juggles/10)
                    up = False
                    differences = 0
                    contact = False

            except: 
                pass
            
################
# Check Tricks #
################

            
            try:
                print(distance(center[0],center[1],currrightfoot[0],currrightfoot[1]), radius+20)
                if distance(center[0],center[1],currrightfoot[0],currrightfoot[1]) < radius+20:
                    overlaps += 1
            except:
                pass

            try:
                if distance(center[0],center[1],currleftfoot[0],currleftfoot[1]) < radius+20:
                    overlaps += 1
            except:
                pass
           

            if overlaps >= 8:
                points += 2 * (Juggles/10)
                overlaps = 0
                if 'Air Stall' not in TricksDone:
                    TricksDone += 'Air Stall'
            print(overlaps)
            print(TricksDone)
            

            for i in range(1, len(pts)):
                if pts[i-1] is None or pts[i] is None:
                    continue
                thickness = int(np.sqrt(args["buffer"] / float(i+1)) * 2.5)
                #cv2.line(image, pts[i - 1], pts[i], (0, 0, 255), thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2

            if QuitMode != True:
                cv2.putText(image, f'Juggles: {Juggles}', (50, 100), font, fontScale, color, thickness, cv2.LINE_AA)

                cv2.putText(image, f'Points: {int(points)}', (50, 200), font, fontScale, color, thickness, cv2.LINE_AA)


            cv2.namedWindow('Media', cv2.WINDOW_NORMAL) 
            #cv2.resizeWindow('Media', 800, 450)
            cv2.imshow('Media', image)
            
            try:
                if ( (leftfootcoords.visibility < .3) or (rightfootcoords.visibility < .3) 
                    and center == None and QuitMode != True):
                
                    cv2.putText(image, 'NOT IN FRAME', (600, 550), font, fontScale, color, thickness, cv2.LINE_AA)
                    offscreentime += 1
                    cv2.imshow('Media', image)
            except:
                pass

            if offscreentime > 170:
                QuitMode = True
                cv2.putText(image, 'Nice Touches Out There!', (100, 200), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(image, 'Press q To See Stats and Touch Map', (100, 400), font, fontScale, color, thickness, cv2.LINE_AA) 
            cv2.imshow('Media', image)

            #print(offscreentime)


            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    
    camera.release()
    cv2.destroyAllWindows
    return ball_locs, ball_intercepts, Juggles, points, TricksDone


####################################
# SetUp Before Game:
####################################


#Like in Kinect Sports Set Yourself Up
def setUpBodyPose(poseheld = 0): #Based on external resources
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")

    args = vars(ap.parse_args())

    pts = deque(maxlen=args["buffer"])


    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic


    # For webcam input:
    camera = cv2.VideoCapture(0)
    #camera = cv2.VideoCapture("5_2.mp4")

    #For Video input:

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while camera.isOpened():

            success, image = camera.read()
            if not success:
                break

            #Recolor image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            #Make Detection
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = imutils.resize(image, width=800, height = 450)
            
            try:

                landmarks = results.pose_landmarks.landmark

                leftfootcoords = (landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])

                rightfootcoords = (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])


                L_Wrist_Vals = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

                L_Elbow_Vals = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])

                L_Shoulder_Vals = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

                
                
                L_Wrist_Coords = (L_Wrist_Vals.x, L_Wrist_Vals.y)
                
                L_Elbow_Coords = (L_Elbow_Vals.x, L_Elbow_Vals.y)

                L_Shoulder_Coords = (L_Shoulder_Vals.x, L_Shoulder_Vals.y)

                L_Arm_Angle = calculate_angle(L_Shoulder_Coords,L_Elbow_Coords,L_Wrist_Coords)

                R_Wrist_Vals = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

                R_Elbow_Vals = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

                R_Shoulder_Vals = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

                
                
                R_Wrist_Coords = (R_Wrist_Vals.x, R_Wrist_Vals.y)
                
                R_Elbow_Coords = (R_Elbow_Vals.x, R_Elbow_Vals.y)

                R_Shoulder_Coords = (R_Shoulder_Vals.x, R_Shoulder_Vals.y)

                R_Arm_Angle = calculate_angle(R_Shoulder_Coords,R_Elbow_Coords,R_Wrist_Coords)

                #print('Right:', R_Arm_Angle)

            except:
                pass

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = .5
            color = (0, 0, 255)
            thickness = 2

            try:

                if ( (L_Arm_Angle >= 170 and L_Arm_Angle <= 185) #Holding Arms Straight
                and (R_Arm_Angle >= 170 and R_Arm_Angle <= 185) 
                and (R_Wrist_Coords[1] < R_Shoulder_Coords[1]) #Above Shoulders 
                and (L_Wrist_Coords[1] < L_Shoulder_Coords[1]) 
                and leftfootcoords.visibility > .75 #Both feet are visible
                and rightfootcoords.visibility > .75 ) :

                    poseheld += 1

            
                else:
                    cv2.putText(image, 'PUT BOTH FEET IN FRAME', (0, 50), font, fontScale, color, thickness, cv2.LINE_AA)
                    cv2.putText(image, 'and', (0, 80), font, fontScale, color, thickness, cv2.LINE_AA)
                    cv2.putText(image, 'HOLD ARMS ABOVE SHOULDER AT 180 DEGREES TO BEGIN', (0, 110), font, fontScale, color, thickness, cv2.LINE_AA)  

            except: 
                pass 
            #Draw Joints and Connections
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                mp_drawing.DrawingSpec(color = (0,0,255), thickness=3, circle_radius=2),
                mp_drawing.DrawingSpec(color = (0,255,0), thickness=3, circle_radius=2))
            try:
                p1 = 200,500
                p2 = poseheld * 10 + 200, 550

                p3 = 197,497
                p4 = 603,553
                cv2.rectangle(image, p1, p2, (0,255,0), cv2.FILLED)
                cv2.rectangle(image, p3, p4, (0,0,0), 2)

            except:
                pass

            print(poseheld)


            cv2.namedWindow('Media', cv2.WINDOW_NORMAL) 

            cv2.imshow('Media', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            if poseheld*10+200 >= 600:
                break

    
        
    camera.release()
    cv2.destroyAllWindows
    ballvals, interceptvals, jugglecount, points, TricksDone = BodyPose()

    FeedBack(ballvals,interceptvals,jugglecount,points,TricksDone)   

####################################
# Top Level App:
####################################


def appStarted(app):
    app.mode = 'splashScreenMode'
    url = 'https://yardsoccer.com/wp-content/uploads/soccer-juggling-tips.jpg'
    app.image1 = app.loadImage(url)

    url2 = '''https://see.fontimg.com/api/renderfont4/PDGB/eyJyIjoiZnMiLCJoIjo4MSwidyI6MTI1MCwiZnMiOjY1LCJmZ2MiOiIjMDAwMDAwIiwiYmdjIjoiI0ZGRkZGRiIsInQiOjF9/MTEyIEp1Z2dsZQ/hokjesgeest.png'''

    app.image2a = app.loadImage(url2)
    app.image2 = app.scaleImage(app.image2a,1/3)

    app.splashopen = True

def drawImage(app,canvas,image,cx,cy):
    canvas.create_image(cx,cy, image=ImageTk.PhotoImage(image))
    imageWidth, imageHeight = image.size

    

def splashScreenMode_mousePressed(app,event):
    
    if (event.x >= app.width/2-150) and (event.x <= app.width/2+150) and (event.y >= 280) and (event.y <= 320):
        playsound('/Users/mitch/OneDrive/Documents/Classes/Project Attempt#2/crowdcheer.mp3')
        app.splashopen = False
        runOtherApp(app)
        app.quit()
        time.sleep(2.0)
    
def runOtherApp(app):
    if app.splashopen == False:
        setUpBodyPose()


def splashScreenMode_redrawAll(app, canvas):

    drawImage(app,canvas,app.image1,app.width/2,app.height/2)

    drawImage(app,canvas,app.image2,app.width/2,app.height/3)

    canvas.create_text(app.width/2, (app.height/4)*3, text='PRESS HERE TO START',
                       font=("Purisa", 16, "bold"), activefill="white")

    canvas.create_rectangle(app.width/2-150, (app.height/4)*3 - 20,app.width/2+150, (app.height/4)*3 + 20,)

runApp(width=800,height=400)
