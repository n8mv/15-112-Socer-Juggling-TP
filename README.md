# 15-112-Socer-Juggling-TP
Nathan Vastey's Spring 2022 Programming Term Project
Demo Video Link: https://youtu.be/OYT_r9ixUbM

112 Juggling:

112 Juggling is an application that uses OpenCV functions to track the motions of juggling a soccer ball, count how many juggles are completed, and recognize certain tricks.

To run the project, simply make sure your camera is on and run JugglingApp.py. The game has 4 different windows that the user will go through. 


The splash screen for starting the game which just includes the button to start the game. 
Then a screen for calibration of the camera where the user has to keep their hands above their head at a 180-degree angle and keep their feet in the frame. 
This is simply so the user can orientate their camera properly before the game starts to avoid any major glitches. 

Next is the actual juggling portion where points and total juggles are tracked. 
When the user is done they can move off-screen for a little while to indicate completion of their session then come back and press 'q' to move on to the final window. 

Finally, there is a screen that displays all the observed ball locations during the session and the points of contact for a juggle along with the total 
juggles, score, and skills completed are displayed at the top (there is only one possible so far).


The necessary modules to run my term project are as follows:

from tkinter.tix  IMAGETEXT
cmu_112_graphics  *
cv2
mediapipe as mp
time
os
collections  deque
from imutils.video  VideoStream
NumPy as np
argparse
imutils
math
playsound 
copy
