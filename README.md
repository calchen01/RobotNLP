# RobotNLP

## Link to Demo Video
https://vimeo.com/374821366

## Instructions to Set up

Put all .py files inside the src folder. Put "data" and "vectors" folders inside your project folder, parallel to the src folder. Put your .magnitude file inside the "vectors" folder.

robot_com.py supports command line IO to control your robot using natural English language. With the addition of audio_io.py, you are able to control your robot using voice!

To run command line IO, you need to install matplotlib:
```
pip3 install matplotlib
```

Next, change the robot ID with your own in line 15 of robot_com.py. Start the server in one Terminal session, cd into the src folder in another Terminal session and type:
```
python3 robot_com.py
```

To run audio IO, you will need to install portaudio and pyaudio:
```
brew install portaudio
pip3 install pyaudio
```

Next, you need an account with Google Cloud Platform (GCP). When you register a new account, you'll get $300 of free credits. You need to enable the speech-to-text module, set up a new project and service account in your GCP account and get a service account key file (this is going to be in .json format). Rename it "credentials.json" and put it under the src folder. You may also need to install and set up Google Cloud SDK locally. Look up GCP's documentation for more details.

Next, change the robot ID with your own in line 166 of audio_io.py. Start the server in one Terminal session, cd into the src folder in another Terminal session and type:
```
python3 audio_io.py
```

Notes:
1. If you want to try audio IO, please try command line IO first.

2. If you are able to successfully run audio_io.py, say your command (using voice!) and see if text appears in Terminal. To end the session, simply say any sentence containing one of the following keywords: "exit", "quit", "bye" or "goodbye".

## Start Server & Run Project

Open Terminal and type:
```
cd .../sphero-project/spherov2.js/examples
sudo yarn server
```

Keep it running. Open another Terminal session and type:
```
cd .../sphero-project/src
```

For command line IO, type:
```
python3 robot_com.py
```

Or for voice IO, type:
```
python3 audio_io.py
```

## Example Commands
1. Make a circle
2. Run in a counter clockwise square
3. Turn you front/back/rear light <color>
4. What color is your front/back/rear light?
5. What’s your name?
6. Directional commands like “go straight, come back and turn left”
7. Dance/sing for me
8. Make some noise/moves
9. You are on a a by b grid
10. You are at a, b
11. There is … at a, b
12. Go to a, b
13. Fall
14. What’s your battery status?
15. Increase/decrease your speed
16. Scream
17. Turn your head to face left/right
18. Look behind you
19. Quit

## Training files merger
Inside the merge folder, you can find merge.py. If you put students' .py files inside the merge folder and call
```
python3 merge.py
```
A new file "r2d2TrainingSentences.txt" will be generated, which contains the merged training data from all students' .py files. The results are sorted according to categories and duplicates are removed.