# RobotNLP

## Link to Demo Video
https://vimeo.com/374821366

## Instructions for Setting up

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
### Light Commands
1. Turn your front/back light green
2. Turn off your front/back light
3. Flash the following colors for 2 seconds each: red, blue, green
### Directional Commands
1. Make a (counter-clockwise) circle/donut
2. Run in a (counter-clockwise) square
3. Increase/decrease your speed
4. Run faster/slower
5. Speed up/slow down
6. Sequential directional commands like “go straight, come back and turn left”
### Animation Commands
1. Fall
2. Run away
3. Dance for me / make some moves
4. Sing for me / make some noise
5. Scream
### Head Commands
1. Turn your head to face left
2. Look behind ya
3. Look forward
### State Commands
1. What's the color of your front/back light?
2. What's your name?
3. How should I call you?
4. I wanna call you Jack
5. How much power do you have left?
6. What's your battery status?
### Grid Commands (Must Say "Comma" between 2 Coordinates)
1. You are on a 3 by 3 grid
2. You are at 0, 0
3. There is a chair at 0, 1
4. Go to the right of the chair
5. Go to 0, 0
6. There are some flowers below the chair
7. There is a bomb at 2, 2
### Other Commands
1. Quit / Bye

## Training files merger
Inside the merge folder, you can find merge.py. If you put students' .py files inside the merge folder and call
```
python3 merge.py
```
A new file "r2d2TrainingSentences.txt" will be generated, which contains the merged training data from all students' .py files. The results are sorted according to categories and duplicates are removed.

## Future Directions
1. Right now, all sentences are fed into intent detection to be classified into one of 6 categories. After a category is determined, we have specific parser for that particular category. In the future, we can split the existing 6 cateogories into more categories, which will reduce the amount of pattern matching within each cateogory. But this will also require more accurate classification/intent detection, which may require a better representation of the sentence or a better way to generate a sentence embedding.
2. Find a better representation of the sentence or a better way to generate a sentence embedding. Right now we are splitting the sentence into words, getting an embedding for each word and taking a component-wise average along each dimension. The results are very good. But as we increase the number of categories, this may not be sufficient.
3. Add text-to-speech to give the robot voice
4. Add more pattern matching to existing parsers
5. Add more training sentences
