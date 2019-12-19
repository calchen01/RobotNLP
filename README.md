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
### Light Commands
1. Turn your front/back light / lights color
2. Turn off your front/back light / lights
3. Dim your front/back light / lights
4. Increase the intensity of your front/back light / lights
5. Flash the following colors for 2 seconds each: red, blue, green
### Directional Commands
1. Make a (counter-clockwise) circle/donut
2. Run in a (counter-clockwise) square
3. Increase/decrease your speed / run faster/slower
4. Sequential directional commands like “go straight, come back and turn left”
### Animation Commands
1. Fall over
2. Run away
3. Dance for me / make some moves
4. Sing for me / make some noise
5. Scream
### Head Commands
1. Turn your head to face left/right/forward/back
2. Look in front of/behind you / look forward
### State Commands
1. What's the color(s) of your front/back light / lights?
2. What's your name?
3. I wanna/want to call you name
4. How much power do you have left?
5. What's your battery status?
### Grid Commands (Must Say "Comma" between 2 Coordinates)
1. You are on a a by b grid
2. You are at a, b
3. There is/are obj at a, b
4. Go to a, b
5. There is/are obj above/below/to/on the left/right of obj
6. Go to the top/bottom/left/right of obj
### Other Commands
1. Quit / Bye

## Training files merger
Inside the merge folder, you can find merge.py. If you put students' .py files inside the merge folder and call
```
python3 merge.py
```
A new file "r2d2TrainingSentences.txt" will be generated, which contains the merged training data from all students' .py files. The results are sorted according to categories and duplicates are removed.

## Future Directions
1. Add text-to-speech to make the IO more interactive
2. Add more pattern matching to the different parsers
3. Add more training sentences
4. Find a better way to represent sentences. Right now we are splitting the sentence into words and take a component-wise average along each dimension with pretty good results.
5. Right now, all sentences are fed into intent detection into one of 6 categories and after a category is determined, we have specific parser for that particular category. This is a trade-off: in the future, we can split the existing 6 cateogories into more categories and this will reduce the amount of pattern matching within each cateogory, but this will also require that our intent detection to be able to correctly classify an embedding, which may require 4) a better representation of the sentences.
