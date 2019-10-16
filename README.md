# RobotNLP

## Natural Language Commands (Command Line / Voice IO)

Put robot_com.py and audio_io.py under the src folder. robot_com.py supports command line IO to control your robot using natural English language. With the addition of audio_io.py, you are able to control your robot using voice!

To run command line IO, you need to install matplotlib:

```
pip3 install matplotlib
```

You also need to uncomment code in robot_com.py, change the robot serial ID with your own, start the server in one Terminal, cd into the src folder in another Terminal and type:

```
python3 robot_com.py
```

To run audio IO, you will need to install portaudio and pyaudio:

```
brew install portaudio
pip3 install pyaudio
```

Next, you need an account with Google Cloud Platform (GCP). When you register a new account, you'll get $300 of free credits. You need to enable speech-to-text module, set up a new project and service account in your GCP account and get a service account key file (this is going to be in .json format). Rename it credentials.json and put it under the src folder. You may also need to install and set up Google Cloud SDK locally. Look up GCP's documentation for more details.

As before, change the robot serial ID with your own in audio_io.py. Make sure you recomment the code in robot_com.py if you decide to use voice IO.

To run voice IO, you need to start the server in one Terminal, cd into the src folder in another Terminal and type:

```
python3 audio_io.py
```

Notes:
1. If you want to try audio IO, please try command line IO first.

2. If you are able to successfully run audio_io.py, say your command (using voice!) and see if text appears in the Terminal. To end the session, simply say any sentence containing one of the following keywords: "exit", "quit", "bye" or "goodbye".

3. Only a few commands are supported at the moment, including single move (e.g. "go straight"), sequential moves (e.g. "go straight and turn left"), dancing, making sound, changing color for one or both light(s), turning off one or both light(s).
