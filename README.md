# RobotNLP

## Link to Demo Video
https://vimeo.com/374821366

## Instructions to Set up

Put robot_com.py and audio_io.py under the src folder. robot_com.py supports command line IO to control your robot using natural English language. With the addition of audio_io.py, you are able to control your robot using voice!

To run command line IO, you need to install matplotlib:
```
pip3 install matplotlib
```

???
You also need to uncomment code in robot_com.py, change the robot serial ID with your own, start the server in one Terminal, cd into the src folder in another Terminal and type:
???
```
python3 robot_com.py
```

To run audio IO, you will need to install portaudio and pyaudio:
```
brew install portaudio
pip3 install pyaudio
```

Next, you need an account with Google Cloud Platform (GCP). When you register a new account, you'll get $300 of free credits. You need to enable the speech-to-text module, set up a new project and service account in your GCP account and get a service account key file (this is going to be in .json format). Rename it "credentials.json" and put it under the src folder. You may also need to install and set up Google Cloud SDK locally. Look up GCP's documentation for more details.

Change the robot ID with your own in audio_io.py in line 166.

To run voice IO, you need to start the server in one Terminal, cd into the src folder in another Terminal and type:
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