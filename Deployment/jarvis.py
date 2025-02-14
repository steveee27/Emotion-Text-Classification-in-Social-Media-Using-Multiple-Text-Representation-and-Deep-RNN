import speech_recognition as sr
import pyttsx3
import pyaudio
import time
import pvporcupine
import struct
import winsound
import pandas as pd
import tensorflow as tf

USER = "your highness"
user_commands = []  # Array to store all user speech inputs
commands_df = pd.DataFrame(columns=["Timestamp", "Commands", "Sentiment"])  # Initialize DataFrame with Sentiment column

max_length = 100  # Adjust based on your data

def speak(text):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    print("J.A.R.V.I.S.: " + text + " \n")
    engine.say(text)
    engine.runAndWait()

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        ReadyChirp1()
        r.adjust_for_ambient_noise(source, duration=1)
        r.pause_threshold = 1  # Wait for a short pause to mark the end of a sentence
        print("Listening... ", end="")
        audio = r.listen(source)
        query = ''
        ReadyChirp2()
        try:
            print("Recognizing... ", end="") 
            query = r.recognize_google(audio, language='en-US')
            print(f"User said: {query}")
            user_commands.append(query)  # Append recognized sentence to the list
        except Exception as e:
            print("Exception: " + str(e))
    
    return query.lower()

def ReadyChirp1():
    winsound.Beep(600, 300)

def ReadyChirp2():
    winsound.Beep(500, 300)

def ConversationFlow():
    global commands_df  # Use the global DataFrame
    conversation = True
    while conversation:
        userSaid = takeCommand()
        if "hello" in userSaid:
            speak("hello")
        elif "bye" in userSaid:
            speak("goodbye")
            conversation = False  # Exit the loop when 'bye' is said
        elif "how are you" in userSaid:
            speak("Doing Well")
        elif "stop" in userSaid:
            speak("Stopping Sir")
            conversation = False
        elif "exit" in userSaid:
            speak("Ending program")
            conversation = False
        elif "open my email" in userSaid:
            speak("This is where I would run a program to open your email.")
        
        # After each command, concatenate all previous commands into one string for the current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        combined_commands = " ".join(user_commands)  # Concatenate all commands up to this point

        # Add the new row to the DataFrame with Timestamp, Commands, and Sentiment
        commands_df = pd.concat(
            [commands_df, pd.DataFrame({"Timestamp": [timestamp], "Commands": [combined_commands]})],
            ignore_index=True
        )
        
        time.sleep(0.1)  # Small delay for smoother interaction
        print("All user speech inputs:", user_commands)
    
    # Print the final DataFrame after conversation ends
    print("All commands with sentiment prediction:")
    print(commands_df)  # Show updated DataFrame

def main():
    print(pvporcupine.KEYWORDS)  # Displays the available built-in keywords
    porcupine = None
    pa = None
    audio_stream = None

    print("J.A.R.V.I.S. version 1.2 - Online and Ready!")
    print("**********************************************************")
    print("J.A.R.V.I.S.: Awaiting your call " + USER)

    try:
        access_key = "yMz4py/cL5DopED8JT6gH3HyUd9g3t9j37IbKDuWe7BAp+ywa7yFig=="
        # Use the built-in "jarvis" wake word
        porcupine = pvporcupine.create(access_key=access_key, keywords=["jarvis"])
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wakeword Detected...", end="")
                speak("Ready to Listen")
                ConversationFlow()
                time.sleep(1)
                break
    finally:
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if pa is not None:
            pa.terminate()

        # Print the final DataFrame after the program ends
        print("\nFinal commands with sentiment prediction:")
        print(commands_df)  # Final DataFrame showing all commands and their predicted sentiment

main()