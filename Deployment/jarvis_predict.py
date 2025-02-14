import speech_recognition as sr
import pyttsx3
import pyaudio
import time
import pvporcupine
import struct
import winsound
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

USER = "your highness"
user_commands = []  # Array to store all user speech inputs
commands_df = pd.DataFrame(columns=["Timestamp", "Commands", "Sentiment"])  # Initialize DataFrame with Sentiment column

max_length = 66  # Adjusted to match model's expected sequence length
embedding_size = 75  # Size of the embeddings, matching the model's expected input
vocab_size = 10000  # Adjust to match the vocab size used during training

gru_model = load_model('C://Users//Stevia Putri//Downloads//Jarvis//Jarvis//word2vec_gru_model_3.h5')

# Recreate the tokenizer
example_commands = [
    "hello",
    "goodbye",
    "how are you",
    "open my email",
    "I am so sad today",
    "I am happy"
]

# Fit the tokenizer on the example commands
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(example_commands)

# Placeholder for embedding matrix (replace with actual embeddings if available)
embedding_matrix = np.random.rand(vocab_size, embedding_size)  # Example embedding matrix

def preprocess_text_with_embedding(command, tokenizer, max_length=66):
    sequences = tokenizer.texts_to_sequences([command])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    embedded_sequences = np.array([[embedding_matrix[token] for token in padded_sequences[0]]])
    return embedded_sequences

def predict_sentiment(command):
    preprocessed_command = preprocess_text_with_embedding(command, tokenizer)
    prediction = gru_model.predict(preprocessed_command)
    predicted_class = prediction.argmax(axis=1)  # Assuming a classification model
    return predicted_class[0]  # Return the predicted class

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
        
        # Get sentiment prediction
        predicted_sentiment = predict_sentiment(userSaid)
        sentiments = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
        sentiment = sentiments[predicted_sentiment]  # Map predicted class to sentiment

        # Log the recognized command and sentiment
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        combined_commands = " ".join(user_commands)  # Concatenate all commands up to this point
        commands_df = pd.concat(
            [commands_df, pd.DataFrame({"Timestamp": [timestamp], "Commands": [combined_commands], "Sentiment": [sentiment]})],
            ignore_index=True
        )
        
        # Respond to the user based on recognized commands
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
        
        time.sleep(0.1)  # Small delay for smoother interaction
        print("All user speech inputs:", user_commands)
    
    # Majority voting for final sentiment
    if not commands_df.empty:
        final_sentiment = commands_df['Sentiment'].mode()[0]
        print(f"\nFinal majority sentiment prediction: {final_sentiment}")
        speak(f"The overall sentiment of the conversation is {final_sentiment}.")
    else:
        print("\nNo sentiment predictions made.")
        speak("No sentiment predictions were made during this session.")

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
