import tensorflow as tf
import numpy as np
import cv2
from mido import MidiFile, MidiTrack, Message
from boxes_detection import detect_notes_on_single_image

FREQ = {
    "C1": 32, "C#1": 34, "Db1": 34, "D1": 36, "D#1": 38, "Eb1": 38, "E1": 41, "F1": 43, "F#1": 46, "Gb1": 46, "G1": 49,
    "G#1": 52, "Ab1": 52, "A1": 55, "A#1": 58, "Bb1": 58, "B1": 61, "C2": 65, "C#2": 69, "Db2": 69, "D2": 73, "D#2": 77,
    "Eb2": 77, "E2": 82, "F2": 87, "F#2": 92, "Gb2": 92, "G2": 98, "G#2": 104, "Ab2": 104, "A2": 110, "A#2": 116, "Bb2": 116,
    "B2": 123, "C3": 130, "C#3": 138, "Db3": 138, "D3": 146, "D#3": 155, "Eb3": 155, "E3": 164, "F3": 174, "F#3": 185, "Gb3": 185,
    "G3": 196, "G#3": 208, "Ab3": 208, "A3": 220, "A#3": 233, "Bb3": 233, "B3": 246, "C4": 261, "C#4": 277, "Db4": 277, "D4": 293,
    "D#4": 311, "Eb4": 311, "E4": 329, "F4": 349, "F#4": 369, "Gb4": 369, "G4": 392, "G#4": 415, "Ab4": 415, "A4": 440, "A#4": 466,
    "Bb4": 466, "B4": 493, "C5": 523, "C#5": 554, "Db5": 554, "D5": 587, "D#5": 622, "E5": 659, "Eb5": 659, "F5": 698, "F#5": 739,
    "Gb5": 739, "G5": 784, "G#5": 830, "Ab5": 830, "A5": 880, "A#5": 932, "Bb5": 932, "B5": 987,
}

def crop_notes(image, boxes):
    height, width, _ = image.shape
    cropped_notes = []

    for box in boxes:
        ymin, xmin, ymax, xmax = box
        x1, y1, x2, y2 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)
        cropped_note = image[y1:y2, x1:x2]
        cropped_notes.append(cropped_note)

    return cropped_notes

def classify_and_convert_to_midi(cropped_notes, classification_model, output_midi_path):
    note_labels = []

    for note in cropped_notes:
        resized_note = cv2.resize(note, (128, 128)) / 255.0
        resized_note = np.expand_dims(resized_note, axis=-1)
        resized_note = np.expand_dims(resized_note, axis=0)

        prediction = classification_model.predict(resized_note)
        note_label = list(FREQ.keys())[np.argmax(prediction)]
        note_labels.append(note_label)

    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    for note_label in note_labels:
        frequency = FREQ[note_label]

        if frequency == 0.0067:
            track.append(Message('note_off', time=480))
        else:
            midi_note = int(69 + 12 * np.log2(frequency / 440))
            track.append(Message('note_on', note=midi_note, velocity=64, time=0))
            track.append(Message('note_off', note=midi_note, velocity=64, time=480))

    midi.save(output_midi_path)