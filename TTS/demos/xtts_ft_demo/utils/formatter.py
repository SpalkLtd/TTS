import os
import gc
import torchaudio
import pandas
from faster_whisper import WhisperModel
from glob import glob

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import torch
import torchaudio
# torch.set_num_threads(1)

from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners

torch.set_num_threads(16)


audio_types = (".wav", ".mp3", ".flac")


def list_audios(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=audio_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an audio and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the audio and yield it
                audioPath = os.path.join(rootDir, filename)
                yield audioPath


def format_audio_list(audio_files, target_language="en", out_path=None, buffer=0.2, eval_percentage=0.15, speaker_name="coqui", gradio_progress=None):
    audio_total_size = 0
    # make sure that ooutput file exists
    os.makedirs(out_path, exist_ok=True)

    # Loading Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Whisper Model!")
    asr_model = WhisperModel("large-v2", device=device, compute_type="float16")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    for audio_path in tqdm_object:
        wav, sr = torchaudio.load(audio_path)
        # stereo to mono if needed
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(
            audio_path, word_timestamps=True, language=target_language)
        segments = list(segments)
        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        # added all segments words in a unique list
        words_list = []
        for _, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        # process each word
        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start = word.start
                # If it is the first sentence, add buffer or get the begining of the file
                if word_idx == 0:
                    # Add buffer to the sentence start
                    sentence_start = max(sentence_start - buffer, 0)
                else:
                    # get previous sentence end
                    previous_word_end = words_list[word_idx - 1].end
                    # add buffer or get the silence midle between the previous sentence and the current one
                    sentence_start = max(
                        sentence_start - buffer, (previous_word_end + sentence_start)/2)

                sentence = word.word
                first_word = False
            else:
                sentence += word.word

            if word.word[-1] in ["!", ".", "?"]:
                sentence = sentence[1:]
                # Expand number and abbreviations plus normalization
                sentence = multilingual_cleaners(sentence, target_language)
                audio_file_name, _ = os.path.splitext(
                    os.path.basename(audio_path))

                audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

                # Check for the next word's existence
                if word_idx + 1 < len(words_list):
                    next_word_start = words_list[word_idx + 1].start
                else:
                    # If don't have more words it means that it is the last sentence then use the audio len as next word start
                    next_word_start = (wav.shape[0] - 1) / sr

                # Average the current word end and next word start
                word_end = min((word.end + next_word_start) /
                               2, word.end + buffer)

                absoulte_path = os.path.join(out_path, audio_file)
                os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)
                i += 1
                first_word = True

                audio = wav[int(sr*sentence_start):int(sr*word_end)].unsqueeze(0)
                # if the audio is too short ignore it (i.e < 0.33 seconds)
                if audio.size(-1) >= sr/3:
                    torchaudio.save(absoulte_path,
                                    audio,
                                    sr
                                    )
                else:
                    continue

                metadata["audio_file"].append(audio_file)
                metadata["text"].append(sentence)
                metadata["speaker_name"].append(speaker_name)

    df = pandas.DataFrame(metadata)
    df = df.sample(frac=1)
    num_val_samples = int(len(df)*eval_percentage)

    df_eval = df[:num_val_samples]
    df_train = df[num_val_samples:]

    df_train = df_train.sort_values('audio_file')
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    df_train.to_csv(train_metadata_path, sep="|", index=False)

    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_eval = df_eval.sort_values('audio_file')
    df_eval.to_csv(eval_metadata_path, sep="|", index=False)

    # deallocate VRAM and RAM
    del asr_model, df_train, df_eval, df, metadata
    gc.collect()

    return train_metadata_path, eval_metadata_path, audio_total_size


def process_audio_file(audio_path, target_language, buffer, out_path, speaker_name, asr_model):
    # Load the audio file
    wav, sr = torchaudio.load(audio_path)
    if wav.size(0) != 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    wav = wav.squeeze()

    # Transcribe audio file to get segments
    segments, _ = asr_model.transcribe(
        audio_path, word_timestamps=True, language=target_language)

    metadata = {"audio_file": [], "text": [], "speaker_name": []}
    i = 0

    # Initialize variables for sentence formation
    sentence = ""
    sentence_start = None
    first_word = True

    # Iterate through each segment and each word to form sentences
    for segment in segments:
        for word in segment.words:
            # Set the start time for the first word of the sentence
            if first_word:
                sentence_start = max(word.start - buffer, 0) if sentence_start is None else max(
                    word.start - buffer, (sentence_end + word.start) / 2)
                sentence = word.word
                first_word = False
            else:
                sentence += " " + word.word

            # Check if the word ends with a punctuation mark indicating the end of a sentence
            if word.word[-1] in ["!", ".", "?"]:
                # Adjust the current sentence
                sentence = sentence.strip()  # Assuming multilingual_cleaners if needed
                audio_file_name, _ = os.path.splitext(
                    os.path.basename(audio_path))
                audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

                # Calculate the end time for the sentence
                sentence_end = word.end + buffer
                absoulte_path = os.path.join(out_path, audio_file)
                os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)

                # Save the audio segment for the sentence
                audio_segment = wav[int(
                    sr * sentence_start):int(sr * sentence_end)].unsqueeze(0)
                # Check if the segment is at least 0.33 seconds long
                if audio_segment.size(-1) >= sr * 0.33:
                    torchaudio.save(absoulte_path, audio_segment, sr)
                    metadata["audio_file"].append(audio_file)
                    metadata["text"].append(sentence)
                    metadata["speaker_name"].append(speaker_name)

                # Reset for the next sentence
                i += 1
                first_word = True
                sentence = ""
                sentence_start = None

    return metadata


def format_audio_list_concurrent(audio_files, target_language="en", out_path=None, buffer=0.2, eval_percentage=0.15, speaker_name="coqui", gradio_progress=None):
    os.makedirs(out_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Whisper Model!")
    asr_model = WhisperModel("large-v2", device=device, compute_type="float16")
    all_metadata = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_audio = {executor.submit(process_audio_file, audio_path, target_language,
                                           buffer, out_path, speaker_name, asr_model): audio_path for audio_path in audio_files}
        if gradio_progress is not None:
            tqdm_object = gradio_progress.tqdm(as_completed(
                future_to_audio), total=len(audio_files), desc="Processing...")
        else:
            tqdm_object = tqdm(as_completed(future_to_audio),
                               total=len(audio_files))

        for future in as_completed(future_to_audio):
            all_metadata.append(future.result())

    # Initialize master metadata dictionary
    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    # Aggregate results from all futures
    for item in all_metadata:
        metadata["audio_file"].extend(item["audio_file"])
        metadata["text"].extend(item["text"])
        metadata["speaker_name"].extend(item["speaker_name"])

    # The rest of your code to handle metadata and cleanup remains the same.
    df = pandas.DataFrame(metadata)
    df = df.sample(frac=1)
    num_val_samples = int(len(df)*eval_percentage)

    df_eval = df[:num_val_samples]
    df_train = df[num_val_samples:]

    df_train = df_train.sort_values('audio_file')
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    df_train.to_csv(train_metadata_path, sep="|", index=False)

    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_eval = df_eval.sort_values('audio_file')
    df_eval.to_csv(eval_metadata_path, sep="|", index=False)

    # deallocate VRAM and RAM
    del asr_model, df_train, df_eval, df, metadata
    gc.collect()

    return train_metadata_path, eval_metadata_path
