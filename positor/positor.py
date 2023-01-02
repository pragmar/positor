from http.client import ACCEPTED
import sys
import os
import re
import json
import datetime
from tabnanny import verbose
import warnings
from itertools import groupby
from typing import Any, List, Tuple

from . import __version__, __whisper_version__
from argparse import ArgumentParser, FileType, Namespace
from pprint import pprint
from colorama import Fore, Back, Style

ACCEPTED_STT_INPUT_EXTENSIONS: Tuple[str] = (".wav", ".mp3", ".mp4", ".m4a")
ACCEPTED_STT_OUTPUT_EXTENSIONS: Tuple[str] = (".txt", ".json", ".vtt", ".srt")
ACCEPTED_STT_WHISPER_MODELS: Tuple[str] = ("tiny", "small", "medium", "large-v2")

# If the output file is *.json, the raw data is written. \
# In the case of image/video/audio, the json data is lzstring'ed into the file metadata.
HELP_DESCRIPTION: str = """
Given a supported audio/video infile, positor writes word-level stt (speech to text) \
data to json or txt. 
""".strip()

def usage() -> str:                                                            
    return """positor -i [infile] options [outfile]\n       positor -i [infile] options [outfile] [outfile2] ..."""

def main():
    quoted_models = ["{0}".format(m) for m in ACCEPTED_STT_WHISPER_MODELS]
    # do this early to get built in --help functionality working
    parser = ArgumentParser(description=HELP_DESCRIPTION, usage=usage())
    parser.add_argument("-i", "--input", help="input audio/video file", type=str)
    parser.add_argument("-w", "--whisper-model", help="supported whisper models (i.e. {0})".format(", ".join(quoted_models)), type=str, default="tiny")
    parser.add_argument("-a", "--absolute", help="use absolute positions (seconds) in json output", action="store_true")
    parser.add_argument("-l", "--lowercase", help="lowercase text in json output", action="store_true")
    parser.add_argument("-v", "--verbose", help="print program information to stdout", action="store_true")
    #parser.add_argument("-f", "--fp16", action="store_true", help="half-precision floating point")
    parser.add_argument("outfile", help="*.json, *.txt, *.vtt, *.srt", nargs="?", type=str)
    parser.add_argument("outfile2", help="optional, additional outfile", nargs="?", type=str)
    parser.add_argument("outfile3", help="optional, additional outfile", nargs="?", type=str)
    args = parser.parse_args()

    # too few arguments, print condensed program description
    if len(sys.argv) < 3:
        print("positor v{0} (whisper/{1})".format(__version__, __whisper_version__))
        print("Speech to data. Infile must contain one (definitive) audio stream.")
        print("usage: {0}".format(usage()))
        print("")
        print(Fore.YELLOW + "Use -h for help." + Style.RESET_ALL)
        sys.exit(0)
        
    args: Namespace = parser.parse_args()
    infile: str = args.input
    whisper_model: str = args.whisper_model
    outfiles: List[str] = [f for f in [args.outfile, args.outfile2, args.outfile3]]
    input_file_ext: str = os.path.splitext(infile)[1].lower()
    if input_file_ext in ACCEPTED_STT_INPUT_EXTENSIONS:
        stt(infile, outfiles, whisper_model, lowercase=args.lowercase, 
            absolute=args.absolute, verbose=args.verbose)
    

def __get_stt_format(absolute) -> str:
    return "stt{0}".format("%" if absolute else "#")

def __get_stt_json(file_name, duration, absolute):
    return {
        "__meta__": {
            "application":"positor/{0}".format(__version__),
            "format": __get_stt_format(absolute),
            "source": {
                "name": os.path.basename(file_name),
                "duration": duration,
            }
        },
        "text": "",
        "positions": [],
    }

def stt(infile: str, outfiles: list[str], whisper_model: str, lowercase=False, absolute=False, verbose=False):

    not_none_outfiles: List[str] = [f for f in outfiles if f is not None]
    filtered_outfiles: List[str] = [f for f in not_none_outfiles if os.path.splitext(f)[1].lower() in ACCEPTED_STT_OUTPUT_EXTENSIONS]
    unusable_outfiles: List[str] = list(set(not_none_outfiles) - set(filtered_outfiles))
    
    if len(filtered_outfiles) == 0 or len(unusable_outfiles) > 0:
        print(Fore.YELLOW + "\nOutfile(s) unspecified or unusable, aborted. Try specifying a file with a supported extension ({0}).\nUnsupported: {1}\n".format(
            ", ".join(ACCEPTED_STT_OUTPUT_EXTENSIONS),
            ", ".join(unusable_outfiles) + Style.RESET_ALL))
        sys.exit(0)

    # defer these imports for a snappier console when not using stt
    from .models import Words, Word, WordBoundaryOverride
    from .whisper_word_level import load_model
    from ffprobe import FFProbe
    
    metadata: FFProbe = FFProbe(infile)
    duration: float = 0
    for stream in metadata.streams:
        if stream.is_audio():
            duration = stream.duration_seconds()
    
    # can't do anything more without an audio channel
    assert(duration != 0)
    # modified model should run just like the regular model but with additional hyperparameters and extra data in results
    assert(whisper_model in ACCEPTED_STT_WHISPER_MODELS)
    
    whisper_model = load_model(whisper_model)  
    
    # whisper results, eat the stdout while in stable-ts regions to quiet, look for better way
    sys.stdout = open(os.devnull, 'w')
    results: dict = whisper_model.transcribe(infile, fp16=False)
    sys.stdout = sys.__stdout__
    
    words = Words()
    words.load_whisper_results(results)

    # for each output file, handle according to .ext
    for outfile in outfiles:

        if outfile is None:
            continue

        text = words.get_all_text()
        if lowercase == True:
            text = text.lower()
        
        outfile_ext = os.path.splitext(outfile)[1].lower()
        if outfile_ext == ".txt":
            open(outfile, "w").write(text)
        elif outfile_ext == ".json":
            stt = __get_stt_json(infile, duration, absolute)
            stt["text"] = text
            for word in words.get_words():
                # 6 is to the millionth, 1/10 second precision for 2 hour file
                # it is also compact. increase if you need higher precision at
                # cost of bloat
                if absolute == True:
                    start = round(word.start, 2)
                    end = round(word.end, 2)
                    stt["positions"].append([start, end])
                else:
                    start = round(word.start/duration, 6)
                    end = round(word.end/duration, 6)
                    stt["positions"].append([start, end])
            open(outfile, "w").write(json.dumps(stt))
        elif outfile_ext == ".vtt":
            # 00:01:14.815 --> 00:01:18.114
            # - What?
            # - Where are we now?
            contents = ["WEBVTT","NOTE webvtt generated by positor/{0}, {1}".format(__version__, __get_stt_format(absolute))]
            grouped_by_line: List[List[Word]] = [list(result) for key, result in groupby(words.get_words(), key=lambda word: word.line_number)]
            for line in grouped_by_line:
                first_word = line[0]
                contents.append("{0} --> {1}\n- {2}".format(
                    Word.seconds_to_timestamp(first_word.line_start), 
                    Word.seconds_to_timestamp(first_word.line_end), 
                    " ".join([w.text for w in line]))
                )
            # trailing white for good measure
            webvtt = "\n\n".join(contents) + "\n"
            open(outfile, "w").write(webvtt)
        elif outfile_ext == ".srt":
            # 1
            # 00:05:00,400 --> 00:05:15,300
            # This is an example of a subtitle.
            contents = ["NOTE srt generated by positor/{0}, {1}".format(__version__, __get_stt_format(absolute))]
            grouped_by_line: List[List[Word]] = [list(result) for key, result in groupby(words.get_words(), key=lambda word: word.line_number)]
            for i, line in enumerate(grouped_by_line):
                first_word = line[0]
                contents.append("{0}\n{1} --> {2}\n{3}".format(
                    i + 1, 
                    Word.seconds_to_timestamp(first_word.line_start).replace(".",","), 
                    Word.seconds_to_timestamp(first_word.line_end).replace(".",","), 
                    " ".join([w.text for w in line]))
                )
            srt = "\n\n".join(contents) + "\n"
            open(outfile, "w").write(srt)
        else:
            raise ValueError("Unsupported output ext. ({0})".format(outfile))
    
    
    # or to get token timestamps that adhere more to the top prediction
    if verbose:
        print(words.get_all_text())
        print("\nPositions:\n")
        for word in words.get_words():
            print("{:>6}. [{:.2f} - {:.2f}] {:<25}".format(word.number, float(word.start), float(word.end), word.text_with_modified_asterisk))
        print("")
