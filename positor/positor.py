import datetime
import sys
import os
import json
import re
import subprocess
import io
import tempfile
import uuid
from enum import Enum
from itertools import groupby
from typing import List, Tuple
from argparse import ArgumentParser, FileType, Namespace
from colorama import Fore, Style
from positor import __version__, __whisper_version__, __tesseract_version__
from positor.positions import JsonPositions

class LibContext(Enum):
    """
    describes the relationship of a Word to its neighbor Words,
    and line structure provided by Whisper. line start/end considered
    highest confidence time data. solo is a one word line, i.e. no 
    bounding words. sequencer marks a program intervention to prevent 
    out of order time/word series.
    """
    Module = 0
    Packaged = 1

try:
    import tkinter
    positor_env = LibContext.Module
except ImportError:
    # tkinter is explicitly excluded from packed version,
    # if it's missing, we're definitely in exe-land. for packaged
    # versions of positor (msi), need to turn off torch jit
    # or there will be errors. extent of performance downgrade? 
    # currently unknown. torch JIT UserErrors emitted only when 
    # packaged as exe, fallback (jit off) works fine
    # note: os.environ["PYTORCH_JIT"] = "0" doesn't work
    import warnings
    warnings.filterwarnings(action="ignore", category=UserWarning)
    positor_env = LibContext.Packaged

ACCEPTED_OCR_INPUT_EXTENSIONS: Tuple[str] = (".bmp", ".png", ".jpg", ".jpeg", ".gif")
ACCEPTED_OCR_OUTPUT_EXTENSIONS: Tuple[str] = (".txt", ".tsv", ".json")
ACCEPTED_STT_INPUT_EXTENSIONS: Tuple[str] = (".wav", ".mp3", ".mp4", ".m4a")
ACCEPTED_STT_OUTPUT_EXTENSIONS: Tuple[str] = (".txt", ".json", ".vtt", ".srt")
# TODO .en varieties not supported yet, errors
ACCEPTED_STT_WHISPER_MODELS: Tuple[str] = ("tiny", "small", "medium", "large-v2")

# If the output file is *.json, the raw data is written. \
# In the case of image/video/audio, the json data is lzstring'ed into the file metadata.
HELP_DESCRIPTION: str = """
Positor extracts word-level data from STT (speech to text) given an audio source, 
or word-level OCR (optical character recognition) data given an image. 
The infile type, audio or image, governs available outputs. Visit https://pragmar.com/positor/ 
for details.
"""

def usage() -> str:                                                            
    return "positor -i [infile] options [outfile]\n" + \
    "       positor -i [infile] options [outfile] [outfile2] ..."

def main():
    """
    The console command function, driven by sys.argv.
    """
    
    quoted_models = ["{0}".format(m) for m in ACCEPTED_STT_WHISPER_MODELS]
    # do this early to get built in --help functionality working
    parser = ArgumentParser(description=HELP_DESCRIPTION, usage=usage())
    parser.add_argument("-i", "--infile", help="audio, video, or image file", type=str)
    parser.add_argument("-w", "--whisper-model", help="supported whisper models (i.e. {0}), stt-only".format(
        ", ".join(quoted_models)), type=str, default="tiny")
    parser.add_argument("-l", "--tesslang", help="tesseract language code, ocr-only", type=str, default="eng")
    parser.add_argument("-d", "--tessdata", help="folder containing tesseract language packs, ocr-only", type=str)
    parser.add_argument("-a", "--absolute", help="use absolute positions (seconds/pixels) in json output",
        action="store_true")
    parser.add_argument("-c", "--lowercase", help="lowercase text in json output", action="store_true")
    parser.add_argument("-p", "--verbose", help="print program information to stdout", action="store_true")
    parser.add_argument("-v", "--version", help="print version information", action="store_true")
    #parser.add_argument("-f", "--fp16", action="store_true", help="half-precision floating point")
    parser.add_argument("outfile", help="*.txt, *.json, *.vtt (stt), *.srt (stt), *.tsv (ocr)", nargs="?", type=str)
    parser.add_argument("outfile2", help="optional, additional outfile", nargs="?", type=str)
    parser.add_argument("outfile3", help="optional, additional outfile", nargs="?", type=str)
    
    args: Namespace = parser.parse_args()

    if args.version == True:
        print(__version__)
        sys.exit(0)

    # this is the positor (zero argument) screen, but is actually more general
    # too few arguments, print condensed program description to stderr
    if len(sys.argv) < 3:
        add_tesseract = "; tesseract/{0}".format(__tesseract_version__) if \
            positor_env == LibContext.Packaged else ""
        sys.stderr.write(
            "\n".join([
                "positor v{0} (whisper/{1}{2})".format(__version__, __whisper_version__, add_tesseract),
                "Speech to data. Infile must contain one (definitive) audio stream.",
                "usage: {0}".format(usage()),
                "",
                "{0}Use -h for help.{1}".format(Fore.YELLOW, Style.RESET_ALL)
            ])
        )
        sys.exit(0)
    
    # things appear on the up and up, proceed.
    infile: str = args.infile
    whisper_model: str = args.whisper_model
    outfiles: List[str] = [f for f in [args.outfile, args.outfile2, args.outfile3]]
    input_file_ext: str = os.path.splitext(infile)[1].lower()
    if input_file_ext in ACCEPTED_STT_INPUT_EXTENSIONS:
        stt(infile, outfiles, whisper_model, lowercase=args.lowercase, 
            absolute=args.absolute, verbose=args.verbose)
    elif input_file_ext in ACCEPTED_OCR_INPUT_EXTENSIONS:
        ocr(infile, outfiles, whisper_model, lowercase=args.lowercase, absolute=args.absolute, 
            tessdata=args.tessdata, tesslang=args.tesslang, verbose=args.verbose)

def __error_and_exit(message: str):
    """
    Take argv outfiles and make sure they are supported. Return filtered list,
    exit if things look bleak.
    """
    print("\n" + Fore.YELLOW + message + Style.RESET_ALL + "\n")
    sys.exit(0)

def __filter_outfiles(outfiles: List[str], acceptable_ext: List[str]):
    """
    Take argv outfiles and make sure they are supported. Return filtered list,
    exit if things look bleak.
    """
    not_none_outfiles: List[str] = [f for f in outfiles if f is not None]
    filtered_outfiles: List[str] = [f for f in not_none_outfiles if \
        os.path.splitext(f)[1].lower() in acceptable_ext]
    unusable_outfiles: List[str] = list(set(not_none_outfiles) - set(filtered_outfiles))
    if len(filtered_outfiles) == 0 or len(unusable_outfiles) > 0:
        __error_and_exit("Outfile(s) unspecified or unusable, aborted. " +
           "Try specifying a file with a supported extension ({0}).\nUnsupported: {1}\n".format(
                ", ".join(acceptable_ext), ", ".join(unusable_outfiles))
        )
        sys.exit(0)
    return filtered_outfiles

def ocr(infile: str, outfiles: list[str], whisper_model: str, lowercase=False, absolute=False, 
        tessdata=None, tesslang=None, verbose=False):
    """
    Handle OCR request.
    @infile - an image. a png, perhaps
    @outfiles - requested ocr output variants
    @whisper_model - the blob model to use, e.g. tiny
    @lowercase - lowercase all text, useful to simplify search
    @absolute - use absolute units of measurement, as opposed to %
    @verbose - show some process info, for debugging, subject to change
    """

    # first things first, look for tesseract, and bail if not found
    tesseract_command = ["tesseract", "-v"]
    tesseract_process = subprocess.Popen( tesseract_command, shell=True, stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)
    stdout, stderr = tesseract_process.communicate()
    if stdout.decode("utf-8") == "":
        # not happening
        __error_and_exit("OCR support is disabled because Tesseract OCR is not installed. To enable, " +
        "either install Tesseract, or install positor via an installer available at https://pragmar.com/positor")
    
    # will exit, prior to loading modules (fast), if anything is off
    filtered_outfiles: List[str] = __filter_outfiles(outfiles, ACCEPTED_OCR_OUTPUT_EXTENSIONS)

    # deferred to keep non-stt/ocr generating console commands zippy
    from .models import OcrWords, OcrWord
    from PIL import Image

    # TODO the language/lang code dance here, echo back to user about tessdata=...
    # consider mapping eng to en etc. even though stt side doesn't use it?
    # override and language pack downloads at https://github.com/tesseract-ocr/tessdata
    
    tempdir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory(prefix="positor_")
    tmpfile_stub = os.path.join(tempdir.name, uuid.uuid4().hex)
    tmpfile = "{0}.tsv".format(tmpfile_stub)

    tesslang = tesslang if tesslang is not None else "eng"
    tesseract_command = ["tesseract", infile, tmpfile_stub, "--oem", "1", "-l", tesslang, "tsv"]
    if tessdata is not None:
        if os.path.isdir(tessdata):
            tessdata_args = ["--tessdata-dir", tessdata]
            tesseract_command = tesseract_command[:-1] + tessdata_args + tesseract_command[-1:]
        else:
            __error_and_exit("tessdata must be a valid directory")
    tesseract_process = subprocess.Popen( tesseract_command, shell=True, stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)
    stdout, stderr = tesseract_process.communicate()
    
    # ignore minor details, e.g. libpng warning: iCCP: known incorrect sRGB profile
    # tessdata is a obvious big one, add others as necessary
    if stderr and "TESSDATA_PREFIX" in stderr.decode("utf-8"):
        __error_and_exit(stderr.decode("utf-8").strip())
    
    # guess we're okay, grab the output that was just generated
    tsv = None
    with io.open(tmpfile, "r", encoding="utf-8") as input_file:
        tsv = input_file.read()
    tempdir.cleanup()
    if tsv in ("", None):
        raise RuntimeError("Unreadable tesseract response. ({0})".format(input_file))
    
    # take tsv result and feed it into words class
    words = OcrWords()
    words.load_tesseract_results(tsv)

    # for each output file, handle according to .ext
    for outfile in filtered_outfiles:
        text = words.get_all_text(lowercase=lowercase)
        outfile_ext = os.path.splitext(outfile)[1].lower()
        if outfile_ext == ".txt":
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(text)
        elif outfile_ext == ".tsv":
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(tsv)
        elif outfile_ext == ".json":
            # get dims, assert non-zero (we're dividing)
            img = Image.open(infile)
            input_width = img.width
            input_height = img.height
            img.close()
            # image data is required
            assert input_width > 0 and input_height > 0
            ocr_json = JsonPositions.get_ocr_json(infile, input_width, input_height, absolute, __version__)
            ocr_json["text"] = text
            words = words.get_words()
            # sanity check, make sure no whitespace in word.text from tesseract
            assert len(words) == len(text.split(" "))
            for word in words:
                if absolute == True:
                    # tradition here is css: clockwise from 12, top, right, bottom, left
                    ocr_json["positions"].append([word.top, word.right, word.bottom, word.left])
                else:
                    # 4 precision is to the one ten-thousandth (width or height)
                    # keeps filesize down, with appropriate precision headroom
                    # try it, can always move to one hundred-thousandth later.
                    top = round(word.top/input_height, 4)
                    right = round(word.right/input_width, 4)
                    bottom = round(word.bottom/input_height, 4)
                    left = round(word.left/input_width, 4)
                    # coords are clockwise from 12 o'clock (css order)
                    ocr_json["positions"].append([top, right, bottom, left])
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(json.dumps(ocr_json))

def stt(infile: str, outfiles: List[str], whisper_model: str, lowercase=False, 
        absolute=False, verbose=False):
    
    # will exit, prior to loading modules (fast), if anything is off
    filtered_outfiles: List[str] = __filter_outfiles(outfiles, ACCEPTED_STT_OUTPUT_EXTENSIONS)
    
    # defer these imports for a snappier console when not using stt
    from .models import SttWords, SttWord, WordBoundaryOverride
    from .stt_word_level import load_model
    import ffmpeg

    # ffprobe binary is 70+ megs uncompressed, and 20 in the msi package
    # opting for roundabout (worse?) ffmpeg duration check, because it's 
    # worth the bloat reduction. this way, don't have to ship both 
    # ffprobe and ffmpeg in packed versions. this is the only time 
    # ffprobe is/was necessary far as i know.
    # probe = ffmpeg.probe(infile)
    # assert len(probe["streams"]) > 0
    # duration_seconds_meta: str = probe["streams"][0]["duration"]
    # duration:float = float(duration_seconds_meta)
    
    duration_re = re.compile("Duration:\s*(\d\d:\d\d:\d\d\.\d+)")
    ffmpeg_process = subprocess.Popen( ["ffmpeg", "-i", infile], shell=True, stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)
    stdout, stderr = ffmpeg_process.communicate()
    duration_result: List[str] = duration_re.findall(stderr.decode("utf-8"))
    
    # absolutely require a duration hit
    assert len(duration_result) == 1
    duration_string: str = duration_result[0]
    
    # we're sure this is the pattern: '00:01:21.68'
    duration_raw_split: List[str] = re.split("[:\.]", duration_string)
    hours, minutes, seconds, microseconds = [int(n) for n in duration_raw_split]
    duration_delta:datetime.timedelta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds, 
        microseconds=microseconds)
    duration:float = duration_delta.total_seconds()
    
    # can't do anything more without it, i gotta have some duration.
    assert duration != 0

    # modified model should run just like the regular model but with additional 
    # hyperparameters and extra data in results there are problems with the .en 
    # model integrations, but not going to deal with it rn.
    assert whisper_model in ACCEPTED_STT_WHISPER_MODELS
    
    whisper_model = load_model(whisper_model)
    # whisper results, eat the stdout while in stable-ts regions to quiet
    sys.stdout = open(os.devnull, 'w')
    results: dict = whisper_model.transcribe(infile, fp16=False)
    sys.stdout = sys.__stdout__
    
    words = SttWords()
    words.load_whisper_results(results)

    # for each output file, handle according to .ext
    for outfile in filtered_outfiles:

        text = words.get_all_text(lowercase=lowercase)
        # if lowercase == True:
        #     text = text.lower()
        
        outfile_ext = os.path.splitext(outfile)[1].lower()
        
        if outfile_ext == ".txt":
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(json.dumps(text))
        elif outfile_ext == ".json":
            stt_json = JsonPositions.get_stt_json(infile, duration, absolute, __version__)
            stt_json["text"] = text
            for word in words.get_words():
                if absolute == True:
                    # 2 is to the hundreth of a second, absolutely positioned
                    start = round(word.start, 2)
                    end = round(word.end, 2)
                    stt_json["positions"].append([start, end])
                else:
                    # 6 is to the millionth, >1/100 second precision for any reasonable
                    # file, and relatively positioned. a percentage start and end on a timeline
                    # of 1. the format is compact. increase if you need higher precision at
                    # cost of bloat
                    start = round(word.start/duration, 6)
                    end = round(word.end/duration, 6)
                    stt_json["positions"].append([start, end])
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(json.dumps(stt_json))
        elif outfile_ext == ".vtt":
            # 00:01:14.815 --> 00:01:18.114
            # - What?
            # - Where are we now?
            contents = ["WEBVTT","NOTE webvtt generated by positor/{0}, {1}".format(
                __version__, JsonPositions.get_stt_format(absolute))]
            grouped_by_line: List[List[SttWord]] = [list(result) for key, result in 
                groupby(words.get_words(), key=lambda word: word.line_number)]
            for line in grouped_by_line:
                first_word = line[0]
                contents.append("{0} --> {1}\n- {2}".format(
                    SttWord.seconds_to_timestamp(first_word.line_start), 
                    SttWord.seconds_to_timestamp(first_word.line_end), 
                    " ".join([w.text for w in line]))
                )
            # trailing white for good measure
            webvtt = "\n\n".join(contents) + "\n"
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(webvtt)
        elif outfile_ext == ".srt":
            # 1
            # 00:05:00,400 --> 00:05:15,300
            # This is an example of a subtitle.
            contents = ["NOTE srt generated by positor/{0}, {1}".format(
                __version__, JsonPositions.get_stt_format(absolute))]
            grouped_by_line: List[List[SttWord]] = [list(result) for key, result in 
                groupby(words.get_words(), key=lambda word: word.line_number)]
            for i, line in enumerate(grouped_by_line):
                first_word = line[0]
                contents.append("{0}\n{1} --> {2}\n{3}".format(
                    i + 1, 
                    SttWord.seconds_to_timestamp(first_word.line_start).replace(".",","), 
                    SttWord.seconds_to_timestamp(first_word.line_end).replace(".",","), 
                    " ".join([w.text for w in line]))
                )
            srt = "\n\n".join(contents) + "\n"
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(srt)
        else:
            raise ValueError("Unsupported output ext. ({0})".format(outfile))

    # or to get token timestamps that adhere more to the top prediction
    if verbose:
        print(words.get_all_text())
        print("\nPositions:\n")
        for word in words.get_words():
            print("{:>6}. [{:.2f} - {:.2f}] {:<25}".format(word.number, float(word.start), 
                float(word.end), word.text_with_modified_asterisk))
        print("")
