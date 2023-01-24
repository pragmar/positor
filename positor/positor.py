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
    describes the runtime situation. module is a python
    pip install situation. packaged is the installer binaries
    running as exe, dmg, etc..
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

ACCEPTED_OCR_INPUT_EXTENSIONS: Tuple[str] = (".bmp", ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff")
ACCEPTED_OCR_OUTPUT_EXTENSIONS: Tuple[str] = (".txt", ".csv", ".tsv", ".json", ".webp")
ACCEPTED_STT_INPUT_EXTENSIONS: Tuple[str] = (".wav", ".mp3", ".mp4", ".m4a")
ACCEPTED_STT_OUTPUT_EXTENSIONS: Tuple[str] = (".txt", ".csv", ".json", ".vtt", ".srt", ".webp")
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
    parser.add_argument("-l", "--tesseract-language", help="tesseract language code, ocr-only", type=str, default="eng")
    parser.add_argument("-d", "--tesseract-directory", help="folder containing tesseract language packs, ocr-only", type=str)
    parser.add_argument("-g", "--json-lowercase", help="lowercase text in json output", action="store_true")
    parser.add_argument("-c", "--json-condensed", help="condensed data-structure json output, for client-side", action="store_true")
    parser.add_argument("-a", "--json-condensed-absolute", help="condensed, but with absolute positions", action="store_true")
    parser.add_argument("-v", "--version", help="print version information", action="store_true")
    parser.add_argument("-x", "--verbose", help="print sometimes helpful information to stdout", action="store_true")
    #parser.add_argument("-f", "--fp16", action="store_true", help="half-precision floating point")
    parser.add_argument("outfile", help="*.txt, *.csv, *.json, *.webp, *.vtt (stt), *.srt (stt), *.tsv (ocr)", nargs="?", type=str)
    parser.add_argument("outfile2", help="optional, additional outfile", nargs="?", type=str)
    parser.add_argument("outfile3", help="optional, additional outfile", nargs="?", type=str)
    parser.add_argument("outfile4", help="optional, additional outfile", nargs="?", type=str)
    args: Namespace = parser.parse_args()

    if args.version == True:
        print(__version__)
        sys.exit(0)

    # this is the positor (zero argument) screen, but is more generally
    # too few arguments, print condensed program description to stderr
    if len(sys.argv) < 3:
        add_tesseract = "; tesseract/{0}".format(__tesseract_version__) if \
            positor_env == LibContext.Packaged else ""
        sys.stderr.write(
            "\n".join([
                "positor v{0} (whisper/{1}{2})".format(__version__, __whisper_version__, add_tesseract),
                "STT/OCR extractor.",
                "usage: {0}".format(usage()),
                "",
                "{0}Use -h for help.{1}".format(Fore.YELLOW, Style.RESET_ALL)
            ])
        )
        sys.exit(0)
    
    # things appear on the up and up, proceed.
    infile: str = args.infile
    if infile is not None and not os.path.exists(infile):
        __error_and_exit("Infile does not exist. ({0})".format(infile))
    
    whisper_model: str = args.whisper_model
    outfiles: List[str] = [f for f in [args.outfile, args.outfile2, args.outfile3, args.outfile4]]
    input_file_ext: str = os.path.splitext(infile)[1].lower()
    if input_file_ext not in ACCEPTED_STT_INPUT_EXTENSIONS + ACCEPTED_OCR_INPUT_EXTENSIONS:
        __error_and_exit("Infile unsupported. \nSTT support: {0}.\nOCR support: {1}".format(
            ", ".join(ACCEPTED_STT_INPUT_EXTENSIONS), ", ".join(ACCEPTED_OCR_INPUT_EXTENSIONS))
        )
    elif input_file_ext in ACCEPTED_STT_INPUT_EXTENSIONS:
        stt(infile, outfiles, whisper_model, condensed=args.json_condensed, lowercase=args.json_lowercase, 
            absolute_condensed=args.json_condensed_absolute, verbose=args.verbose)
    elif input_file_ext in ACCEPTED_OCR_INPUT_EXTENSIONS:
        ocr(infile, outfiles, whisper_model, condensed=args.json_condensed, lowercase=args.json_lowercase, 
           absolute_condensed=args.json_condensed_absolute, tessdata=args.tesseract_directory, 
           language=args.tesseract_language, verbose=args.verbose)

def __error_and_exit(message: str):
    """
    Take argv outfiles and make sure they are supported. Return filtered list,
    exit if things look bleak.
    """
    sys.stderr.write("\n" + Fore.YELLOW + "Error. " + message + Style.RESET_ALL + "\n")
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
    return filtered_outfiles

def ocr(infile: str, outfiles: list[str], whisper_model: str, condensed=False, lowercase=False,
         absolute_condensed=False, tessdata=None, language=None, verbose=False):
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
    
    # track how long the process takes
    timer_start = datetime.datetime.utcnow()

    # will exit, prior to loading modules (fast), if anything is off
    filtered_outfiles: List[str] = __filter_outfiles(outfiles, ACCEPTED_OCR_OUTPUT_EXTENSIONS)

    # deferred to keep non-stt/ocr generating console commands zippy
    from .models import OcrWords
    from .images import MetaImageSource
    from .positions import JsonPositions
    
    # sort of odd use of command line, one bool greater than the next
    is_condensed: bool = condensed or absolute_condensed
    is_absolute: bool = absolute_condensed
    
    # consider mapping eng to en etc. even though stt side doesn't use it?
    # override and language pack downloads at https://github.com/tesseract-ocr/tessdata    
    tempdir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory(prefix="positor_")
    tmpfile_stub = os.path.join(tempdir.name, uuid.uuid4().hex)
    tmpfile = "{0}.tsv".format(tmpfile_stub)

    # ensure defaulted as necessary, also note 3-letter code is non-standard
    # these can also be extended to multiple, e.g. eng+spa. more complicated than it looks
    language = language if language is not None else "eng"
    # --oem 1, neural net over legacy ocr engine
    # -c thresholding_method=2 is Sauvola binarization over Otsu (legacy)
    # https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html
    tesseract_command = ["tesseract", infile, tmpfile_stub, "--oem", "1", "-l", language, "tsv"]
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
    
    # grab the output that was just generated
    tsv = None
    with io.open(tmpfile, "r", encoding="utf-8") as input_file:
        tsv = input_file.read()
    tempdir.cleanup()
    if tsv in ("", None):
        raise RuntimeError("Unreadable tesseract response. ({0})".format(input_file))
    
    # take tsv result and feed it into words class
    ocrwords = OcrWords()
    ocrwords.load_tesseract_results(tsv)

    # for each output file, handle according to extension (.ext)
    for outfile in filtered_outfiles:
        text = ocrwords.get_all_text(lowercase=lowercase)
        outfile_ext = os.path.splitext(outfile)[1].lower()
        if outfile_ext == ".txt":
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(text)
        elif outfile_ext == ".webp":
            ocr_json = JsonPositions.get_ocr_json(text, ocrwords, infile, is_condensed, is_absolute, __version__)
            MetaImageSource.export_webp(infile, outfile, json.dumps(ocr_json))
        elif outfile_ext == ".tsv":
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(tsv)
        elif outfile_ext == ".json":
            ocr_json = JsonPositions.get_ocr_json(text, ocrwords, infile, is_condensed, is_absolute, __version__)
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(json.dumps(ocr_json))
        elif outfile_ext == ".csv":
            # get dims, assert non-zero (we're dividing)
            input_width, input_height = JsonPositions.__get_infile_dimensions(infile)
            lines = ["text,top,right,bottom,left,#image_width:{0},#image_height:{1}".format(input_width, input_height)]
            for word in ocrwords.get_words():
                word_text: str = word.text
                # can't have rogue commas in csv, wrap in quotes, csv-escape existing quotes
                if "," in word_text:
                    word_text = '"{0}"'.format(word_text.replace('"','""'))
                lines.append("{0},{1},{2},{3},{4}".format(word_text, word.top, word.right, word.bottom, word.right))
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write("\n".join(lines))
        else:
            raise ValueError("Unsupported output ext. ({0})".format(outfile))

    # or to get token timestamps that adhere more to the top prediction
    if verbose:
        delta = datetime.datetime.utcnow() - timer_start
        print("Processing Time: {0}".format(delta))
        print("Text:")
        print(ocrwords.get_all_text())
        print("")

def stt(infile: str, outfiles: List[str], whisper_model: str, condensed=False, 
        lowercase=False, absolute_condensed=False, verbose=False):
    
    # will exit, prior to loading modules (fast), if anything is off
    filtered_outfiles: List[str] = __filter_outfiles(outfiles, ACCEPTED_STT_OUTPUT_EXTENSIONS)
    
    # defer these imports for a snappier console when not using stt
    from .models import SttWords, SttWord
    from .stt_word_level import load_model
    from .positions import JsonPositions, CaptionPositions
    from .images import MetaImageWaveform

    # track how long the process takes
    timer_start = datetime.datetime.utcnow()

    # these command options stack on one another
    is_condensed: bool = condensed or absolute_condensed
    is_absolute: bool = absolute_condensed

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
    
    # can't do anything more without it, gotta have some duration.
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
    
    sttwords = SttWords()
    sttwords.load_whisper_results(results)
    text = sttwords.get_all_text(lowercase=lowercase)
    # for each output file, handle according to .ext
    for outfile in filtered_outfiles:
        outfile_ext = os.path.splitext(outfile)[1].lower()
        if outfile_ext == ".txt":
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(text)
        elif outfile_ext == ".webp":
            stt_json = JsonPositions.get_stt_json(text, sttwords, infile, duration, is_condensed, is_absolute, __version__)
            MetaImageWaveform.export_webp(infile, outfile, json.dumps(stt_json))
        elif outfile_ext == ".json":
            stt_json = JsonPositions.get_stt_json(text, sttwords, infile, duration, is_condensed, is_absolute, __version__)
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(json.dumps(stt_json))
        elif outfile_ext == ".vtt":
            webvtt = CaptionPositions.get_webvtt(text, sttwords, duration, __version__)
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(webvtt)
        elif outfile_ext == ".srt":
            srt = CaptionPositions.get_srt(text, sttwords, duration, __version__)
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write(srt)
        elif outfile_ext == ".csv":
            lines = ["text,line_index,start,end,#audio_duration:{0}".format(duration)]
            for word in sttwords.get_words():
                word_text: str = word.text
                if "," in word_text:
                    word_text = '"{0}"'.format(word_text.replace('"','""'))
                lines.append("{0},{1},{2},{3}".format(word_text, word.line_index, word.start, word.end))
            with io.open(outfile,"w", encoding="utf-8") as out:
                out.write("\n".join(lines))
        else:
            raise ValueError("Unsupported output ext. ({0})".format(outfile))

    # or to get token timestamps that adhere more to the top prediction
    if verbose:
        delta = datetime.datetime.utcnow() - timer_start
        print("Processing Time: {0}".format(delta))
        print("Text:")
        print(sttwords.get_all_text())
        print("\nPositions:\n")
        for word in sttwords.get_words():
            print("{:>6}. [{:.2f} - {:.2f}] {:<25}".format(word.index, float(word.start), 
                float(word.end), word.text_with_modified_asterisk))
        print("")
