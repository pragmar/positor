import os
from PIL import Image
from typing import List, Tuple
from itertools import groupby
from .models import SttWord

class CaptionPositions:
    """
    formats webvtt and srt captions from stt text and wordlist
    """
    @staticmethod
    def get_webvtt(text, sttwords, duration, positor_version) -> str:
        # 00:01:14.815 --> 00:01:18.114
        # This is an example of a subtitle.
        schema = JsonPositions.get_json_format("stt", False, True)
        contents = ["WEBVTT","NOTE webvtt generated by positor/{0}, {1}".format(
            positor_version, schema)]
        grouped_by_line: List[List[SttWord]] = [list(result) for key, result in 
            groupby(sttwords.get_words(), key=lambda word: word.line_index)]
        for line in grouped_by_line:
            first_word = line[0]
            contents.append("{0} --> {1}\n{2}".format(
                SttWord.seconds_to_timestamp(first_word.line_start), 
                SttWord.seconds_to_timestamp(first_word.line_end), 
                " ".join([w.text for w in line]))
            )
        # trailing white for good measure
        webvtt = "\n\n".join(contents) + "\n"
        return webvtt

    @staticmethod
    def get_srt(text, sttwords, duration, positor_version) -> str:
        # 1
        # 00:05:00,400 --> 00:05:15,300
        # This is an example of a subtitle.
        schema = JsonPositions.get_json_format("stt", False, True)
        contents = ["NOTE srt generated by positor/{0}, {1}".format(
            positor_version, schema)]
        grouped_by_line: List[List[SttWord]] = [list(result) for key, result in 
            groupby(sttwords.get_words(), key=lambda word: word.line_index)]
        for i, line in enumerate(grouped_by_line):
            first_word = line[0]
            contents.append("{0}\n{1} --> {2}\n{3}".format(
                i + 1, 
                SttWord.seconds_to_timestamp(first_word.line_start).replace(".",","), 
                SttWord.seconds_to_timestamp(first_word.line_end).replace(".",","), 
                " ".join([w.text for w in line]))
            )
        srt = "\n\n".join(contents) + "\n"
        return srt

class JsonPositions:
    """
    formats json from stt or ocr text, wordlist, and supporting meta values
    """
    @staticmethod
    def __get_common_json(file_name: str, extractor: str, condensed: bool, absolute: bool, positor_version: str) -> dict:
        """
        Utility reusable, create base object for positor json
        @file_name - filename of file processed by stt (original)
        @duration - duration, in seconds
        @absolute - (or relative positions) boolean
        """
        return {
            "__meta__": {
                "application":"positor/{0}".format(positor_version),
                "schema": JsonPositions.get_json_format(extractor, condensed, absolute),
                "source": {
                    "name": os.path.basename(file_name),
                }
            },
            "text": "",
            "positions": [],
        }
    @staticmethod
    def __get_infile_dimensions(infile: str):
        # get dims, assert non-zero (division comes next)
        img = Image.open(infile)
        input_width = img.width
        input_height = img.height
        img.close()
        assert input_width > 0 and input_height > 0
        return input_width, input_height

    @staticmethod
    def get_json_format(extractor: str, condensed: bool, absolute: bool) -> str:
        """
        Utility reusable, returns json output code.
        @absolute - use absolute positions
        """
        # default is *, or the broader data response
        format_id = "*"
        if condensed == True:
            # condensed options, % 0.XXXX, # XXXX
            format_id = "#" if absolute else "%"
        return "{0}{1}".format(extractor, format_id)

    @staticmethod
    def get_stt_json(text, ocrwords, infile, duration, condensed, absolute, positor_version) -> dict:
        """
        create positor json, stt edition
        @file_name - filename of file processed by stt (original)
        @duration - duration, in seconds
        @absolute - (or relative positions) boolean
        """
        # file_name: str, extractor: str, condensed: bool, absolute: bool, positor_version: str
        stt_json = JsonPositions.__get_common_json(infile, "stt", condensed, absolute,  positor_version)
        stt_json["__meta__"]["source"]["duration"] = duration
        stt_json["text"] = text
        for word in ocrwords.get_words():
            if condensed == False:
                stt_json["positions"].append({
                    "text": word.text,
                    "start": word.start,
                    "end": word.end, 
                    "line_index": word.line_index, 
                })
            elif absolute == True:
                # 2 is to the hundreth of a second, absolutely positioned
                start = round(word.start, 2)
                end = round(word.end, 2)
                stt_json["positions"].append([start, end])
            else:
                # 6 is to the millionth, >1/10 second precision for up to 24 hours of stream 
                # audio, relatively positioned. a percentage start and end on a timeline
                # of 1. the format is compact. increase if you need higher precision at
                # cost of bloat
                start = round(word.start/duration, 6)
                end = round(word.end/duration, 6)
                stt_json["positions"].append([start, end])
        return stt_json

    @staticmethod
    def get_ocr_json(text, ocrwords, infile, condensed, absolute, positor_version) -> dict:
        """
        create positor json, ocr edition
        @file_name - filename of file processed by ocr (original)
        @width/height - dimensions, in pixels
        @absolute - (or relative positions) boolean
        """
        input_width, input_height = JsonPositions.__get_infile_dimensions(infile)
        ocr_json = JsonPositions.__get_common_json(infile, "ocr", condensed, absolute, positor_version)
        ocr_json["__meta__"]["source"]["width"] = input_width
        ocr_json["__meta__"]["source"]["height"] = input_height
        ocr_json["text"] = text
        words = ocrwords.get_words()
        words_count:int = len(words)
        # sanity check, make sure no whitespace in word.text from tesseract
        # len("".split(" ")) == 1, when words is 0, otherwise good
        assert (words_count in (len(text.split(" ")), 0))
        for word in words:
            # not condensed is default request, assume maximal optionality
            # hand back bits and pieces not in condensed. use extensible 
            # dict object to make future updates drama free
            if not condensed:
                ocr_json["positions"].append({
                    "text": word.text,
                    "top": word.top,
                    "right": word.right, 
                    "bottom": word.bottom, 
                    "left": word.left,
                    "line_index": word.line_index,
                    "confidence": word.confidence,
                    # these are tesseract dependent. want to leave option of 
                    # changing engines, this will be eventually an issue if exposed
                    # "_line_number": word.line_number,
                    # "_block_number": word._block_number,
                    # "_paragraph_number": word._paragraph_number,
                })
            elif absolute == True:
                # tradition here is css: clockwise from 12, top, right, bottom, left
                ocr_json["positions"].append([word.top, word.right, word.bottom, word.left])
            else:
                # 4 precision is to the one ten-thousandth (width or height)
                # appropriate precision headroom, can always move to one hundred-thousandth 
                # later. >9999 pixel width images seem fringe
                top = round(word.top/input_height, 4)
                right = round(word.right/input_width, 4)
                bottom = round(word.bottom/input_height, 4)
                left = round(word.left/input_width, 4)
                # coords are clockwise from 12 o'clock (css order)
                ocr_json["positions"].append([top, right, bottom, left])
        return ocr_json

