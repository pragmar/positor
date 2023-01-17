import warnings
import math
import datetime
import numpy as np
from itertools import groupby
from enum import Enum
from typing import Any, List

class WordBoundaryOverride(Enum):
    """
    describes the relationship of an SttWord to its neighbor SttWords,
    and line structure provided by Whisper. line start/end considered
    highest confidence time data. solo is a one word line, i.e. no 
    bounding words. sequencer marks a program intervention to prevent 
    out of order time/word series.
    """
    Undefined = 0
    LineStart = 1
    LineEnd = 2
    Solo = 3
    Sequencer = 4

# -----------------------------------------------------------------------------
# Common
# -----------------------------------------------------------------------------

class WordBase:
    """
    a Word with differing contexts, some audio (SttWord), some not (OcrWord)
    """

    def __init__(self, words, text, line_index=None):
        """
        base word, little is shared
        line_index isn't guaranteed, availability subject to ocr/stt variance
        """
        self._words = words
        self._text = text.strip()
        self._index: int = len(words.get_words())
        self._line_index: int = line_index
    
    @property
    def text(self) -> str:
        return self._text

    @property
    def index(self) -> int:
        return self._index
    
    @property
    def line_index(self) -> int:
        return self._line_index
    
class WordsBase:
    """
    a collection of Words with positions, pulled from stt/ocr. this is the 
    base class. ocr and stt differ in dimensionality, but the container 
    can be consistent.
    """

    def __init__(self):
        """
        add list to be populated later
        line_index isn't guaranteed, availability subject to ocr/stt variance
        """
        self._words: List[Any] = []
        self._lines: List[Any] = []

    def get_words(self) -> List[Any]:
        """ 
        returns the complete list of Word instances.
        """
        return self._words

    def get_lines(self) -> List[Any]:
        """ 
        returns the complete list of Any instances representing lines.
        this may be an SttLine instances or something else.
        """
        return self._lines
    
    def get_all_text(self, lowercase: bool = False) -> str:
        """ 
        get the entire contents, as joined words. returns string.
        @lowercase - convert text to lower?
        """
        #for w in self._words:
        #    print(w)
        result = " ".join([w.text for w in self._words])
        return result if lowercase == False else result.lower()
    
    def get_count(self) -> int:
        """ 
        get total words. Returns int.
        """
        return len(self._words)

    def _add_word(self, word: Any):
        """
        subclasses should roll thier own _add_word.
        @word - could be anything, but in reality it's OcrWord or SttWord
        """
        raise NotImplementedError()

# -----------------------------------------------------------------------------
# OCR
# -----------------------------------------------------------------------------

class OcrWord(WordBase):
    """
    a word with OCR position attached
    """
    def __init__(self, words, text: str, top:int, right:int, bottom: int, left: int,
                 confidence=None, line_number=None, block_number=None, 
                 paragraph_number=None):

        assert top <= bottom and right >= left

        # initialize indexes, normalize the ocr line number to 0-based index
        # tesseract is one-based, and some lines may be removed altogether
        # it's really important to note line_number is from tesseract, and does
        # not describe the internal line_number which is an increment when a
        # new line_number is seen
        if line_number is not None and line_number > 0:
            lines: list = words.get_lines()
            # len(lines) is default, for a new one
            line_index: int = lines.index(line_number) if line_number in lines else len(lines)
            super().__init__(words, text, line_index=line_index)
            # add line_number so it keeps record for above "in" test
            if line_number not in lines:
                lines.append(line_number)
        else:
            # line_index is None, which is fine
            super().__init__(words, text)
        self._words: OcrWords = words
        #self._text: str = text
        self._top: int = top
        self._right: int = right
        self._bottom: int = bottom
        self._left: int = left
        # normalize these "numbers" (one-based) to _index properties
        # this is the best way of maintaining naming consistency across
        # ocr/stt engines. so... everything goes in as 0-based "index"
        # line, and word. then if "numbers" are necessary, they are handled
        # with properties.
        # collect all tesseract but word # (which is internally tracked)
        # and line number, which was normalized to line_index above
        # note block and paragraph numbers are 1-based, and are not 
        # converted to an index. they don't seem important at the moment.
        self._confidence: int = confidence
        self._block_number: int = block_number
        self._paragraph_number: int = paragraph_number
    
    @property
    def text(self) -> str:
        return self._text
    
    @property
    def top(self) -> int:
        return self._top

    @property
    def right(self) -> int:
        return self._right

    @property
    def bottom(self) -> int:
        return self._bottom

    @property
    def left(self) -> int:
        return self._left

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def block_number(self) -> int:
        return self._block_number

    @property
    def paragraph_number(self) -> int:
        return self._paragraph_number

    def __str__(self) -> str:
        return "{0} [{1:0.0f}, {2:0.0f}, {1:0.0f}, {2:0.0f}]".format(
            self.text, self.top, self.right, self.bottom, self.right)


class OcrWords(WordsBase):
    """
    a group of Words (subtitles style), as provided from whisper.
    used to keep timings anchored while reordering timestamps.
    """

    def __init__(self):
        """
        inherits get_words, get_all_text, get_count from WordsBase
        """
        super().__init__()

    def _add_word(self, row: dict):
        """ 
        add a Word instance the list of Word instances.
        """
        # top, etc. come in as floats because of brusque conversion to str or 
        # float in reader. but it's cool, we make it explicit here
        top: int = int(row["top"])
        bottom: int = top + int(row["height"])
        left: int = int(row["left"])
        right: int = left + int(row["height"])
        # getting into ocr engine specific values
        confidence: float = row["conf"]
        line_number: int = row["line_num"]
        block_number: int = row["block_num"]
        paragraph_number: int = row["par_num"]
        # css order, clockwise from 12 o'clock 
        word = OcrWord(self, row["text"], top, right, bottom, left, confidence=confidence, 
            line_number=line_number, block_number=block_number, paragraph_number=paragraph_number)
        self._words.append(word)

    def _get_tesseract_value(self, label: str, value: str):
        """
        cast strings to whatever is called for. refer to tesseract tsv output for headers.
        """
        if label in ("level", "page_num", "block_num", "par_num", "line_num", 
                "word_num", "left", "top", "width", "height"):
            return int(value)
        elif label == "conf":
            return float(value)
        else:
            return value

    def load_tesseract_results(self, results: str):
        """
        load tsv into _words
        @result - input tesseract tsv
        """
        table = results.split("\n")
        labels = None
        for i, row in enumerate(table):
            values = row.split("\t")
            values_count = len(values)
            # make sure we know what we're dealing with
            assert values_count in (1, 12)
            if values_count == 1:
                # end of file, happens for sure. useless row in any case
                pass
            elif i == 0:
                # first row's data are column labels
                labels = values[:]
            else:
                # create dict from header labels, confidence is float
                row_object = { labels[j]: self._get_tesseract_value(labels[j], value) for 
                    j, value in enumerate(values) }
                
                # skip non-texual information
                if row_object["conf"] == -1 or row_object["text"].strip() == "":
                    continue

                # skip noise. tiny, tiny boxes of garbage. it's a problem on 
                # xeroxy-looking images with scan grain.
                # box less than 5px (2x2 pixel box or less) is going to 
                # be illegible and useless 99.9999% of the time
                box_area = (row_object["right"] - row_object["left"]) * \
                    (row_object["bottom"] - row_object["top"]);
                if box_area < 5.0:
                    continue

                # additional skip filters go here
                # otherwise, passes muster, in you go
                self._add_word(row_object)

# -----------------------------------------------------------------------------
# STT
# -----------------------------------------------------------------------------

class SttLine():
    """
    a group of Words (subtitles style), as provided from whisper.
    Used to keep timings anchored while reordering timestamps.
    Index is engine-agnostic line number, zero-based. The number game
    is a quagmire, 0-based, 1-based, it'll never be consistent across
    different 3rd party libs, so normalize to _index/0,1,2,... alter at 
    output stage, if need be.
    """
    def __init__(self, words: WordsBase, start: float, end: float):
        self._words: float = words
        self._start: float = start
        self._end: float = end
        self._index: int = len(words.get_lines())
    
    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def index(self) -> int:
        return self._index

class SttWord(WordBase):
    """
    a word, likely one of many. parent/container is SttWords which represents
    an extracted audio stream from audio/video.
    """

    def __init__(self, words, text: str, timestamps: List[float], line: SttLine, 
            override: WordBoundaryOverride):

        super().__init__(words, text, line_index=line.index)
        
        # Whisper wants to work in phrases/lines of text, has timings available 
        # take line start/end seriously, since it's the most intentional value returned
        # by whisper. words can (and do!) exceed line bounds.
        self._line: SttLine = line

        # the Word instances which represent all words/positions
        self._words: SttWords = words

        # reject_outliers will remove any timestamps outside of 2 std deviations
        # this reduces outrageous situations where the numbers don't make sense.
        self._np_timestamps: List[float] = SttWord.reject_outliers(np.array(timestamps))

        # start/end to be determined when all words loaded in, and looped into 
        # final shape, tbd TODO looped into shape bit
        if override == WordBoundaryOverride.LineStart:
            # first word, use the start line_boundary
            word_start = word_end = line.start
        elif override == WordBoundaryOverride.LineEnd:
            # last word, use the end line_boundary
            word_start = word_end = line.end
        elif override == WordBoundaryOverride.Solo:
            # only word, defer to line boundary
            word_start = line.start
            word_end = line.end
        elif override == WordBoundaryOverride.Undefined:
            # 98% situation, a word sandwiched bewtween two others
            word_start = word_end = np.median(self._np_timestamps)
        
        self._word_start: float = word_start
        self._word_end: float = word_end
        self._word_boundary_override: WordBoundaryOverride = override
        self._word_boundary_overidden: bool = False if override == WordBoundaryOverride.Undefined else True
    
    def __str__(self) -> str:
        return "{0} [{1:4.2f} - {2:4.2f}]".format(self.text, self.start, self.end)
    
    def next(self):
        """
        get the next Word in the series. Always the word._index
        of + 1, within Words._words. If no next exists, None
        is returned.
        """
        _words = self._words.get_words()
        return _words[self._index + 1] if (
            self._index + 1) < self._words.get_count() else None
    
    def previous(self):
        """
        get the previous Word in the series. Always the word._index
        of - 1, within Words._words. If no previous exists, None
        is returned.
        """
        _words = self._words.get_words()
        return _words[self._index - 1] if self._index > 0 else None
    
    def extend(self, text: str, timestamps: List[float], override: WordBoundaryOverride):
        """
        extend the Word object, adding additional text (generally 
        punctuation).
        """
        self._text = "{0}{1}".format(self._text, text)
        if override == WordBoundaryOverride.LineEnd:
            self.update_boundary(self.line_end, self.line_end, override)
    
    def update_boundary(self, start:float, end:float, override: WordBoundaryOverride):
        """
        update word boundaries, leaving a log of the operation.
        """
        self._word_boundary_override = override
        self._word_boundary_overidden = True
        self._word_start = start
        self._word_end = end

    @property
    def line_start(self) -> float:
        return self._line.start

    @property
    def line_end(self) -> float:
        return self._line.end

    @property
    def line_index(self) -> int:
        return self._line.index
    
    @property
    def line_contained(self) -> bool:
        """
        returns boolean, describing whether Word fits within the line container
        """
        return self._line.start <= self.start and self._line.end >= self.end
    
    @property
    def neighbor_contained(self) -> bool:
        """
        returns boolean, describing whether Word fits within bordering Words.
        """
        next: SttWord = self.next()
        previous: SttWord = self.previous()        
        if next is None and previous is None:
            return True
        elif next is None:
            return self.start >= previous.start 
        elif previous is None:
            return self.start <= next.start
        else:
            return self.start >= previous.start and self.start <= next.start
    
    @property
    def min(self) -> float: 
        return np.min(self._np_timestamps)

    @property
    def max(self) -> float:
        return np.max(self._np_timestamps)
    
    @property
    def median(self) -> float:
        return np.median(self._np_timestamps)
    
    @property
    def boundary_override(self) -> WordBoundaryOverride:
        return self._word_boundary_override
      
    @property
    def text_with_modified_asterisk(self) -> str:
        asterisk_note = "* [{0}]".format(self._word_boundary_override.name) if \
            self._word_boundary_overidden else ""
        return "{0}{1}".format(self.text, asterisk_note)

    @property
    def stdev(self) -> float:
        return np.std(self._np_timestamps)
    
    @property
    def number(self) -> float:
        return self._index
    
    @property
    def start(self) -> float:
        return self._word_start
    
    @property
    def end(self) -> float:
        return self._word_end

    # https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    @staticmethod
    def reject_outliers(data, m=1):
        return data[abs(data - np.mean(data)) < m * np.std(data)]
    
    @staticmethod
    def get_override(partial_loop_index, partials_count) -> WordBoundaryOverride:
        if partial_loop_index == 0 and partials_count == 1:
            override = WordBoundaryOverride.Solo
        elif partial_loop_index == 0:
            override = WordBoundaryOverride.LineStart
        elif partial_loop_index == partials_count - 1:
            override = WordBoundaryOverride.LineEnd
        else:
            override = WordBoundaryOverride.Undefined
        return override

    @staticmethod
    def seconds_to_timestamp(seconds) -> str:
        """
        returns a vtt style timestamp given seconds. webvtt desires second 
        precision to .000. e.g. 00:01:14.815 --> 00:01:18.114
        """

        # this won't work with a file exceeding 24 hours duration, seems reasonable.
        assert(seconds >= 0 and datetime.timedelta(seconds=seconds) < datetime.timedelta(hours=24))

        delta: datetime.timedelta = datetime.timedelta(seconds=seconds)
        y2k: datetime.datetime = datetime.datetime(2000,1,1)
        timestamp_raw = (y2k + delta).strftime("%H:%M:%S.%f")
        # python precision is millionth (.000000) '00:00:34.000000'
        # truncate for webvtt to thousandth (.000)
        stamp, millis = timestamp_raw.split(".")
        timestamp = "{0}.{1}".format(stamp, millis[:3])
        return timestamp
        
class SttWords(WordsBase):
    """
    constainer for words and temporal metadata extracted from audio
    """

    def __init__(self):
        """
        inherits get_words, get_all_text, get_count from WordsBase
        """
        super().__init__()

    def _add_word(self, word: Any):
        """ 
        add a Word instance the list of Word instances.
        """
        self._words.append(word)
    
    def load_whisper_results(self, results: dict):
        """ 
        does the work of converting whisper results to positor json.
        """
        segments = results['segments'] if isinstance(results['segments'], list) else []
        if not segments:
            warnings.warn('No segments found. No results.')
        
        # use current word to store multi-segment words, e.g. the below should combine to "weeks,"
        # {'end': 13.079999923706055, 'start': 12.479999542236328, 'text': ' weeks'}
        # {'end': 13.119999885559082, 'start': 13.079999923706055, 'text': ','}   
        current_word = None
        
        for i, line_segment in enumerate(segments):

            # set the outer boundaries of this line_segment, used to constrain words within
            line_segment_word_partials = line_segment["unstable_word_timestamps"]
            line_segment_word_partials_count = len(line_segment_word_partials)
            line = SttLine(self, line_segment["start"], line_segment["end"])
            self._lines.append(line)

            # loop over word partials, some only contain punctuation
            for j, line_segment_word_partial in enumerate(line_segment_word_partials):
                text = line_segment_word_partial["word"]
                timestamps = line_segment_word_partial["timestamps"]
                # there needs to be something, throw now, continue/adapt later if it happens
                assert len(timestamps) >= 1
                # get override from static get_override
                override: WordBoundaryOverride = SttWord.get_override(j, line_segment_word_partials_count)
                # if word starts with " " (space), it's a new word.
                # i once saw some weird .NET behavior in whisper (" " || ".")
                if text[0] == " " or (text[0] == "." and len(text) > 1):
                    current_word = SttWord(self, text, timestamps, line, override)
                    self._add_word(current_word)
                # continuation of current word, likely punctuation, unknown
                elif current_word is not None:
                    # extend is destructive. it combines the text
                    # it adds trailing punctuation to the preceding word
                    current_word.extend(text, timestamps, override)
            
        self._sequence()
    
    def _splice_times(self, out_of_order_group: List[SttWord]):
        """
        smear out_of_order_group timestamps into time range 
        defined by the bordering Words. All Words must have bordering
        next/previous or assertion will fail.
        """
        # gotta have at least an element to process
        out_of_order_group_count = len(out_of_order_group)
        if out_of_order_group_count == 0:
            # nothing to do
            return

        # first and last may be the same word, and in theory
        # should be start stamped sequentially. if not, more
        # defensive posture required
        group_first_word = out_of_order_group[0]
        group_last_word = out_of_order_group[-1]

        # grab the Words that will contain the group, timestamp-wise
        previous: SttWord =  group_first_word.previous()
        next: SttWord =  group_last_word.next()
        assert previous is not None and next is not None

        start_boundary: float = max(previous.start, previous.end)
        end_boundary: float = max(next.start, next.end)
        boundary_width: float = math.fabs(end_boundary - start_boundary)
        incremental_width: float = boundary_width/(out_of_order_group_count + 1)
        cursor = start_boundary
        for word in out_of_order_group:
            cursor += incremental_width
            word.update_boundary(cursor, cursor, WordBoundaryOverride.Sequencer)
        
    def _sequence(self):
        """
        reorders word timestamps to make sure the are sequential (>= last word)
        """

        # get all words, then separate into lists by line_index
        words: List[SttWord] = self.get_words()
        grouped_by_line: List[List[SttWord]] = [list(result) for key, result in groupby(words, key=lambda word: word.line_index)]
        
        # cursor tracks known success as word.start timestamps
        cursor = None
        
        # cycle words, run some reordering on the timestamps using line bounds as anchors
        # timestamps are unpredictable, not necessarilly sequential, this reorders the start/end
        # so at least word1.start >= word1.end >= word2.start >= word2.start is true, and so forth.
        # that, and being within line start/end is the only thing you can depend on. some data gets 
        # pretty knarly (>5 second distributions for one word, and such). reel it back on into the 
        # real world. you must first understad what WordBoundaryOverride is, to understand the flow.
        for line_of_words in grouped_by_line:
            accumulated_rejects: List[SttWord] = []
            for word in line_of_words:
                # set default as line start
                cursor = cursor if cursor is not None else word.line_start
                if word.boundary_override in (WordBoundaryOverride.Sequencer,):
                    raise RuntimeError("Resequencing is not supported.")
                elif word.boundary_override in (WordBoundaryOverride.LineStart, WordBoundaryOverride.Solo):
                    # advance cursor if the word out front
                    if word.line_contained and word.neighbor_contained and word.start >= cursor:
                        cursor = word.start
                    continue
                elif word.boundary_override == WordBoundaryOverride.LineEnd:
                    # make certain we are last in line, as expected
                    assert (word.index == line_of_words[-1].index)
                    break
                elif word.boundary_override == WordBoundaryOverride.Undefined:
                    next: SttWord = word.next()
                    previous: SttWord = word.previous()
                    # undefined boundary signals sandwiched status, verify this is the case
                    assert None not in (next, previous)
                    if word.line_contained and word.neighbor_contained and word.start >= cursor:
                        # success, word is where it is expected chronologically
                        # if there are accumulated rejects, they are processed
                        # and cursor advanced
                        self._splice_times(accumulated_rejects)
                        accumulated_rejects = []
                        cursor = word.start
                        continue
                    else:
                        # word does not order within tolerances, handle with care
                        word_is_sequential_reject = len(accumulated_rejects) == 0 or word.index == accumulated_rejects[-1].index + 1
                        if word_is_sequential_reject:
                            accumulated_rejects.append(word)
                        else:
                            # abort accumulation, dump rejects and reset the accumulation
                            self._splice_times(accumulated_rejects)
                            accumulated_rejects = [ word ]
                
            # reorder any accumulated rejects
            self._splice_times(accumulated_rejects)

