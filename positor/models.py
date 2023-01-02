import warnings
import math
import datetime
import numpy as np
from itertools import groupby
from enum import Enum
from typing import Any, List

class WordBoundaryOverride(Enum):
    """
    Describes the relationship of a Word to it's neighbors,
    and the line structure provided by Whisper.
    """
    Undefined = 0
    LineStart = 1
    LineEnd = 2
    Solo = 3
    Sequencer = 4



class Line():
    """
    A group of Words (subtitles style), as provided from whisper.
    Used to keep timings anchored while reordering timestamps.
    """
    def __init__(self, start: float, end: float, number: int):
        self.__start: float = start
        self.__end: float = end
        self.__number: int = number
    
    @property
    def start(self) -> float:
        return self.__start

    @property
    def end(self) -> float:
        return self.__end

    @property
    def number(self) -> int:
        return self.__number

class Word:

    def __init__(self, text: str, timestamps: List[float], words, line: Line, override: WordBoundaryOverride):
        
        # text is the word text, it can be modified by extend()
        self.text = text.strip()
        
        # number updated from Words container
        # start/end to be determined when all words loaded in, and looped into final shape
        self.__word_number: int = words.get_count()

        # Whisper wants to work in phrases/lines of text, has timings available 
        # take line start/end seriously, since it's the most intentional value returned
        # by whisper. words can (and do!) exceed line bounds.
        self.__line: Line = line

        # the Word instances which represent all words/positions
        self.__words: Words = words

        # reject_outliers will remove any timestamps outside of 2 std deviations
        # this reduces outrageous situations where the numbers don't make sense.
        self.__np_timestamps: List[float] = Word.reject_outliers(np.array(timestamps))

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
            word_start = word_end = np.median(self.__np_timestamps)
        
        self.__word_start: float = word_start
        self.__word_end: float = word_end
        self.__word_boundary_override: WordBoundaryOverride = override
        self.__word_boundary_overidden: bool = False if override == WordBoundaryOverride.Undefined else True
    
    def __str__(self) -> str:
        return "{0} [{1:4.2f} - {2:4.2f}]".format(self.text, self.start, self.end)
    
    def next(self):
        """
        Get the next Word in the series. Always the word.number
        of + 1, within Words.__words. If no next exists, None
        is returned.
        """
        _words = self.__words.get_words()
        return _words[self.__word_number + 1] if (
            self.__word_number + 1) < self.__words.get_count() else None
    
    def previous(self):
        """
        Get the previous Word in the series. Always the word.number
        of - 1, within Words.__words. If no previous exists, None
        is returned.
        """
        _words = self.__words.get_words()
        return _words[self.__word_number - 1] if self.__word_number > 0 else None
    
    def extend(self, text: str, timestamps: List[float], override: WordBoundaryOverride):
        """
        Extend the Word object, adding additional text (generally 
        punctuation).
        """
        self.text = "{0}{1}".format(self.text, text)
        if override == WordBoundaryOverride.LineEnd:
            self.update_boundary(self.line_end, self.line_end, override)
    
    def update_boundary(self, start:float, end:float, override: WordBoundaryOverride):
        """
        Update word boundaries, leaving a log of the operation.
        """
        self.__word_boundary_override = override
        self.__word_boundary_overidden = True
        self.__word_start = start
        self.__word_end = end

    @property
    def line_start(self) -> float:
        return self.__line.start

    @property
    def line_end(self) -> float:
        return self.__line.end
    
    @property
    def line_number(self) -> int:
        return self.__line.number
    
    @property
    def line_contained(self) -> bool:
        """
        Returns boolean, describing whether Word fits within the line container
        """
        return self.__line.start <= self.start and self.__line.end >= self.end
    
    @property
    def neighbor_contained(self) -> bool:
        """
        Returns boolean, describing whether Word fits within bordering Words.
        """
        next: Word = self.next()
        previous: Word = self.previous()        
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
        return np.min(self.__np_timestamps)

    @property
    def max(self) -> float:
        return np.max(self.__np_timestamps)
    
    @property
    def median(self) -> float:
        return np.median(self.__np_timestamps)
    
    @property
    def boundary_override(self) -> WordBoundaryOverride:
        return self.__word_boundary_override
      
    @property
    def text_with_modified_asterisk(self) -> str:
        return "{0}{1}".format(self.text, "* [{0}]".format(self.__word_boundary_override.name) if self.__word_boundary_overidden else "")

    @property
    def stdev(self) -> float:
        return np.std(self.__np_timestamps)
    
    @property
    def number(self) -> float:
        return self.__word_number
    
    @property
    def start(self) -> float:
        return self.__word_start
    
    @property
    def end(self) -> float:
        return self.__word_end

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
        Returns a vtt style timestamp given seconds. Webvtt desires second 
        precision to .000. e.g. 00:01:14.815 --> 00:01:18.114
        """

        # this won't work with a file exceeding 24 hours duration, seems reasonable.
        assert(seconds >= 0 and datetime.timedelta(seconds=seconds) < datetime.timedelta(hours=24))

        delta: datetime.timedelta = datetime.timedelta(seconds=seconds)
        twok: datetime.datetime = datetime.datetime(2000,1,1)
        timestamp_raw = (twok + delta).strftime("%H:%M:%S.%f")
        # python precision is millionth (.000000) '00:00:34.000000'
        # truncate for webvtt to thousandth (.000)
        stamp, millis = timestamp_raw.split(".")
        timestamp = "{0}.{1}".format(stamp, millis[:3])
        return timestamp
        
class Words:

    def __init__(self):
        self.__words: List[Word] = []

    def get_words(self) -> List[Word]:
        """ 
        Returns the complete list of Word instances.
        """
        return self.__words
    
    def get_all_text(self) -> str:
        """ 
        Get the entire contents, as joined words. Returns string.
        """
        return " ".join([w.text for w in self.__words])
    
    def get_count(self):
        """ 
        Get total words. Returns int.
        """
        return len(self.__words)

    def add_word(self, word: Any):
        """ 
        Add a Word instance the list of Word instances.
        """
        self.__words.append(word)
    
    def load_whisper_results(self, results: dict):
        """ 
        Does the work of converting whisper results to positor json.
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
            line = Line(line_segment["start"], line_segment["end"], i)

            # loop over word partials, some only contain punctuation
            for j, line_segment_word_partial in enumerate(line_segment_word_partials):
                text = line_segment_word_partial["word"]
                timestamps = line_segment_word_partial["timestamps"]
                # there needs to be something, throw now, continue/adapt later if it happens
                assert len(timestamps) >= 1
                # get override from static get_override
                override: WordBoundaryOverride = Word.get_override(j, line_segment_word_partials_count)
                # if word starts with " " (space), it's a new word.
                # i once saw some weird .NET behavior in whisper (" " || ".")
                if text[0] == " " or (text[0] == "." and len(text) > 1):
                    current_word = Word(text, timestamps, self, line, override)
                    self.add_word(current_word)
                # continuation of current word, likely punctuation, unknown
                elif current_word is not None:
                    # extend is destructive. it combines the text
                    # it adds trailing punctuation to the preceding word
                    current_word.extend(text, timestamps, override)
                    # print("{0} {1}".format(current_word, text))
            
        self.__sequence()
    
    def __splice_times(self, out_of_order_group: List[Word]):
        """
        Smear out_of_order_group timestamps into time range 
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
        previous: Word =  group_first_word.previous()
        next: Word =  group_last_word.next()
        assert previous is not None and next is not None

        start_boundary: float = max(previous.start, previous.end)
        end_boundary: float = max(next.start, next.end)
        boundary_width: float = math.fabs(end_boundary - start_boundary)
        incremental_width: float = boundary_width/(out_of_order_group_count + 1)
        cursor = start_boundary
        for word in out_of_order_group:
            cursor += incremental_width
            word.update_boundary(cursor, cursor, WordBoundaryOverride.Sequencer)
        
    def __sequence(self):
        """
        Reorders word timestamps to make sure the are sequential (>= last word)
        """

        # get all words, then separate into lists by line_number
        words: List[Word] = self.get_words()
        grouped_by_line: List[List[Word]] = [list(result) for key, result in groupby(words, key=lambda word: word.line_number)]

        # cursor tracks known success as word.start timestamps
        cursor = None
        
        # cycle words, run some psuedo-autotune on the timestamps using line bounds as anchors
        # timestamps are unpredictable, not necessarilly sequential, this reorders the start/end
        for line_of_words in grouped_by_line:
            accumulated_rejects: List[Word] = []
            for word in line_of_words:
                # set default as line start
                cursor = cursor if cursor is not None else word.line_start

                if word.boundary_override in (WordBoundaryOverride.Sequencer,):
                    raise ValueError("Resequencing is not supported.")
                elif word.boundary_override in (WordBoundaryOverride.LineStart, WordBoundaryOverride.Solo):
                    # advance cursor if the word out front
                    if word.line_contained and word.neighbor_contained and word.start >= cursor:
                        cursor = word.start
                    continue
                elif word.boundary_override == WordBoundaryOverride.LineEnd:
                    # make certain we are last in line, as expected
                    assert (word.number == line_of_words[-1].number)
                    break
                elif word.boundary_override == WordBoundaryOverride.Undefined:
                    next: Word = word.next()
                    previous: Word = word.previous()
                    # undefined boundary signals sandwiched status, verify this is the case
                    assert None not in (next, previous)
                    if word.line_contained and word.neighbor_contained and word.start >= cursor:
                        # success, word is where it is expected chronologically
                        # if there are accumulated rejects, they are processed
                        # and cursor advanced
                        self.__splice_times(accumulated_rejects)
                        accumulated_rejects = []
                        cursor = word.start
                        continue
                    else:
                        # word does not order within tolerances, handle with care
                        word_is_sequential_reject = len(accumulated_rejects) == 0 or word.number == accumulated_rejects[-1].number + 1
                        if word_is_sequential_reject:
                            accumulated_rejects.append(word)
                        else:
                            # abort accumulation, dump rejects and reset the accumulation
                            self.__splice_times(accumulated_rejects)
                            accumulated_rejects = [ word ]
                
            # reorder any accumulated rejects
            self.__splice_times(accumulated_rejects)
