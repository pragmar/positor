import sys
import os
import tempfile
import uuid
import subprocess
import re
import piexif
from typing import List, Tuple
import lzstring
from PIL import Image, features
import exiv2

class MetaImage:
    """
    reusable utility functions
    """
    @staticmethod
    def _validate_or_raise(infile, outfile):
        # check for webp, bail if necessary
        has_webp: bool = features.check("webp")
        has_exif: bool = features.check_feature("webp_mux")
        if False in (has_webp, has_exif):
            raise EnvironmentError("Either libwebp/webp_mux is not installed on this system, or Pillow was built without support.")
        if infile is None or not os.path.exists(infile):
            raise IOError("File not provided, or doesn't exist. ({0})".format(infile))
        if os.path.splitext(outfile)[1].lower() != ".webp":
            raise IOError("Outfile is not a .webp. ({0})".format(outfile))
    
    @staticmethod
    def _tag_image(outfile: str, positor_json: str):
        # add positor json to webp metadata (exif2)
        # webp usercomment, i think, can store upwards of 90k, whatever
        # it is, it's pretty near inexhausable for --json-condensed
        lz = lzstring.LZString()
        compressed_positor_comment_b64 = lz.compressToBase64(positor_json)

        # eat stdout/stderr, otherwise stderr:
        # Exif.Photo.UserComment: changed type from 'Comment' to 'Undefined'.
        # everything works though
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        # if I could get PIL exif to work, I'd use that
        # the problem is piexif UserComment always gets dumped as 
        # charset=InvalidCharsetId
        # no documentation on how to work with charset
        # so this is a two pass IO (img, then metadata) situation
        image = exiv2.ImageFactory.open(outfile)
        image.readMetadata()
        data = image.exifData()
        data["Exif.Photo.UserComment"] = ("charset=Ascii {0}".format(compressed_positor_comment_b64))
        image.writeMetadata()

        # reconnect stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

class MetaImageSource(MetaImage):

    @staticmethod
    def export_webp(infile:str, outfile: str, positor_json: str):
        """
        create a copy of source image and stuff positor json within exif UserComment
        """
        MetaImageSource._validate_or_raise(infile, outfile)

        # convert to webp, export to outfile
        waveform_source = Image.open(infile)
        # low footprint png below, adaptive pallete (2 color)
        #waveform_source = waveform_source.convert("P", palette=Image.ADAPTIVE, colors=256)
        waveform_source.save(outfile, "webp", lossless=True, method=6, quality=100, exact=True)
        waveform_source.close()

        MetaImageSource._tag_image(outfile, positor_json)

class MetaImageWaveform(MetaImage):

    """
    a visual representation of a wav/aiff file. Looks exactly like you'd expect,
    carries positor json in exif UserComment.
    """

    @staticmethod
    def export_webp(infile: str, outfile: str, positor_json: str):

        """
        create a waveform image and stuff positor json within exif UserComment
        """
        MetaImageWaveform._validate_or_raise(infile, outfile)
        tempdir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory(prefix="positor_")
        tmpfile_stub = os.path.join(tempdir.name, uuid.uuid4().hex)
        tmpfile = "{0}.png".format(tmpfile_stub)
        # why blue waveform? because it should be a fundamental rgb color, and pure blue
        # is the most pleasant on the eyes of the three. black or white becomes lossy 
        # relative to css filter hue-rotate, and it can't all be clawed back with sepia.         
        # this means certain colors become unattainable with css filters when source is black/white.
        # both black and white can, however, be attained through the use of css 
        # grayscale and brightness, given primary color. so blue is the pliable option.
        # should it be a command arg? maybe, but not a priority at the moment. 
        # -y is overwrite
        ffmpeg_command = ["ffmpeg", "-i", infile, "-y", "-filter_complex", 
          "[0:a]aformat=channel_layouts=mono,compand,showwavespic=s=4096x256:colors=blue", 
          "-frames:v", "1", tmpfile]
        ffmpeg_process = subprocess.Popen( ffmpeg_command, shell=True, stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE)
        stdout, stderr = ffmpeg_process.communicate()

        # convert to webp, export to outfile
        waveform_source = Image.open(tmpfile)
        waveform_source = waveform_source.convert("P", palette=Image.ADAPTIVE, colors=256)
        waveform_source.save(outfile, "webp", lossless=True, method=6, quality=100, exact=True)
        waveform_source.close()

        # we're done with the png used to generated the webp
        tempdir.cleanup()
        # add json to already generated image
        MetaImageWaveform._tag_image(outfile, positor_json)

        

