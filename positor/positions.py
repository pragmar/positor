import os

class JsonPositions:

    @staticmethod
    def __get_json_format(extractor: str, condensed: bool, absolute: bool) -> str:
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
                "schema": JsonPositions.__get_json_format(extractor, condensed, absolute),
                "source": {
                    "name": os.path.basename(file_name),
                }
            },
            "text": "",
            "positions": [],
        }

    @staticmethod
    def get_stt_json(file_name, duration, condensed, absolute, positor_version) -> dict:
        """
        Utility reusable, create base object for positor json
        @file_name - filename of file processed by stt (original)
        @duration - duration, in seconds
        @absolute - (or relative positions) boolean
        """
        # file_name: str, extractor: str, condensed: bool, absolute: bool, positor_version: str
        stt_json = JsonPositions.__get_common_json(file_name, "stt", condensed, absolute,  positor_version)
        stt_json["__meta__"]["source"]["duration"] = duration
        return stt_json

    @staticmethod
    def get_ocr_json(file_name, width, height, condensed, absolute, positor_version) -> dict:
        """
        Utility reusable, create base object for positor json
        @file_name - filename of file processed by ocr (original)
        @width/height - dimensions, in pixels
        @absolute - (or relative positions) boolean
        """
        ocr_json = JsonPositions.__get_common_json(file_name, "ocr", condensed, absolute, positor_version)
        ocr_json["__meta__"]["source"]["width"] = width
        ocr_json["__meta__"]["source"]["height"] = height
        return ocr_json
