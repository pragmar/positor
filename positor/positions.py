import os

class JsonPositions:

    @staticmethod
    def __get_json_format(extractor: str, absolute: bool) -> str:
        """
        Utility reusable, returns json output code.
        @absolute - use absolute positions
        """
        return "{0}{1}".format(extractor, "%" if absolute else "#")

    @staticmethod
    def __get_common_json(file_name: str, absolute: bool, extractor: str, positor_version: str) -> dict:
        """
        Utility reusable, create base object for positor json
        @file_name - filename of file processed by stt (original)
        @duration - duration, in seconds
        @absolute - (or relative positions) boolean
        """
        return {
            "__meta__": {
                "application":"positor/{0}".format(positor_version),
                "format": JsonPositions.__get_json_format(extractor, absolute),
                "source": {
                    "name": os.path.basename(file_name),
                }
            },
            "text": "",
            "positions": [],
        }

    @staticmethod
    def get_stt_json(file_name, duration, absolute, positor_version) -> dict:
        """
        Utility reusable, create base object for positor json
        @file_name - filename of file processed by stt (original)
        @duration - duration, in seconds
        @absolute - (or relative positions) boolean
        """
        stt_json = JsonPositions.__get_common_json(file_name, absolute, "stt", positor_version)
        stt_json["__meta__"]["source"]["duration"] = duration
        return stt_json

    @staticmethod
    def get_ocr_json(file_name, width, height, absolute, positor_version) -> dict:
        """
        Utility reusable, create base object for positor json
        @file_name - filename of file processed by ocr (original)
        @width/height - dimensions, in pixels
        @absolute - (or relative positions) boolean
        """
        ocr_json = JsonPositions.__get_common_json(file_name, absolute, "ocr", positor_version)
        ocr_json["__meta__"]["source"]["width"] = width
        ocr_json["__meta__"]["source"]["height"] = height
        return ocr_json
