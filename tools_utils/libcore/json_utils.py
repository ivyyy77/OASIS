

import numpy as np
import base64

def readMatrixFromJson(value):
    rows = len(value)
    if (rows == 0):
        return np.ndarray([0, 0])

    cols = len(value[0])
    mat = np.zeros([rows, cols])
    for i in range(0, rows):
        for j in range(0, cols):
            mat[i][j] = value[i][j]
    return mat

def readVectorFromJson(value):
    num = len(value)
    vec = np.zeros(num)
    for i in range(0, num):
        vec[i] = value[i]
    return vec

def loadMatFromJson(value):
    H = value["height"]
    W = value["width"]

    mat = None
    if H <= 0 or W <= 0:
        return mat

    if value['dtype'] == "8U":
        mat = np.frombuffer(base64.b64decode(value['data']), dtype=np.uint8).reshape([H, W])
    elif value['dtype'] == "8UC3":
        mat = np.frombuffer(base64.b64decode(value['data']), dtype=np.uint8).reshape([H, W, 3])
    elif value['dtype'] == "8UC4":
        mat = np.frombuffer(base64.b64decode(value['data']), dtype=np.uint8).reshape([H, W, 4])
    elif value['dtype'] == "16U":
        mat = np.frombuffer(base64.b64decode(value['data']), dtype=np.uint16).reshape([H, W])
    elif value['dtype'] == "32F":
        mat = np.frombuffer(base64.b64decode(value['data']), dtype=np.float32).reshape([H, W])
    else:
        return mat

    return mat




import json
from _ctypes import PyObj_FromPtr
import re

class JsonNoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value

def json_format_ndarray(dict):
    for key in dict:
        if isinstance(dict[key], np.ndarray):
            dict[key] = JsonNoIndent(dict[key].tolist())
        elif isinstance(dict[key], dict):
            dict[key] = json_format_ndarray(dict[key])

class FormatEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):

        self.__sort_keys = kwargs.get('sort_keys', None)
        super(FormatEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, JsonNoIndent)
                else super(FormatEncoder, self).default(obj))

    def encode(self, obj):
        json_format_ndarray(obj)

        format_spec = self.FORMAT_SPEC
        json_repr = super(FormatEncoder, self).encode(obj)




        for match in self.regex.finditer(json_repr):

            uid = int(match.group(1))
            no_indent = PyObj_FromPtr(uid)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)



            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(uid)), json_obj_repr)

        return json_repr

def write_to_json(fn, dict):
    dict_json = json.dumps(dict, cls=FormatEncoder, indent=1)
    with open(fn, 'w') as f:
        f.write(dict_json)
        f.close()
