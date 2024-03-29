"""
##############################################################################
# Multiscale Universal Interface Code Coupling Library                       #
#                                                                            #
# Copyright (C) 2023 E. R. Fernandez                                         #
#                                                                            #
# This software is jointly licensed under the Apache License, Version 2.0    #
# and the GNU General Public License version 3, you may use it according     #
# to either.                                                                 #
#                                                                            #
# ** Apache License, version 2.0 **                                          #
#                                                                            #
# Licensed under the Apache License, Version 2.0 (the "License");            #
# you may not use this file except in compliance with the License.           #
# You may obtain a copy of the License at                                    #
#                                                                            #
# http://www.apache.org/licenses/LICENSE-2.0                                 #
#                                                                            #
# Unless required by applicable law or agreed to in writing, software        #
# distributed under the License is distributed on an "AS IS" BASIS,          #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   #
# See the License for the specific language governing permissions and        #
# limitations under the License.                                             #
#                                                                            #
# ** GNU General Public License, version 3 **                                #
#                                                                            #
# This program is free software: you can redistribute it and/or modify       #
# it under the terms of the GNU General Public License as published by       #
# the Free Software Foundation, either version 3 of the License, or          #
# (at your option) any later version.                                        #
#                                                                            #
# This program is distributed in the hope that it will be useful,            #
# but WITHOUT ANY WARRANTY; without even the implied warranty of             #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
# GNU General Public License for more details.                               #
#                                                                            #
# You should have received a copy of the GNU General Public License          #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.      #
##############################################################################
#
# @file common.py
# @author E. R. Fernandez
# @date 25 January 2019
# @brief Common functions for MUI Python wrapper.
#
"""

import mui4py.mui4py_mod as mui4py_mod
from mui4py.config import get_default_config
from mui4py.types import map_type, get_float_type_str, get_int_type_str, get_io_type_str
import re
import numpy as np


class CppClass(object):
    def __init__(self, config=None, args=(), kwargs={}):
        self._cpp_class_name = None
        self._cpp_point_class_name = None
        self.raw_point = None
        self.raw = None
        self.io_data_type = None
        # Convert args to Args()
        self.args = tuple([Arg(a) if not issubclass(a.__class__, Arg) else a for a in args])
        self.namespace = ""
        # Filter None-valued entries to take C++ default values.
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.configured = False
        self._ALLOWED_IO_TYPES = None
        if config is None:
            self.config = get_default_config()
        else:
            self.config = config
        self.signature = self._signature()

    def _signature(self):
        sig = self._split_class_name()
        args_str = [str(a) for a in self.get_plain_args()]
        kwargs_str = ["{}={}".format(k, v) for k, v in self.kwargs.items()]
        if args_str:
            sig += "_ARGS_" + "_".join(args_str)
        if kwargs_str:
            sig += "_KWARGS_" + "_".join(kwargs_str)
        return sig

    def _split_class_name(self, title=True):
        tokens = re.findall('[A-Z][^A-Z]*', self.__class__.__name__)
        tokens = [t.lower() for t in tokens]
        if title:
            tokens[0] = tokens[0].title()
        return "_".join(tokens)

    def get_plain_args(self):
        return tuple([a.arg for a in self.args])

    def get_plain_kwargs(self):
        return

    def configure(self, config, io_data_type=None, cpp_obj=None, onlycheck=False):
        self.config = config
        self.point_class_name = get_cpp_name("Point", config.dim,
                                             config.float_type, config.int_type)
        self.raw_point = getattr(mui4py_mod, self.point_class_name)
        self.io_data_type = map_type[io_data_type]
        if self.io_data_type is not None and self.io_data_type not in self._ALLOWED_IO_TYPES:
            raise Exception("Data type not supported by spatial sampler ''. "
                            "Supported types : [float, np.float32, np.float64, etc.]")
        if onlycheck:
            self.io_data_type = None
        self.raw = cpp_obj
        self._cpp_class_name = get_cpp_name(self._split_class_name(), config.dim, config.float_type,
                                            config.int_type, namespace=self.namespace, type_io=self.io_data_type)
        if self.raw is None:
            # Configure class arguments
            for a in self.args:
                a.configure(config, self.raw_point)
            self.raw = getattr(mui4py_mod, self._cpp_class_name)(*self.get_plain_args(), **self.kwargs)
        self.configured = True


class Arg(object):
    def __init__(self, arg):
        self.arg = arg

    def configure(self, config, cpp_point):
        pass


class _Point(Arg):
    def __init__(self, point_rep):
        super(_Point, self).__init__(None)
        self.point_rep = point_rep

    def configure(self, config, cpp_point):
        self.arg = array2Point(self.point_rep, config, cpp_point)


def array2Point(arr, config, cpp_point):
    arr_aux = arr
    if not isinstance(arr, list) and\
       not isinstance(arr, tuple) and\
       not isinstance(arr, np.ndarray):
        arr_aux = [arr]
    # TODO:Maybe check for point type?
    if len(arr_aux) == config.dim:
        return cpp_point(arr_aux)
    else:
        raise Exception("Size of point is different than uniface dimensions.")


def get_cpp_name(cname, dim, float_type, int_type, namespace="", type_io=None):
    s = ""
    if namespace:
        s += "_" + namespace
    s += "_{}{}d_{}_{}".format(cname, dim, get_float_type_str(float_type),
                               get_int_type_str(int_type))
    if type_io is not None:
        s += "_" + get_io_type_str(type_io)
    return s
