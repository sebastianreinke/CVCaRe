"""
This file is part of CVCaRe. It is a cyclic voltammogram analysis tool that enables you to calculate capacitance and resistance from capacitive cyclic voltammograms.
    Copyright (C) 2022-2024 Sebastian Reinke

CVCaRe is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

CVCaRe is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with CVCaRe. If not, see <https://www.gnu.org/licenses/>.

"""
import functools
import time

def time_this_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time:.6f} seconds.")
    return wrapper
