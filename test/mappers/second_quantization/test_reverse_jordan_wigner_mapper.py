# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Reverse Jordan Wigner Mapper """

import unittest

from test import QiskitNatureTestCase

from qiskit.opflow import X, Y, Z, I

from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy


#
class TestReverseJordanWignerMapper(QiskitNatureTestCase):
    """Test Reverse Jordan Wigner Mapper"""

    def test_mapping(self):
        """Test mapping to qubit operator"""
        raise NotImplementedError()
        
if __name__ == "__main__":
    unittest.main()
