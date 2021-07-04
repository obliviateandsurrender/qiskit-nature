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
import math

from test import QiskitNatureTestCase

from qiskit.opflow import X, Y, Z, I

from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.mappers.second_quantization import ReverseJordanWignerMapper
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.operators.second_quantization import FermionicOp

#
class TestReverseJordanWignerMapper(QiskitNatureTestCase):
    """Test Reverse Jordan Wigner Mapper"""

    def test_mapping_based_on_jordan_wigner_mapper(self):
        """Test mapping to one qubit operator"""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "test_driver_hdf5.hdf5", "drivers/second_quantization/hdf5d"
            )
        )
        q_molecule = driver.run()
        fermionic_op = ElectronicEnergy.from_driver_result(q_molecule).second_q_ops()[0]
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(fermionic_op)
        reverse_mapper = ReverseJordanWignerMapper()
        transformed_fermionic_op = reverse_mapper.map(qubit_op)

        # Assert equal fails because of small difference with 1e-17 order. 
        # self.assertEqual(fermionic_op, transformed_fermionic_op)

        # check RMS difference is almost equal to zero
        self.assertAlmostEqual(math.sqrt(sum([abs(x[1])**2 for x in (fermionic_op - transformed_fermionic_op).reduce().to_list()])), 0)
        
if __name__ == "__main__":
    unittest.main()
