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

"""Fermionic Mapper."""

from abc import abstractmethod

from qiskit.opflow import OperatorBase
from qiskit_nature.operators.second_quantization import FermionicOp


class ReverseJordanWignerMapper:
    """Reverse Jordan-Wigner mapping."""

    def map(self, second_q_op: OperatorBase) -> FermionicOp:
        """Maps a class:`PauliSumOp` to a `FermionicOp`.

        Args:
            second_q_op: the :class:`PauliSumOp` to be mapped.

        Returns:
            The `FermionicOp` corresponding to the Hamiltonian in the Fermionic space.
        """
        raise NotImplementedError()
