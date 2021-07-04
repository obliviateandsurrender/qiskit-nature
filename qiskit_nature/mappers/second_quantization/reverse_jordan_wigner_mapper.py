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
from itertools import product
import math
from typing import List, Tuple

from qiskit.opflow import OperatorBase
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit.quantum_info import Pauli

class ReverseJordanWignerMapper:
    """Reverse Jordan-Wigner mapping."""

    def map(self, second_q_op: OperatorBase) -> FermionicOp:
        """Maps a class:`OperatorBase` to a `FermionicOp`.

        Args:
            second_q_op: the :class:`OperatorBase` to be mapped.

        Returns:
            The `FermionicOp` corresponding to the Hamiltonian in the Fermionic space.
        """

        num_qubits = second_q_op.num_qubits # get number of qubits from input second quantized operator
        fermionic_op = None
        for term in second_q_op: 
            transform_list : List[Tuple(str, float)] = []  # list of tuple(pauli, coeff)
            pauli_op = term.to_pauli_op()
            coef_term = pauli_op.coeff # coefficient for each pauli string term in input second quantized operator
            target_pauli_op = pauli_op.primitive
            for i in range(num_qubits):
                one_pauli = target_pauli_op[num_qubits - 1 - i]
                if one_pauli.to_label() == 'Z': # dealing Pauli Z op
                    transform_list.append((('I', 1), ('N', -2))) # Zj -> I - 2*Nj => [('I', 1), ('N', -2)]
                elif one_pauli.to_label() == 'X': # dealing Pauli X op
                    transform_list.append((('+', 1), ('-', 1))) # Xj -> aj_dag + aj => [('+', 1), ('-', 1)]
                    target_pauli_op &= Pauli("I" * (i+1) + "Z" * (num_qubits - i - 1)) # apply Z(j-1)Z(j-2) ... Z(0)
                elif one_pauli.to_label() == 'Y': # dealing Pauli Y op
                    transform_list.append((('+', -1j), ('-', 1j))) # Yj -> i(aj - aj_dag) => [('+', -1j), ('-', 1j)]
                    target_pauli_op &= Pauli("I" * (i+1) + "Z" * (num_qubits - i - 1)) # apply Z(j-1)Z(j-2) ... Z(0)
                else: # dealing Pauli I op
                    transform_list.append((('I', 0.5), ('I', 0.5))) # Ij -> Ij => [('I', 0.5), ('I', 0.5)]; split I into 0.5I + 0.5I for code consistency
            # dealing the phase
            if target_pauli_op.phase == 1:
                coef_term *= -1j
            elif target_pauli_op.phase == 2:
                coef_term *= -1
            elif target_pauli_op.phase == 3:
                coef_term *= 1j
                
            pauli_coefs = []
            pauli_strings = []
            # create fermionic operator for a term based on transform_list
            for idxes in product(*[[0, 1]]*num_qubits):
                pauli_coefs.append(math.prod([t[i][1] for t, i in zip(transform_list, idxes)]))
                pauli_strings.append("".join([t[i][0] for t, i in zip(transform_list, idxes)])[::-1])
            if not fermionic_op:
                fermionic_op = coef_term * FermionicOp(list(zip(pauli_strings, pauli_coefs))).reduce()
            else:
                fermionic_op += coef_term * FermionicOp(list(zip(pauli_strings, pauli_coefs))).reduce()

        return fermionic_op.reduce()