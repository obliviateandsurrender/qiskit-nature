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

from qiskit.opflow import OperatorBase
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit.opflow.primitive_ops import PauliSumOp

class ReverseParityMapper:
    """The Parity qubit-to-fermion mapping."""
    def __init__(self):
        # memoization for recursion
        self._x_transform_memo, self._y_transform_memo, self._z_transform_memo = None, None, None
    
    def _pauli_z_transform(self, index, num_qubits):
        """ Method for Pauli-Z tranformation for parity mapping """
        if self._z_transform_memo[index] is not None:
            return self._z_transform_memo[index]
        
        z_transform_term = FermionicOp(f"I_{index}", num_qubits) - \
                                            2*FermionicOp(f"N_{index}", num_qubits)
        if not index:
            self._z_transform_memo[index] = z_transform_term.reduce() # memoization
            return z_transform_term
        else:
            z_transform_term = z_transform_term.compose(self._pauli_z_transform(index-1, num_qubits))
            self._z_transform_memo[index] = z_transform_term.reduce() # memoization
            return z_transform_term
        
    def _pauli_x_transform(self, index, num_qubits):
        """ Method for Pauli-X tranformation for parity mapping """
        if self._x_transform_memo[index] is not None:
            return self._x_transform_memo[index]
        
        x_transform_term = FermionicOp(f"+_{index}", num_qubits) + FermionicOp(f"-_{index}", num_qubits)
        if index:
                x_transform_term = x_transform_term.compose(self._pauli_z_transform(index-1, num_qubits))
        
        if index == num_qubits - 1:
            self._x_transform_memo[index] = x_transform_term.reduce() # memoization
            return x_transform_term
        
        else:                               
            for idx in range(index+1, num_qubits):
                x_transform_term = self._pauli_x_transform(idx, num_qubits).compose(x_transform_term)
            self._x_transform_memo[index] = x_transform_term.reduce() # memoization
            return x_transform_term
        
    def _pauli_y_transform(self, index, num_qubits):
        """ Method for Pauli-Y tranformation for parity mapping """
        if self._y_transform_memo[index] is not None:
            return self._y_transform_memo[index]
        
        y_transform_term = 1j*(FermionicOp(f"+_{index}", num_qubits) - FermionicOp(f"-_{index}", num_qubits))
        
        if index == num_qubits - 1:
            self._y_transform_memo[index] = y_transform_term.reduce() # memoization
            return y_transform_term
        else:
            for idx in range(index+1, num_qubits):
                y_transform_term = self._pauli_x_transform(idx, num_qubits).compose(y_transform_term)
            self._y_transform_memo[index] = y_transform_term.reduce() # memoization
            return y_transform_term

    def map(self, qubit_op: PauliSumOp) -> FermionicOp:
        
        r""" Maps a `PauliSumOp` to a `FermionicOp` using the Parity transform.

            Operators are mapped as follows:
                Z_j -> (I - 2 a^\dagger_j a_j) Z_{j-1} -> (I - 2 N_j) Z_{j-1}
                X_j -> X_{n-1} X_{n-2} .. X_{j+1} (a^\dagger_j + a_j) Z_{j-1}
                Y_j -> i X_{n-1} X_{n-2} .. X_{j+1} (a^\dagger_j - a_j) 

        Args:
            qubit_op: the :class:`PauliSumOp` to be mapped.

        Returns:
            The `FermionicOp` corresponding to the Hamiltonian in the Fermionic space.
        """
        
        num_qubits = qubit_op.num_qubits # get number of qubits
        fermionic_op = None 
        fermionic_transforms = []
        
        self._x_transform_memo = [None]*num_qubits
        self._y_transform_memo = [None]*num_qubits
        self._z_transform_memo = [None]*num_qubits 
        
        for qb_op in qubit_op.to_pauli_op():
            qb_fermionic_transforms = [] # store ferm_op for individual term
            # reverse traversing in accordance with qubit numbering
            for qb_idx, op in enumerate(reversed(qb_op.primitive.to_label())):
                
                fermionic_transform = None # store transform for individual pauli terms
    
                # handle Pauli I (for consistency)
                if op == 'I':
                    fermionic_transform = FermionicOp(f"I_{qb_idx}", num_qubits)
                # handle Pauli Z
                elif op == 'Z':
                    fermionic_transform = self._pauli_z_transform(qb_idx, num_qubits) 
                # handle Pauli X 
                elif op == 'X':
                        fermionic_transform = self._pauli_x_transform(qb_idx, num_qubits)   
                # handle Pauli Y
                elif op == 'Y': 
                        fermionic_transform = self._pauli_y_transform(qb_idx, num_qubits)
                    
                qb_fermionic_transforms.append(fermionic_transform)
            
            transformed_term = qb_fermionic_transforms[0] * qb_op.coeff # coefficient of qubit_op term
            for qb_term in qb_fermionic_transforms[1:]: # tensor product for individual term
                transformed_term = transformed_term.compose(qb_term)
            fermionic_transforms.append(transformed_term.reduce())
    
        fermionic_op = fermionic_transforms[0] 
        for fermion_term in fermionic_transforms[1:]: # sum up individual fermionic terms
            fermionic_op = fermionic_op.add(fermion_term)
        fermionic_op = fermionic_op.reduce() # for merging terms with same labels
        
        return fermionic_op
