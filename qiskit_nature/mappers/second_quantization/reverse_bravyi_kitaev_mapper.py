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

class ReverseBravyiKitaevMapper:
    """The Bravyi-Kitaev qubit-to-fermion mapping."""
    def __init__(self):
        # memoization for recursion
        self._x_transform_memo, self._y_transform_memo, self._z_transform_memo = None, None, None
        self._parity_set, self._update_set, self._flip_set, self._remainder_set = [], [], [], []
    
    def _pauli_z_transform(self, index, num_qubits):
        """ Pauli-Z tranformation for bravyi-kitaev mapping """
        if self._z_transform_memo[index] is not None:
            return self._z_transform_memo[index]
        
        z_transform_term = FermionicOp(f"I_{index}", num_qubits) - \
                                            2*FermionicOp(f"N_{index}", num_qubits)
        if not index:
            self._z_transform_memo[index] = z_transform_term.reduce() # memoization
            return z_transform_term
        else:
            if index % 2:
                for flip_idx in self._flip_set[index]:
                    z_transform_term = z_transform_term.compose(self._pauli_z_transform(int(flip_idx), 
                                                                                        num_qubits))
            self._z_transform_memo[index] = z_transform_term.reduce() # memoization
            return z_transform_term
        
    def _pauli_x_transform(self, index, num_qubits):
        """ Pauli-X tranformation for bravyi-kitaev mapping """
        if self._x_transform_memo[index] is not None:
            return self._x_transform_memo[index]
        
        x_transform_term = FermionicOp(f"+_{index}", num_qubits) + FermionicOp(f"-_{index}", num_qubits)
        if index:
            for flip_idx in self._parity_set[index]:
                    x_transform_term = x_transform_term.compose(self._pauli_z_transform(int(flip_idx), 
                                                                                        num_qubits))
        if index == num_qubits - 1:
            self._x_transform_memo[index] = x_transform_term.reduce() # memoization
            return x_transform_term
        
        else:                               
            for update_idx in self._update_set[index]:
                x_transform_term = self._pauli_x_transform(int(update_idx), 
                                                           num_qubits).compose(x_transform_term)
            self._x_transform_memo[index] = x_transform_term.reduce() # memoization
            return x_transform_term
        
    def _pauli_y_transform(self, index, num_qubits):
        """ Pauli-Y tranformation for bravyi-kitaev mapping """
        if self._y_transform_memo[index] is not None:
            return self._y_transform_memo[index]
        
        y_transform_term = 1j*(FermionicOp(f"+_{index}", num_qubits) - FermionicOp(f"-_{index}", num_qubits))
        
        if index:
            if index % 2:
                for rem_idx in self._remainder_set[index]:
                        y_transform_term = y_transform_term.compose(self._pauli_z_transform(int(rem_idx), 
                                                                                            num_qubits))                
            else:
                for flip_idx in self._parity_set[index]:
                        y_transform_term = y_transform_term.compose(self._pauli_z_transform(int(flip_idx), 
                                                                                            num_qubits))                
        if index == num_qubits - 1:
            self._y_transform_memo[index] = y_transform_term.reduce() # memoization
            return y_transform_term
        else:
            for update_idx in self._update_set[index]:
                y_transform_term = self._pauli_x_transform(int(update_idx), 
                                                           num_qubits).compose(y_transform_term)
            self._y_transform_memo[index] = y_transform_term.reduce() # memoization
            return y_transform_term

    def map(self, qubit_op: PauliSumOp) -> FermionicOp:
        
        r""" Maps a `PauliSumOp` to a `FermionicOp` using the bravyi kitaev transform.

            Operators are mapped as follows:
                Z_{j=even} -> (I - 2 a^\dagger_j a_j) -> (I - 2 N_j) 
                Z_{j=odd} -> (I - 2 a^\dagger_j a_j) Z_F(j) -> (I - 2 N_j) Z_F(j)
                X_j -> X_U(j) (a^\dagger_j + a_j) Z_P(j)
                Y_{j=even) -> i X_U(j) (a^\dagger_j - a_j) Z_P(j)
                Y_{j=odd) -> i X_U(j) (a^\dagger_j - a_j) Z_R(j)
                
        Args:
            qubit_op: the :class:`PauliSumOp` to be mapped.

        Returns:
            The `FermionicOp` corresponding to the Hamiltonian in the Fermionic space.
        """
        
        num_qubits = qubit_op.num_qubits # get number of qubits
        
        def parity_set(j, n):
            """
            Computes the parity set of the j-th orbital in n modes.
            Args:
                j (int) : the orbital index
                n (int) : the total number of modes
            Returns:
                numpy.ndarray: Array of mode indices
            """
            indices = np.array([])
            if n % 2 != 0:
                return indices

            if j < n / 2:
                indices = np.append(indices, parity_set(j, n / 2))
            else:
                indices = np.append(
                    indices, np.append(parity_set(j - n / 2, n / 2) + n / 2, n / 2 - 1)
                )
            return indices

        def update_set(j, n):
            """
            Computes the update set of the j-th orbital in n modes.
            Args:
                j (int) : the orbital index
                n (int) : the total number of modes
            Returns:
                numpy.ndarray: Array of mode indices
            """
            indices = np.array([])
            if n % 2 != 0:
                return indices
            if j < n / 2:
                indices = np.append(indices, np.append(n - 1, update_set(j, n / 2)))
            else:
                indices = np.append(indices, update_set(j - n / 2, n / 2) + n / 2)
            return indices

        def flip_set(j, n):
            """
            Computes the flip set of the j-th orbital in n modes.
            Args:
                j (int) : the orbital index
                n (int) : the total number of modes
            Returns:
                numpy.ndarray: Array of mode indices
            """
            indices = np.array([])
            if n % 2 != 0:
                return indices
            if j < n / 2:
                indices = np.append(indices, flip_set(j, n / 2))
            elif n / 2 <= j < n - 1:
                indices = np.append(indices, flip_set(j - n / 2, n / 2) + n / 2)
            else:
                indices = np.append(
                    np.append(indices, flip_set(j - n / 2, n / 2) + n / 2), n / 2 - 1
                )
            return indices
        
        for index in range(num_qubits):
            self._parity_set.append(parity_set(index, num_qubits))
            self._update_set.append(update_set(index, num_qubits))
            self._flip_set.append(flip_set(index, num_qubits))
            self._remainder_set.append(np.setdiff1d(self._parity_set[index], self._flip_set[index]))

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
