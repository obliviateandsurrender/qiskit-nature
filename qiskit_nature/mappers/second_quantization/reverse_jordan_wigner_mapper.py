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

class ReverseJordanWignerMapper:
    """Reverse Jordan-Wigner mapping."""

    def map(self, qubit_op: PauliSumOp) -> FermionicOp:
        
        r""" Maps a `PauliSumOp` to a `FermionicOp` using the Jordan-Wigner transform.

            Operators are mapped as follows:
                Z_j -> I - 2 a^\dagger_j a_j -> I - 2 N_j
                X_j -> (a^\dagger_j + a_j) Z_{j-1} Z_{j-2} .. Z_0
                Y_j -> i (a^\dagger_j - a_j) Z_{j-1} Z_{j-2} .. Z_0

        Args:
            qubit_op: the :class:`PauliSumOp` to be mapped.

        Returns:
            The `FermionicOp` corresponding to the Hamiltonian in the Fermionic space.
        """
        
        num_qubits = qubit_op.num_qubits # get number of qubits
        fermionic_op = None 
        fermionic_transforms = []
        
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
                    fermionic_transform = FermionicOp(f"I_{qb_idx}", num_qubits) - \
                                            2*FermionicOp(f"N_{qb_idx}", num_qubits)
                # handle Pauli X and Pauli Y
                else:
                    if op == 'X':
                        fermionic_transform = FermionicOp(f"+_{qb_idx}", num_qubits) + \
                                                FermionicOp(f"-_{qb_idx}", num_qubits)
                    elif op == 'Y': 
                        fermionic_transform = 1j*(FermionicOp(f"+_{qb_idx}", num_qubits) - \
                                                FermionicOp(f"-_{qb_idx}", num_qubits))
                    # handle exchange phase factor
                    for ph_qb_idx in range(qb_idx, 0, -1):
                        phase_term = FermionicOp(f"I_{ph_qb_idx-1}", num_qubits) - \
                                        2*FermionicOp(f"N_{ph_qb_idx-1}", num_qubits)
                        fermionic_transform = fermionic_transform.compose(phase_term)
                    
                qb_fermionic_transforms.append(fermionic_transform)
            
            transformed_term = qb_fermionic_transforms[0] * qb_op.coeff # coefficient of qubit_op term
            for qb_term in qb_fermionic_transforms[1:]: # tensor product for individual term
                transformed_term = transformed_term.compose(qb_term)
            fermionic_transforms.append(transformed_term)
    
        fermionic_op = fermionic_transforms[0] 
        for fermion_term in fermionic_transforms[1:]: # sum up individual fermionic terms
            fermionic_op = fermionic_op.add(fermion_term)
        
        return fermionic_op.reduce() # for merging terms with same labels
