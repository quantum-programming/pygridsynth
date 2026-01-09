from .domega_unitary import DOmegaUnitary
from .normal_form import NormalForm
from .quantum_circuit import QuantumCircuit
from .quantum_gate import HGate, QuantumGate, SGate, SXGate, TGate, WGate, w_phase
from .ring import OMEGA_POWER

BIT_SHIFT = [0, 0, 1, 0, 2, 0, 1, 3, 3, 3, 0, 2, 2, 1, 0, 0]
BIT_COUNT = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]


def _reduce_denomexp(
    unitary: DOmegaUnitary, wires: list[int]
) -> tuple[list[QuantumGate], DOmegaUnitary]:
    def t_power_and_h(m: int) -> list[QuantumGate]:
        if m == 0:
            return [HGate(target_qubit=wires[0])]
        elif m == 1:
            return [TGate(target_qubit=wires[0]), HGate(target_qubit=wires[0])]
        elif m == 2:
            return [SGate(target_qubit=wires[0]), HGate(target_qubit=wires[0])]
        elif m == 3:
            return [
                TGate(target_qubit=wires[0]),
                SGate(target_qubit=wires[0]),
                HGate(target_qubit=wires[0]),
            ]
        else:
            raise ValueError

    residue_z = unitary.z.residue
    residue_w = unitary.w.residue
    residue_squared_z = (unitary.z.u * unitary.z.conj.u).residue

    m = BIT_SHIFT[residue_w] - BIT_SHIFT[residue_z]
    if m < 0:
        m += 4
    if residue_squared_z == 0b0000:
        unitary = unitary.mul_by_H_and_T_power_from_left(0).renew_denomexp(
            unitary.k - 1
        )
        return t_power_and_h(0), unitary
    elif residue_squared_z == 0b1010:
        unitary = unitary.mul_by_H_and_T_power_from_left(-m).renew_denomexp(
            unitary.k - 1
        )
        return t_power_and_h(m), unitary
    elif residue_squared_z == 0b0001:
        if BIT_COUNT[residue_z] == BIT_COUNT[residue_w]:
            unitary = unitary.mul_by_H_and_T_power_from_left(-m).renew_denomexp(
                unitary.k - 1
            )
            return t_power_and_h(m), unitary
        else:
            unitary = unitary.mul_by_H_and_T_power_from_left(-m)
            return t_power_and_h(m), unitary
    else:
        raise ValueError


def decompose_domega_unitary(
    unitary: DOmegaUnitary, wires: list[int], up_to_phase: bool = False
) -> QuantumCircuit:
    circuit = QuantumCircuit()
    while unitary.k > 0:
        g, unitary = _reduce_denomexp(unitary, wires=wires)
        circuit += g

    if unitary.n & 1:
        circuit.append(TGate(target_qubit=wires[0]))
        unitary = unitary.mul_by_T_inv_from_left()
    if unitary.z == 0:
        circuit.append(SXGate(target_qubit=wires[0]))
        unitary = unitary.mul_by_X_from_left()

    m_W = 0
    for m in range(8):
        if unitary.z == OMEGA_POWER[m]:
            m_W = m
            unitary = unitary.mul_by_W_power_from_left(-m_W)
            break

    m_S = unitary.n >> 1
    for _ in range(m_S):
        circuit.append(SGate(target_qubit=wires[0]))
    unitary = unitary.mul_by_S_power_from_left(-m_S)
    if up_to_phase:
        circuit.phase = m_W * w_phase()
    else:
        for _ in range(m_W):
            circuit.append(WGate())

    assert unitary == DOmegaUnitary.identity(), "decomposition failed..."
    circuit = NormalForm.from_circuit(circuit).to_circuit(wires=wires)
    return circuit
