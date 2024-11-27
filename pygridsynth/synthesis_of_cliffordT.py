from .ring import OMEGA_POWER
from .unitary import DOmegaUnitary
from .normal_form import NormalForm

BIT_SHIFT = [0, 0, 1, 0, 2, 0, 1, 3, 3, 3, 0, 2, 2, 1, 0, 0]
BIT_COUNT = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]


def _reduce_denomexp(unitary):
    T_POWER_and_H = ["H", "TH", "SH", "TSH"]
    residue_z = unitary.z.residue
    residue_w = unitary.w.residue
    residue_squared_z = (unitary.z.u * unitary.z.conj.u).residue

    m = BIT_SHIFT[residue_w] - BIT_SHIFT[residue_z]
    if m < 0:
        m += 4
    if residue_squared_z == 0b0000:
        unitary = unitary.mul_by_H_and_T_power_from_left(0).renew_denomexp(unitary.k - 1)
        return T_POWER_and_H[0], unitary
    elif residue_squared_z == 0b1010:
        unitary = unitary.mul_by_H_and_T_power_from_left(-m).renew_denomexp(unitary.k - 1)
        return T_POWER_and_H[m], unitary
    elif residue_squared_z == 0b0001:
        if BIT_COUNT[residue_z] == BIT_COUNT[residue_w]:
            unitary = unitary.mul_by_H_and_T_power_from_left(-m).renew_denomexp(unitary.k - 1)
            return T_POWER_and_H[m], unitary
        else:
            unitary = unitary.mul_by_H_and_T_power_from_left(-m)
            return T_POWER_and_H[m], unitary


def decompose_domega_unitary(unitary):
    gates = ""
    while unitary.k > 0:
        g, unitary = _reduce_denomexp(unitary)
        gates += g

    if unitary.n & 1:
        gates += "T"
        unitary = unitary.mul_by_T_inv_from_left()
    if unitary.z == 0:
        gates += "X"
        unitary = unitary.mul_by_X_from_left()
    for m in range(8):
        if unitary.z == OMEGA_POWER[m]:
            m_W = m
            unitary = unitary.mul_by_W_power_from_left(-m_W)
            break
    m_S = unitary.n >> 1
    gates += "S" * m_S
    unitary = unitary.mul_by_S_power_from_left(-m_S)
    gates += "W" * m_W

    assert unitary == DOmegaUnitary.identity(), "decomposition failed..."
    gates = NormalForm.from_gates(gates).to_gates()
    return gates
