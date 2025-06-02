// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
#![allow(clippy::too_many_arguments)]

use std::f64::consts::PI;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use approx::relative_eq;
use hashbrown::{HashMap, HashSet};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use ndarray::prelude::*;
use num_complex::{Complex, Complex64};
use numpy::IntoPyArray;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use smallvec::{smallvec, SmallVec};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyString, PyType};
use pyo3::wrap_pyfunction;
use pyo3::Python;

use qiskit_circuit::converters::{circuit_to_dag, QuantumCircuitData};
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports;
use qiskit_circuit::operations::{Operation, OperationRef, Param, PyGate, StandardGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::Qubit;

use crate::euler_one_qubit_decomposer::{
    unitary_to_gate_sequence_inner, EulerBasis, EulerBasisSet, EULER_BASES, EULER_BASIS_NAMES,
};
use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::{NormalOperation, Target};
use crate::two_qubit_decompose::{
    RXXEquivalent, TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer, TwoQubitGateSequence,
    TwoQubitWeylDecomposition,
};
use crate::QiskitError;

const PI2: f64 = PI / 2.;
const PI4: f64 = PI / 4.;

#[derive(Clone, Debug)]
enum DecomposerType {
    TwoQubitBasis(Box<TwoQubitBasisDecomposer>),
    TwoQubitControlledU(Box<TwoQubitControlledUDecomposer>),
    XX(PyObject),
}

#[derive(Clone, Debug)]
struct DecomposerElement {
    decomposer: DecomposerType,
    packed_op: PackedOperation,
    params: SmallVec<[Param; 3]>,
}

#[derive(Clone, Debug)]
struct TwoQubitUnitarySequence {
    gate_sequence: TwoQubitGateSequence,
    decomp_op: PackedOperation,
    decomp_params: SmallVec<[Param; 3]>,
}

// These two variables are used to exit the decomposer search early in
// `get_2q_decomposers_from_target`.
// If the available 2q basis is a subset of GOODBYE_SET, TwoQubitBasisDecomposer provides
// an ideal decomposition and we can exit the decomposer search. Similarly, if it is a
// subset of PARAM_SET, TwoQubitControlledUDecomposer provides an ideal decompostion.
static GOODBYE_SET: [&str; 3] = ["cx", "cz", "ecr"];
static PARAM_SET: [&str; 8] = ["rzz", "rxx", "ryy", "rzx", "crx", "cry", "crz", "cphase"];

/// Given a list of basis gates, find a corresponding euler basis to use.
/// This will determine the available 1q synthesis basis for different decomposers.
fn get_euler_basis_set(basis_list: IndexSet<&str>) -> EulerBasisSet {
    let mut euler_basis_set: EulerBasisSet = EulerBasisSet::new();
    EULER_BASES
        .iter()
        .enumerate()
        .filter_map(|(idx, gates)| {
            if !gates.iter().all(|gate| basis_list.contains(gate)) {
                return None;
            }
            let basis = EULER_BASIS_NAMES[idx];
            Some(basis)
        })
        .for_each(|basis| euler_basis_set.add_basis(basis));

    if euler_basis_set.basis_supported(EulerBasis::U3)
        && euler_basis_set.basis_supported(EulerBasis::U321)
    {
        euler_basis_set.remove(EulerBasis::U3);
    }
    if euler_basis_set.basis_supported(EulerBasis::ZSX)
        && euler_basis_set.basis_supported(EulerBasis::ZSXX)
    {
        euler_basis_set.remove(EulerBasis::ZSX);
    }
    euler_basis_set
}

/// Given a `Target`, find an euler basis that is supported for a specific `PhysicalQubit`.
/// This will determine the available 1q synthesis basis for different decomposers.
fn get_target_basis_set(target: &Target, qubit: PhysicalQubit) -> EulerBasisSet {
    let mut target_basis_set: EulerBasisSet = EulerBasisSet::new();
    let target_basis_list = target.operation_names_for_qargs(Some(&smallvec![qubit]));
    match target_basis_list {
        Ok(basis_list) => {
            target_basis_set = get_euler_basis_set(basis_list.into_iter().collect());
        }
        Err(_) => {
            target_basis_set.support_all();
            target_basis_set.remove(EulerBasis::U3);
            target_basis_set.remove(EulerBasis::ZSX);
        }
    }
    target_basis_set
}

/// Apply synthesis output (`synth_dag`) to final `DAGCircuit` (`out_dag`).
/// `synth_dag` is a subgraph, and the `qubit_ids` are relative to the subgraph
///  size/orientation, so `out_qargs` is used to track the final qubit ids where
/// it should be applied.
fn apply_synth_dag(
    py: Python<'_>,
    out_dag: &mut DAGCircuit,
    out_qargs: &[Qubit],
    synth_dag: &DAGCircuit,
) -> PyResult<()> {
    for out_node in synth_dag.topological_op_nodes()? {
        let mut out_packed_instr = synth_dag[out_node].unwrap_operation().clone();
        let synth_qargs = synth_dag.get_qargs(out_packed_instr.qubits);
        let mapped_qargs: Vec<Qubit> = synth_qargs
            .iter()
            .map(|qarg| out_qargs[qarg.0 as usize])
            .collect();
        out_packed_instr.qubits = out_dag.qargs_interner.insert(&mapped_qargs);
        out_dag.push_back(py, out_packed_instr)?;
    }
    out_dag.add_global_phase(py, &synth_dag.get_global_phase())?;
    Ok(())
}

/// Apply synthesis output (`sequence`) to final `DAGCircuit` (`out_dag`).
/// `sequence` contains a representation of gates to be applied to a subgraph,
/// and the `qubit_ids` are relative to the subgraph size/orientation,
/// so `out_qargs` is used to track the final qubit ids where they should be applied.
fn apply_synth_sequence(
    py: Python<'_>,
    out_dag: &mut DAGCircuit,
    out_qargs: &[Qubit],
    sequence: &TwoQubitUnitarySequence,
) -> PyResult<()> {
    let mut instructions = Vec::with_capacity(sequence.gate_sequence.gates().len());
    for (gate, params, qubit_ids) in sequence.gate_sequence.gates() {
        let packed_op = match gate {
            None => &sequence.decomp_op,
            Some(gate) => &PackedOperation::from_standard_gate(*gate),
        };
        let mapped_qargs: Vec<Qubit> = qubit_ids.iter().map(|id| out_qargs[*id as usize]).collect();
        let new_params: Option<Box<SmallVec<[Param; 3]>>> = match gate {
            Some(_) => Some(Box::new(params.iter().map(|p| Param::Float(*p)).collect())),
            None => {
                if !sequence.decomp_params.is_empty()
                    && matches!(sequence.decomp_params[0], Param::Float(_))
                {
                    Some(Box::new(sequence.decomp_params.clone()))
                } else {
                    Some(Box::new(params.iter().map(|p| Param::Float(*p)).collect()))
                }
            }
        };

        let new_op: PackedOperation = match packed_op.py_copy(py)?.view() {
            OperationRef::Gate(gate) => {
                gate.gate.setattr(
                    py,
                    "params",
                    new_params
                        .as_deref()
                        .map(SmallVec::as_slice)
                        .unwrap_or(&[])
                        .iter()
                        .map(|param| param.clone_ref(py))
                        .collect::<SmallVec<[Param; 3]>>(),
                )?;
                Box::new(PyGate {
                    gate: gate.gate.clone(),
                    qubits: gate.qubits,
                    clbits: gate.clbits,
                    params: gate.params,
                    op_name: gate.op_name.clone(),
                })
                .into()
            }
            OperationRef::StandardGate(_) => packed_op.clone(),
            _ => {
                return Err(QiskitError::new_err(
                    "Decomposed gate sequence contains unexpected operations.",
                ))
            }
        };

        let instruction = PackedInstruction {
            op: new_op,
            qubits: out_dag.qargs_interner.insert(&mapped_qargs),
            clbits: out_dag.cargs_interner.get_default(),
            params: new_params,
            label: None,
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        };
        instructions.push(instruction);
    }
    out_dag.extend(py, instructions.into_iter())?;
    out_dag.add_global_phase(py, &Param::Float(sequence.gate_sequence.global_phase()))?;
    Ok(())
}

/// Iterate over `DAGCircuit` to perform unitary synthesis.
/// For each elegible gate: find decomposers, select the synthesis
/// method with the highest fidelity score and apply decompositions. The available methods are:
///     * 1q synthesis: OneQubitEulerDecomposer
///     * 2q synthesis: TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer, XXDecomposer (Python, only if target is provided)
///     * 3q+ synthesis: QuantumShannonDecomposer (Python)
/// This function is currently used in the Python `UnitarySynthesis`` transpiler pass as a replacement for the `_run_main_loop` method.
/// It returns a new `DAGCircuit` with the different synthesized gates.
#[pyfunction]
#[pyo3(name = "run_main_loop", signature=(dag, qubit_indices, min_qubits, target, basis_gates, synth_gates, coupling_edges, approximation_degree=None, natural_direction=None, pulse_optimize=None))]
fn py_run_main_loop(
    py: Python,
    dag: &mut DAGCircuit,
    qubit_indices: Vec<usize>,
    min_qubits: usize,
    target: Option<&Target>,
    basis_gates: HashSet<String>,
    synth_gates: HashSet<String>,
    coupling_edges: HashSet<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<bool>,
) -> PyResult<DAGCircuit> {
    // We need to use the python converter because the currently available Rust conversion
    // is lossy. We need `QuantumCircuit` instances to be used in `replace_blocks`.
    let dag_to_circuit = imports::DAG_TO_CIRCUIT.get_bound(py);

    let mut out_dag = dag.copy_empty_like(py, "alike")?;

    // Iterate over dag nodes and determine unitary synthesis approach
    for node in dag.topological_op_nodes()? {
        let mut packed_instr = dag[node].unwrap_operation().clone();

        if packed_instr.op.control_flow() {
            let OperationRef::Instruction(py_instr) = packed_instr.op.view() else {
                unreachable!("Control flow op must be an instruction")
            };
            let raw_blocks: Vec<PyResult<Bound<PyAny>>> = py_instr
                .instruction
                .getattr(py, "blocks")?
                .bind(py)
                .try_iter()?
                .collect();
            let mut new_blocks = Vec::with_capacity(raw_blocks.len());
            for raw_block in raw_blocks {
                let new_ids = dag
                    .get_qargs(packed_instr.qubits)
                    .iter()
                    .map(|qarg| qubit_indices[qarg.0 as usize])
                    .collect_vec();
                let res = py_run_main_loop(
                    py,
                    &mut circuit_to_dag(
                        py,
                        QuantumCircuitData::extract_bound(&raw_block?)?,
                        false,
                        None,
                        None,
                    )?,
                    new_ids,
                    min_qubits,
                    target,
                    basis_gates.clone(),
                    synth_gates.clone(),
                    coupling_edges.clone(),
                    approximation_degree,
                    natural_direction,
                    pulse_optimize,
                )?;
                new_blocks.push(dag_to_circuit.call1((res,))?);
            }
# --------------------------
# Python Integration (transpiler.py)
# --------------------------
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit_unitary_synth import EulerBasis, get_euler_basis_set

class RustBasisAnalysis(AnalysisPass):
    """Qiskit transpiler pass using Rust-accelerated basis analysis"""
    
    def __init__(self):
        super().__init__()
        self.required = ["euler_basis"]

    def run(self, dag):
        basis_gates = self.backend.configuration().basis_gates
        rust_basis = get_euler_basis_set(basis_gates)
        
        # Convert to Qiskit's native basis representation
        self.property_set['euler_basis'] = [
            basis.name.lower() for basis in rust_basis.get_bases()
        ]
