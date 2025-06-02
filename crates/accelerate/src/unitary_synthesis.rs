// --------------------------
// Rust Implementation (lib.rs)
// --------------------------
use pyo3::prelude::*;
use indexmap::IndexSet;

/// Python-compatible Euler basis enum
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EulerBasis {
    U3,
    U321,
    ZSX,
    ZSXX
}

/// Python-exposed basis set container
#[pyclass]
pub struct EulerBasisSet {
    inner: IndexSet<EulerBasis>
}

#[pymethods]
impl EulerBasisSet {
    #[new]
    fn new() -> Self {
        EulerBasisSet { inner: IndexSet::new() }
    }

    #[pyo3(name = "add_basis")]
    fn add_basis_py(&mut self, basis: EulerBasis) {
        self.inner.insert(basis);
    }

    #[pyo3(name = "get_bases")]
    fn get_bases_py(&self) -> Vec<EulerBasis> {
        self.inner.iter().copied().collect()
    }
}

const EULER_BASES: &[&[&str]] = &[
    &["u3"], 
    &["u3", "u2", "u1"],
    &["z", "sx"],
    &["z", "sx", "x"]
];

/// Qiskit-compatible basis analysis (Python entry point)
#[pyfunction]
#[pyo3(name = "get_euler_basis_set")]
fn get_euler_basis_set_py(
    basis_gates: Vec<&str>
) -> PyResult<EulerBasisSet> {
    let basis_set = basis_gates
        .into_iter()
        .collect::<IndexSet<_>>();
    
    let mut result = EulerBasisSet::new();
    
    for (idx, gates) in EULER_BASES.iter().enumerate() {
        if gates.iter().all(|g| basis_set.contains(g)) {
            let basis = match idx {
                0 => EulerBasis::U3,
                1 => EulerBasis::U321,
                2 => EulerBasis::ZSX,
                3 => EulerBasis::ZSXX,
                _ => continue
            };
            result.add_basis_py(basis);
        }
    }

    // Qiskit-specific basis prioritization
    if result.get_bases_py().contains(&EulerBasis::U321) {
        result.inner.swap_remove(&EulerBasis::U3);
    }
    if result.get_bases_py().contains(&EulerBasis::ZSXX) {
        result.inner.swap_remove(&EulerBasis::ZSX);
    }

    Ok(result)
}

///
