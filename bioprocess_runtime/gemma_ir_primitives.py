from __future__ import annotations

import hashlib
import itertools
from typing import Any

from .gemma_ir import SEMANTICS, verify_gemma_ir
from .serialization import canonical_json

try:
    import torch
    from torch.nn import functional
except ImportError:
    torch = None
    functional = None


EXACT_INDEX_OPCODES = frozenset(
    {
        "ARANGE",
        "EMBEDDING",
        "CAUSAL_MASK",
        "RESHAPE_TRANSPOSE_HEADS",
        "REPEAT_KV",
        "TRANSPOSE_RESHAPE_HEADS",
        "SLICE_LAST_TOKEN",
        "ARGMAX",
    }
)
NUMERICAL_OPCODES = frozenset(SEMANTICS) - EXACT_INDEX_OPCODES


def _require_torch() -> None:
    if torch is None or functional is None:
        raise RuntimeError("Gemma primitive qualification requires PyTorch")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _append_case(
    records: list[dict[str, Any]],
    opcode: str,
    case_id: str,
    expected: Any,
    actual: Any,
) -> None:
    body = {
        "opcode": opcode,
        "case_id": case_id,
        "expected_sha256": _sha256(expected),
        "actual_sha256": _sha256(actual),
        "exact": expected == actual,
    }
    records.append({**body, "record_sha256": _sha256(body)})


def _index_cases() -> list[dict[str, Any]]:
    _require_torch()
    records: list[dict[str, Any]] = []

    for length in range(1, 9):
        expected = list(range(length))
        actual = torch.arange(length, dtype=torch.int64).tolist()
        _append_case(records, "ARANGE", f"length_{length}", expected, actual)

    for dtype in (torch.float32, torch.bfloat16):
        weight = torch.arange(15, dtype=dtype).reshape(5, 3)
        for case_index, ids in enumerate(([[0]], [[4, 1, 4]], [[2, 0], [3, 1]])):
            expected = [
                [[float(weight[token, coordinate]) for coordinate in range(3)] for token in row]
                for row in ids
            ]
            actual = functional.embedding(torch.tensor(ids), weight).tolist()
            _append_case(
                records,
                "EMBEDDING",
                f"dtype_{dtype}_ids_{case_index}",
                expected,
                actual,
            )

    for dtype, blocked in (
        (torch.float32, -float.fromhex("0x1.fffffep+127")),
        (torch.bfloat16, -float.fromhex("0x1.fep+127")),
    ):
        for sequence in range(1, 7):
            for window in (None, 1, 2, 4):
                expected_matrix = []
                for query in range(sequence):
                    row = []
                    for key in range(sequence):
                        allowed = key <= query and (
                            window is None or key > query - window
                        )
                        row.append(0.0 if allowed else blocked)
                    expected_matrix.append(row)
                row_index = torch.arange(sequence)[:, None]
                column_index = torch.arange(sequence)[None, :]
                allowed = column_index <= row_index
                if window is not None:
                    allowed = allowed & (column_index > row_index - window)
                zero = torch.zeros((sequence, sequence), dtype=dtype)
                actual = torch.where(
                    allowed, zero, torch.full_like(zero, torch.finfo(dtype).min)
                ).tolist()
                _append_case(
                    records,
                    "CAUSAL_MASK",
                    f"dtype_{dtype}_sequence_{sequence}_window_{window}",
                    expected_matrix,
                    actual,
                )

    for batch, sequence, heads, dimension in itertools.product(
        range(1, 3), range(1, 4), range(1, 4), range(1, 4)
    ):
        source = torch.arange(
            batch * sequence * heads * dimension, dtype=torch.int64
        ).reshape(batch, sequence, heads * dimension)
        expected = [
            [
                [
                    [
                        int(source[b, s, h * dimension + d])
                        for d in range(dimension)
                    ]
                    for s in range(sequence)
                ]
                for h in range(heads)
            ]
            for b in range(batch)
        ]
        actual = source.view(batch, sequence, heads, dimension).transpose(1, 2).tolist()
        _append_case(
            records,
            "RESHAPE_TRANSPOSE_HEADS",
            f"b{batch}_s{sequence}_h{heads}_d{dimension}",
            expected,
            actual,
        )

    for batch, kv_heads, repetitions, sequence, dimension in itertools.product(
        range(1, 3), range(1, 3), range(1, 4), range(1, 3), range(1, 3)
    ):
        source = torch.arange(
            batch * kv_heads * sequence * dimension, dtype=torch.int64
        ).reshape(batch, kv_heads, sequence, dimension)
        expected = [
            [
                [
                    [int(source[b, h // repetitions, s, d]) for d in range(dimension)]
                    for s in range(sequence)
                ]
                for h in range(kv_heads * repetitions)
            ]
            for b in range(batch)
        ]
        expanded = source[:, :, None, :, :].expand(
            batch, kv_heads, repetitions, sequence, dimension
        )
        actual = expanded.reshape(
            batch, kv_heads * repetitions, sequence, dimension
        ).tolist()
        _append_case(
            records,
            "REPEAT_KV",
            f"b{batch}_k{kv_heads}_r{repetitions}_s{sequence}_d{dimension}",
            expected,
            actual,
        )

    for batch, heads, sequence, dimension in itertools.product(
        range(1, 3), range(1, 4), range(1, 4), range(1, 4)
    ):
        source = torch.arange(
            batch * heads * sequence * dimension, dtype=torch.int64
        ).reshape(batch, heads, sequence, dimension)
        expected = [
            [
                [
                    int(source[b, h, s, d])
                    for h in range(heads)
                    for d in range(dimension)
                ]
                for s in range(sequence)
            ]
            for b in range(batch)
        ]
        actual = source.transpose(1, 2).contiguous().reshape(
            batch, sequence, heads * dimension
        ).tolist()
        _append_case(
            records,
            "TRANSPOSE_RESHAPE_HEADS",
            f"b{batch}_h{heads}_s{sequence}_d{dimension}",
            expected,
            actual,
        )

    for batch, sequence, hidden in itertools.product(
        range(1, 3), range(1, 5), range(1, 5)
    ):
        source = torch.arange(batch * sequence * hidden, dtype=torch.int64).reshape(
            batch, sequence, hidden
        )
        expected = [
            [[int(source[b, sequence - 1, h]) for h in range(hidden)]]
            for b in range(batch)
        ]
        actual = source[:, -1:, :].tolist()
        _append_case(
            records,
            "SLICE_LAST_TOKEN",
            f"b{batch}_s{sequence}_h{hidden}",
            expected,
            actual,
        )

    for length in range(2, 6):
        for values in itertools.product((-1, 0, 1), repeat=length):
            maximum = max(values)
            expected = values.index(maximum)
            actual = int(torch.argmax(torch.tensor(values, dtype=torch.int64)).item())
            _append_case(records, "ARGMAX", f"values_{'_'.join(map(str, values))}", expected, actual)

    return records


def build_primitive_qualification_certificate() -> dict[str, Any]:
    records = _index_cases()
    case_counts = {
        opcode: sum(record["opcode"] == opcode for record in records)
        for opcode in sorted(EXACT_INDEX_OPCODES)
    }
    body = {
        "schema_version": 1,
        "scope": "Independent pure-Python coordinate-oracle conformance over declared finite test domains for non-arithmetic Gemma IR primitives; this is not unrestricted-domain or floating-point primitive qualification.",
        "oracle": "pure Python coordinate and selection equations",
        "comparator": "PyTorch tensor operations used by the IR dispatcher",
        "semantics_sha256": {
            opcode: _sha256(SEMANTICS[opcode]) for opcode in sorted(EXACT_INDEX_OPCODES)
        },
        "records": records,
        "case_counts": case_counts,
        "all_cases_exact": all(record["exact"] for record in records),
        "independently_tested_opcodes": sorted(EXACT_INDEX_OPCODES),
        "unqualified_numerical_opcodes": sorted(NUMERICAL_OPCODES),
        "unrestricted_domain_qualification_established": False,
        "floating_point_primitive_qualification_established": False,
    }
    return {**body, "certificate_sha256": _sha256(body)}


def verify_primitive_qualification_certificate(
    certificate: dict[str, Any], reexecute: bool = True
) -> dict[str, Any]:
    if not isinstance(certificate, dict):
        return {"valid": False}
    body = {
        key: value for key, value in certificate.items() if key != "certificate_sha256"
    }
    try:
        certificate_hash_valid = _sha256(body) == certificate.get("certificate_sha256")
    except (TypeError, ValueError):
        return {"valid": False}
    records = certificate.get("records")

    def record_valid(record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        try:
            body = {
                key: value for key, value in record.items() if key != "record_sha256"
            }
            return bool(
                record.get("opcode") in EXACT_INDEX_OPCODES
                and record.get("exact") is True
                and record.get("expected_sha256") == record.get("actual_sha256")
                and record.get("record_sha256") == _sha256(body)
            )
        except (TypeError, ValueError, AttributeError):
            return False

    records_valid = bool(
        isinstance(records, list) and records and all(record_valid(record) for record in records)
    )
    calculated_counts = {
        opcode: sum(
            isinstance(record, dict) and record.get("opcode") == opcode
            for record in records
        )
        for opcode in sorted(EXACT_INDEX_OPCODES)
    } if isinstance(records, list) else {}
    claims_consistent = bool(
        certificate.get("schema_version") == 1
        and certificate.get("case_counts") == calculated_counts
        and all(count > 0 for count in calculated_counts.values())
        and certificate.get("all_cases_exact") is True
        and certificate.get("independently_tested_opcodes")
        == sorted(EXACT_INDEX_OPCODES)
        and certificate.get("unqualified_numerical_opcodes")
        == sorted(NUMERICAL_OPCODES)
        and certificate.get("semantics_sha256")
        == {
            opcode: _sha256(SEMANTICS[opcode])
            for opcode in sorted(EXACT_INDEX_OPCODES)
        }
        and certificate.get("unrestricted_domain_qualification_established") is False
        and certificate.get("floating_point_primitive_qualification_established") is False
    )
    reexecution_exact = True
    if reexecute:
        try:
            reexecution_exact = certificate == build_primitive_qualification_certificate()
        except (RuntimeError, TypeError, ValueError):
            reexecution_exact = False
    valid = all(
        (
            certificate_hash_valid,
            records_valid,
            claims_consistent,
            reexecution_exact,
        )
    )
    return {
        "valid": valid,
        "certificate_hash_valid": certificate_hash_valid,
        "records_valid": records_valid,
        "claims_consistent": claims_consistent,
        "reexecution_performed": reexecute,
        "reexecution_exact": reexecution_exact,
        "record_count": len(records) if isinstance(records, list) else 0,
        "case_counts": calculated_counts,
    }


def build_primitive_qualification_gate(
    program: dict[str, Any],
    certificate: dict[str, Any],
    bfloat16_certificate: dict[str, Any],
    reduction_certificate: dict[str, Any],
    reduction_backend_binding: dict[str, Any],
    nsight_suite: dict[str, Any],
    wmma_probe_certificate: dict[str, Any],
) -> dict[str, Any]:
    if not verify_gemma_ir(program)["valid"]:
        raise ValueError("Cannot qualify primitives for an invalid Gemma IR")
    qualification = verify_primitive_qualification_certificate(certificate)
    if not qualification["valid"]:
        raise ValueError("Primitive qualification certificate is invalid")
    from .gemma_float_semantics import verify_bfloat16_semantics_certificate

    bfloat16_qualification = verify_bfloat16_semantics_certificate(
        bfloat16_certificate
    )
    if not bfloat16_qualification["valid"]:
        raise ValueError("Bfloat16 semantics certificate is invalid")
    from .gemma_reduction_semantics import (
        verify_reduction_characterization_certificate,
    )

    reduction_qualification = verify_reduction_characterization_certificate(
        reduction_certificate
    )
    if not reduction_qualification["valid"]:
        raise ValueError("Reduction characterization certificate is invalid")
    from .gemma_reduction_backend import verify_gemma_reduction_backend_binding

    backend_verification = verify_gemma_reduction_backend_binding(
        program,
        reduction_certificate,
        nsight_suite,
        reduction_backend_binding,
    )
    if not backend_verification["valid"]:
        raise ValueError("Reduction backend binding is invalid")
    from .gemma_wmma_probe import verify_wmma_probe_certificate

    wmma_verification = verify_wmma_probe_certificate(
        program,
        reduction_certificate,
        nsight_suite,
        reduction_backend_binding,
        wmma_probe_certificate,
    )
    if not wmma_verification["valid"]:
        raise ValueError("WMMA probe certificate is invalid")
    reached = sorted({instruction["opcode"] for instruction in program["instructions"]})
    independently_tested = sorted(set(reached) & EXACT_INDEX_OPCODES)
    specified_bfloat16 = sorted(set(reached) & {"ADD", "MUL", "SCALE"})
    characterized_reductions = sorted(
        set(reached) & {"LINEAR", "MATMUL_QK", "MATMUL_AV"}
    )
    unresolved = sorted(set(reached) - EXACT_INDEX_OPCODES)
    body = {
        "schema_version": 1,
        "scope": "Qualification gate for primitive opcodes reached by one canonical Gemma IR; independently tested indexing/data-movement primitives remain distinct from unresolved floating-point semantics.",
        "program_sha256": program["program_sha256"],
        "primitive_certificate_sha256": certificate["certificate_sha256"],
        "bfloat16_semantics_certificate_sha256": bfloat16_certificate[
            "certificate_sha256"
        ],
        "reduction_characterization_certificate_sha256": reduction_certificate[
            "certificate_sha256"
        ],
        "reduction_backend_binding_sha256": reduction_backend_binding[
            "binding_sha256"
        ],
        "nsight_suite_sha256": nsight_suite["suite_sha256"],
        "wmma_probe_certificate_sha256": wmma_probe_certificate[
            "certificate_sha256"
        ],
        "reached_opcodes": reached,
        "independently_tested_index_opcodes": independently_tested,
        "independently_specified_finite_bfloat16_opcodes": specified_bfloat16,
        "bounded_cpu_cuda_conformant_bfloat16_opcodes": specified_bfloat16,
        "bounded_characterized_reduction_opcodes": characterized_reductions,
        "unrestricted_floating_point_opcodes": unresolved,
        "reached_opcode_count": len(reached),
        "independently_tested_opcode_count": len(independently_tested),
        "independently_specified_finite_bfloat16_opcode_count": len(
            specified_bfloat16
        ),
        "bounded_characterized_reduction_opcode_count": len(
            characterized_reductions
        ),
        "controlled_linear_roles_have_attested_kernel_identity": backend_verification[
            "all_linear_roles_have_attested_symbol_overlap"
        ],
        "canonical_eager_attention_kernel_attestation_complete": backend_verification[
            "canonical_eager_attention_symbols_attested_in_deployed_suite"
        ],
        "wmma_probe_count": wmma_verification["probe_count"],
        "wmma_probe_reexecuted_for_gate": wmma_verification[
            "reexecution_performed"
        ],
        "wmma_probe_distinct_result_count": wmma_verification[
            "distinct_result_count"
        ],
        "wmma_operand_position_invariance_established": wmma_verification[
            "operand_position_invariance_established"
        ],
        "wmma_accumulator_mapping_identified": wmma_verification[
            "wmma_accumulator_mapping_identified"
        ],
        "controlled_values_equal_recorded_model_tensors": False,
        "per_invocation_argument_binding_established": False,
        "unique_reduction_profile_identified": False,
        "reduction_order_semantics_established": False,
        "all_reached_primitive_semantics_qualified": False,
        "complete_bfloat16_binary_truth_tables_established": False,
        "bit_exact_numerical_execution_qualified": False,
        "global_exactness_activation_allowed": False,
    }
    return {**body, "gate_sha256": _sha256(body)}


def verify_primitive_qualification_gate(
    program: dict[str, Any],
    certificate: dict[str, Any],
    bfloat16_certificate: dict[str, Any],
    reduction_certificate: dict[str, Any],
    reduction_backend_binding: dict[str, Any],
    nsight_suite: dict[str, Any],
    wmma_probe_certificate: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    try:
        expected = build_primitive_qualification_gate(
            program,
            certificate,
            bfloat16_certificate,
            reduction_certificate,
            reduction_backend_binding,
            nsight_suite,
            wmma_probe_certificate,
        )
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return {"valid": False}
    gate_hash_valid = _sha256(
        {key: value for key, value in gate.items() if key != "gate_sha256"}
    ) == gate.get("gate_sha256")
    exact_match = gate == expected
    return {
        "valid": gate_hash_valid and exact_match,
        "gate_hash_valid": gate_hash_valid,
        "derived_gate_exact_match": exact_match,
        "all_reached_primitive_semantics_qualified": expected[
            "all_reached_primitive_semantics_qualified"
        ],
        "global_exactness_activation_allowed": expected[
            "global_exactness_activation_allowed"
        ],
        "independently_tested_opcode_count": expected[
            "independently_tested_opcode_count"
        ],
        "independently_specified_finite_bfloat16_opcode_count": expected[
            "independently_specified_finite_bfloat16_opcode_count"
        ],
        "bounded_characterized_reduction_opcode_count": expected[
            "bounded_characterized_reduction_opcode_count"
        ],
        "controlled_linear_roles_have_attested_kernel_identity": expected[
            "controlled_linear_roles_have_attested_kernel_identity"
        ],
        "canonical_eager_attention_kernel_attestation_complete": expected[
            "canonical_eager_attention_kernel_attestation_complete"
        ],
        "wmma_probe_count": expected["wmma_probe_count"],
        "wmma_probe_reexecuted_for_gate": expected[
            "wmma_probe_reexecuted_for_gate"
        ],
        "wmma_probe_distinct_result_count": expected[
            "wmma_probe_distinct_result_count"
        ],
        "wmma_operand_position_invariance_established": expected[
            "wmma_operand_position_invariance_established"
        ],
        "wmma_accumulator_mapping_identified": expected[
            "wmma_accumulator_mapping_identified"
        ],
        "unique_reduction_profile_identified": expected[
            "unique_reduction_profile_identified"
        ],
        "reduction_order_semantics_established": expected[
            "reduction_order_semantics_established"
        ],
        "unrestricted_floating_point_opcode_count": len(
            expected["unrestricted_floating_point_opcodes"]
        ),
    }
