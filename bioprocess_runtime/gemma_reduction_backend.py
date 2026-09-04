from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

from .gemma_ir import verify_gemma_ir
from .gemma_reduction_semantics import (
    _candidate_results,
    verify_reduction_characterization_certificate,
)
from .nsight_attestation import verify_nsight_kernel_suite
from .serialization import canonical_json

try:
    import torch
    from torch.nn import functional
    from torch.profiler import ProfilerActivity, profile
except ImportError:
    torch = None
    functional = None
    ProfilerActivity = None
    profile = None


def _require_cuda() -> None:
    if (
        torch is None
        or functional is None
        or profile is None
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Gemma-shape reduction profiling requires CUDA PyTorch")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _bfloat16_vector(bits: list[int]) -> Any:
    return torch.tensor(bits, dtype=torch.uint16).view(torch.bfloat16).cuda()


def _result_bits(value: Any) -> str:
    return f"0x{int(value.detach().cpu().reshape(1).view(torch.uint16).item()):04x}"


VECTOR_CLASSES = ("cancellation", "grouped", "pseudo")


def _vector_bits(length: int, vector_class: str) -> list[int]:
    repetitions = (length + 3) // 4
    if vector_class == "cancellation":
        return ([0x4E80, 0x3F80, 0xCE80, 0x3F80] * repetitions)[:length]
    if vector_class == "grouped":
        return ([0x4E80, 0xCE80, 0x3F80, 0x3F80] * repetitions)[:length]
    if vector_class != "pseudo":
        raise ValueError("Unknown reduction vector class")
    state = 0x2468ACE1 ^ length
    values = []
    for _ in range(length):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        sign = 0x8000 if state & 1 else 0
        exponent = 120 + ((state >> 8) % 15)
        fraction = (state >> 16) & 0x7F
        values.append(sign | (exponent << 7) | fraction)
    return values


def _operand_bits(
    primitive: str, length: int, vector_class: str
) -> tuple[list[int], list[int]]:
    vector = _vector_bits(length, vector_class)
    if primitive == "MATMUL_AV":
        return [0x3D00] * length, vector
    return vector, [0x3F80] * length


def _profile_call(call: Callable[[], Any]) -> tuple[Any, list[str]]:
    call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
        output = call()
        torch.cuda.synchronize()
    names = []
    for event in captured.events():
        if event.device_type == torch.autograd.DeviceType.CUDA and event.name not in names:
            names.append(event.name)
    if not names:
        raise RuntimeError("No CUDA kernel was observed for controlled reduction")
    return output, names


def _linear_record(
    role: str,
    roles: list[str],
    rows: int,
    outputs: int,
    inner: int,
    vector_class: str,
    suite_names: set[str],
) -> dict[str, Any]:
    left_bits, right_bits = _operand_bits("LINEAR", inner, vector_class)
    left_vector = _bfloat16_vector(left_bits)
    right_vector = _bfloat16_vector(right_bits)
    input_tensor = left_vector.reshape(1, inner).expand(rows, inner).contiguous()
    weight = right_vector.reshape(1, inner).expand(outputs, inner).contiguous()
    result, kernel_names = _profile_call(lambda: functional.linear(input_tensor, weight))
    candidates = _candidate_results(left_bits, right_bits)
    actual = _result_bits(result[0, 0])
    body = {
        "role": role,
        "vector_class": vector_class,
        "equivalent_roles": roles,
        "primitive": "LINEAR",
        "input_shape": [rows, inner],
        "weight_shape": [outputs, inner],
        "output_shape": [rows, outputs],
        "inner_dimension": inner,
        "left_vector_sha256": _sha256(left_bits),
        "right_vector_sha256": _sha256(right_bits),
        "actual_result_bits": actual,
        "candidate_results": candidates,
        "matching_candidates": sorted(
            name for name, candidate in candidates.items() if candidate == actual
        ),
        "cuda_kernel_names": kernel_names,
        "exact_nsight_suite_symbol_matches": sorted(set(kernel_names) & suite_names),
    }
    return {**body, "record_sha256": _sha256(body)}


def _attention_record(
    primitive: str,
    heads: int,
    sequence: int,
    dimension: int,
    vector_class: str,
    suite_names: set[str],
) -> dict[str, Any]:
    inner = dimension if primitive == "MATMUL_QK" else sequence
    left_bits, right_bits = _operand_bits(primitive, inner, vector_class)
    left_vector = _bfloat16_vector(left_bits)
    right_vector = _bfloat16_vector(right_bits)
    if primitive == "MATMUL_QK":
        query = left_vector.reshape(1, 1, 1, inner).expand(
            1, heads, sequence, inner
        ).contiguous()
        key = right_vector.reshape(1, 1, 1, inner).expand(
            1, heads, sequence, inner
        ).contiguous()
        result, kernel_names = _profile_call(
            lambda: torch.matmul(query, key.transpose(2, 3))
        )
        input_shapes = [[1, heads, sequence, inner], [1, heads, sequence, inner]]
        output_shape = [1, heads, sequence, sequence]
    else:
        left_matrix = left_vector.reshape(1, 1, 1, inner).expand(
            1, heads, sequence, inner
        ).contiguous()
        value = right_vector.reshape(1, 1, inner, 1).expand(
            1, heads, inner, dimension
        ).contiguous()
        result, kernel_names = _profile_call(
            lambda: torch.matmul(left_matrix, value)
        )
        input_shapes = [[1, heads, sequence, inner], [1, heads, inner, dimension]]
        output_shape = [1, heads, sequence, dimension]
    candidates = _candidate_results(left_bits, right_bits)
    actual = _result_bits(result[0, 0, 0, 0])
    body = {
        "role": "explicit_eager_attention_scores" if primitive == "MATMUL_QK" else "explicit_eager_attention_values",
        "vector_class": vector_class,
        "equivalent_roles": [],
        "primitive": primitive,
        "left_operand_constraint": (
            {
                "kind": "uniform nonnegative bfloat16",
                "value_bits": "0x3d00",
                "exact_row_sum": {"numerator": sequence, "denominator": 32},
            }
            if primitive == "MATMUL_AV"
            else {"kind": "finite cancellation-sensitive bfloat16"}
        ),
        "input_shapes": input_shapes,
        "output_shape": output_shape,
        "inner_dimension": inner,
        "left_vector_sha256": _sha256(left_bits),
        "right_vector_sha256": _sha256(right_bits),
        "actual_result_bits": actual,
        "candidate_results": candidates,
        "matching_candidates": sorted(
            name for name, candidate in candidates.items() if candidate == actual
        ),
        "cuda_kernel_names": kernel_names,
        "exact_nsight_suite_symbol_matches": sorted(set(kernel_names) & suite_names),
    }
    return {**body, "record_sha256": _sha256(body)}


def build_gemma_reduction_backend_binding(
    program: dict[str, Any],
    reduction_certificate: dict[str, Any],
    nsight_suite: dict[str, Any],
    sequence_length: int = 30,
    max_tensor_elements: int = 200_000_000,
) -> dict[str, Any]:
    _require_cuda()
    if not verify_gemma_ir(program)["valid"]:
        raise ValueError("Gemma IR is invalid")
    if not verify_reduction_characterization_certificate(reduction_certificate)["valid"]:
        raise ValueError("Reduction characterization certificate is invalid")
    suite_verification = verify_nsight_kernel_suite(nsight_suite)
    if not suite_verification["valid"]:
        raise ValueError("Nsight kernel suite is invalid")
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length <= 0:
        raise ValueError("Sequence length must be a positive integer")
    if (
        not isinstance(max_tensor_elements, int)
        or isinstance(max_tensor_elements, bool)
        or max_tensor_elements <= 0
    ):
        raise ValueError("Maximum tensor elements must be a positive integer")
    configuration = program["configuration"]
    hidden = configuration["hidden_size"]
    intermediate = configuration["intermediate_size"]
    heads = configuration["attention_heads"]
    kv_heads = configuration["key_value_heads"]
    dimension = configuration["head_dimension"]
    vocabulary = configuration["vocabulary_size"]
    suite_names = {entry["kernel_name"] for entry in nsight_suite["entries"]}
    profiles = (
        ("query_projection", ["query_projection"], sequence_length, heads * dimension, hidden),
        ("key_value_projection", ["key_projection", "value_projection"], sequence_length, kv_heads * dimension, hidden),
        ("attention_output_projection", ["attention_output_projection"], sequence_length, hidden, heads * dimension),
        ("mlp_gate_up_projection", ["mlp_gate_projection", "mlp_up_projection"], sequence_length, intermediate, hidden),
        ("mlp_down_projection", ["mlp_down_projection"], sequence_length, hidden, intermediate),
        ("vocabulary_projection", ["vocabulary_projection"], 1, vocabulary, hidden),
    )
    largest_linear_tensor = max(
        max(rows * inner, outputs * inner, rows * outputs)
        for _, _, rows, outputs, inner in profiles
    )
    largest_attention_tensor = max(
        heads * sequence_length * dimension,
        heads * sequence_length * sequence_length,
    )
    largest_profiled_tensor = max(
        largest_linear_tensor, largest_attention_tensor
    )
    if largest_profiled_tensor > max_tensor_elements:
        raise ValueError(
            "Controlled reduction profile exceeds the configured tensor-element limit"
        )
    records = [
        _linear_record(
            role,
            roles,
            rows,
            outputs,
            inner,
            vector_class,
            suite_names,
        )
        for role, roles, rows, outputs, inner in profiles
        for vector_class in VECTOR_CLASSES
    ]
    records.extend(
        _attention_record(
            primitive,
            heads,
            sequence_length,
            dimension,
            vector_class,
            suite_names,
        )
        for primitive in ("MATMUL_QK", "MATMUL_AV")
        for vector_class in VECTOR_CLASSES
    )
    exact_overlap_records = sum(
        bool(record["exact_nsight_suite_symbol_matches"]) for record in records
    )
    role_candidate_intersections = {}
    for role in sorted({record["role"] for record in records}):
        role_records = [record for record in records if record["role"] == role]
        common = set(role_records[0]["matching_candidates"])
        for record in role_records[1:]:
            common &= set(record["matching_candidates"])
        role_candidate_intersections[role] = sorted(common)
    body = {
        "schema_version": 1,
        "scope": "Controlled Gemma-shape CUDA reduction-kernel identities and candidate outputs bound to one typed IR and Nsight suite; controlled values are not the recorded model tensors and symbol overlap is not reduction-semantic proof.",
        "program_sha256": program["program_sha256"],
        "reduction_characterization_certificate_sha256": reduction_certificate[
            "certificate_sha256"
        ],
        "nsight_suite_sha256": nsight_suite["suite_sha256"],
        "batch_size": 1,
        "sequence_length": sequence_length,
        "vector_classes": list(VECTOR_CLASSES),
        "max_tensor_elements": max_tensor_elements,
        "largest_profiled_tensor_elements": largest_profiled_tensor,
        "largest_profiled_tensor_bytes_bfloat16": largest_profiled_tensor * 2,
        "records": records,
        "record_count": len(records),
        "exact_nsight_symbol_overlap_record_count": exact_overlap_records,
        "role_candidate_intersections": role_candidate_intersections,
        "roles_with_unique_stable_candidate": sorted(
            role
            for role, candidates in role_candidate_intersections.items()
            if len(candidates) == 1
        ),
        "all_linear_roles_have_attested_symbol_overlap": all(
            record["exact_nsight_suite_symbol_matches"]
            for record in records
            if record["primitive"] == "LINEAR"
        ),
        "canonical_eager_attention_symbols_attested_in_deployed_suite": all(
            record["exact_nsight_suite_symbol_matches"]
            for record in records
            if record["primitive"] in {"MATMUL_QK", "MATMUL_AV"}
        ),
        "controlled_values_equal_recorded_model_tensors": False,
        "per_invocation_argument_binding_established": False,
        "reduction_order_semantics_established": False,
        "tensor_core_accumulator_semantics_established": False,
        "hardware_instruction_semantics_established": False,
    }
    return {**body, "binding_sha256": _sha256(body)}


def verify_gemma_reduction_backend_binding(
    program: dict[str, Any],
    reduction_certificate: dict[str, Any],
    nsight_suite: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(binding, dict):
        return {"valid": False}
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    try:
        binding_hash_valid = _sha256(body) == binding.get("binding_sha256")
    except (TypeError, ValueError):
        return {"valid": False}
    records = binding.get("records")
    suite_names = {
        entry.get("kernel_name")
        for entry in nsight_suite.get("entries", [])
        if isinstance(entry, dict)
    }
    expected_records: dict[tuple[str, str], dict[str, Any]] = {}
    heads = 0
    dimension = 0
    sequence_length = binding.get("sequence_length")
    configuration = program.get("configuration")
    if (
        isinstance(sequence_length, int)
        and not isinstance(sequence_length, bool)
        and sequence_length > 0
        and isinstance(configuration, dict)
    ):
        hidden = configuration.get("hidden_size")
        intermediate = configuration.get("intermediate_size")
        heads = configuration.get("attention_heads")
        kv_heads = configuration.get("key_value_heads")
        dimension = configuration.get("head_dimension")
        vocabulary = configuration.get("vocabulary_size")
        dimensions = (hidden, intermediate, heads, kv_heads, dimension, vocabulary)
        if all(isinstance(value, int) and value > 0 for value in dimensions):
            linear_profiles = (
                ("query_projection", ["query_projection"], sequence_length, heads * dimension, hidden),
                ("key_value_projection", ["key_projection", "value_projection"], sequence_length, kv_heads * dimension, hidden),
                ("attention_output_projection", ["attention_output_projection"], sequence_length, hidden, heads * dimension),
                ("mlp_gate_up_projection", ["mlp_gate_projection", "mlp_up_projection"], sequence_length, intermediate, hidden),
                ("mlp_down_projection", ["mlp_down_projection"], sequence_length, hidden, intermediate),
                ("vocabulary_projection", ["vocabulary_projection"], 1, vocabulary, hidden),
            )
            expected_records = {
                (role, vector_class): {
                    "role": role,
                    "vector_class": vector_class,
                    "equivalent_roles": roles,
                    "primitive": "LINEAR",
                    "inner_dimension": inner,
                    "input_shape": [rows, inner],
                    "weight_shape": [outputs, inner],
                    "output_shape": [rows, outputs],
                }
                for role, roles, rows, outputs, inner in linear_profiles
                for vector_class in VECTOR_CLASSES
            }
            attention_profiles = {
                "explicit_eager_attention_scores": {
                    "equivalent_roles": [],
                    "primitive": "MATMUL_QK",
                    "left_operand_constraint": {
                        "kind": "finite cancellation-sensitive bfloat16"
                    },
                    "inner_dimension": dimension,
                    "input_shapes": [
                        [1, heads, sequence_length, dimension],
                        [1, heads, sequence_length, dimension],
                    ],
                    "output_shape": [1, heads, sequence_length, sequence_length],
                },
                "explicit_eager_attention_values": {
                    "equivalent_roles": [],
                    "primitive": "MATMUL_AV",
                    "left_operand_constraint": {
                        "kind": "uniform nonnegative bfloat16",
                        "value_bits": "0x3d00",
                        "exact_row_sum": {
                            "numerator": sequence_length,
                            "denominator": 32,
                        },
                    },
                    "inner_dimension": sequence_length,
                    "input_shapes": [
                        [1, heads, sequence_length, sequence_length],
                        [1, heads, sequence_length, dimension],
                    ],
                    "output_shape": [1, heads, sequence_length, dimension],
                },
            }
            expected_records.update(
                {
                    (role, vector_class): {
                        "role": role,
                        "vector_class": vector_class,
                        **declaration,
                    }
                    for role, declaration in attention_profiles.items()
                    for vector_class in VECTOR_CLASSES
                }
            )

    def record_valid(record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        try:
            record_body = {
                key: value for key, value in record.items() if key != "record_sha256"
            }
            kernels = record.get("cuda_kernel_names")
            candidates = record.get("candidate_results")
            inner = record.get("inner_dimension")
            expected = expected_records.get(
                (record.get("role"), record.get("vector_class"))
            )
            if (
                not isinstance(inner, int)
                or isinstance(inner, bool)
                or inner <= 0
                or expected is None
            ):
                return False
            left_bits, right_bits = _operand_bits(
                record.get("primitive"), inner, record.get("vector_class")
            )
            expected_candidates = _candidate_results(left_bits, right_bits)
            expected_fields_match = all(
                record.get(key) == value for key, value in expected.items()
            )
            return bool(
                expected_fields_match
                and re.fullmatch(r"0x[0-9a-f]{4}", str(record.get("actual_result_bits", "")))
                is not None
                and isinstance(kernels, list)
                and kernels
                and all(isinstance(name, str) for name in kernels)
                and candidates == expected_candidates
                and record.get("left_vector_sha256") == _sha256(left_bits)
                and record.get("right_vector_sha256") == _sha256(right_bits)
                and record.get("matching_candidates")
                == sorted(
                    name
                    for name, result in candidates.items()
                    if result == record.get("actual_result_bits")
                )
                and record.get("exact_nsight_suite_symbol_matches")
                == sorted(set(kernels) & suite_names)
                and record.get("record_sha256") == _sha256(record_body)
            )
        except (TypeError, ValueError, AttributeError):
            return False

    records_valid = bool(
        isinstance(records, list)
        and len(records) == len(expected_records) == 24
        and all(isinstance(record, dict) for record in records)
        and {
            (record.get("role"), record.get("vector_class"))
            for record in records
        }
        == set(expected_records)
        and all(record_valid(record) for record in records)
    )
    overlap_count = (
        sum(bool(record.get("exact_nsight_suite_symbol_matches")) for record in records)
        if isinstance(records, list) and all(isinstance(record, dict) for record in records)
        else -1
    )
    linear_overlap = bool(
        isinstance(records, list)
        and all(
            record.get("exact_nsight_suite_symbol_matches")
            for record in records
            if isinstance(record, dict) and record.get("primitive") == "LINEAR"
        )
    )
    attention_overlap = bool(
        isinstance(records, list)
        and all(
            record.get("exact_nsight_suite_symbol_matches")
            for record in records
            if isinstance(record, dict)
            and record.get("primitive") in {"MATMUL_QK", "MATMUL_AV"}
        )
    )
    calculated_role_intersections = {}
    if isinstance(records, list) and all(isinstance(record, dict) for record in records):
        for role in sorted({record.get("role") for record in records}):
            role_records = [record for record in records if record.get("role") == role]
            common = set(role_records[0].get("matching_candidates", []))
            for record in role_records[1:]:
                common &= set(record.get("matching_candidates", []))
            calculated_role_intersections[role] = sorted(common)
    calculated_unique_roles = sorted(
        role
        for role, candidates in calculated_role_intersections.items()
        if len(candidates) == 1
    )
    largest_expected_tensor = (
        max(
            max(
                shape[0] * shape[1]
                for shape in (
                    expected["input_shape"],
                    expected["weight_shape"],
                    expected["output_shape"],
                )
            )
            for expected in expected_records.values()
            if expected["primitive"] == "LINEAR"
        )
        if expected_records
        else -1
    )
    if largest_expected_tensor > 0:
        largest_expected_tensor = max(
            largest_expected_tensor,
            heads * sequence_length * dimension,
            heads * sequence_length * sequence_length,
        )
    source_bindings_valid = bool(
        verify_gemma_ir(program)["valid"]
        and verify_reduction_characterization_certificate(
            reduction_certificate, reexecute=False
        )["valid"]
        and verify_nsight_kernel_suite(nsight_suite)["valid"]
        and binding.get("program_sha256") == program.get("program_sha256")
        and binding.get("reduction_characterization_certificate_sha256")
        == reduction_certificate.get("certificate_sha256")
        and binding.get("nsight_suite_sha256") == nsight_suite.get("suite_sha256")
    )
    claims_consistent = bool(
        binding.get("schema_version") == 1
        and binding.get("batch_size") == 1
        and isinstance(sequence_length, int)
        and not isinstance(sequence_length, bool)
        and sequence_length > 0
        and binding.get("vector_classes") == list(VECTOR_CLASSES)
        and binding.get("role_candidate_intersections")
        == calculated_role_intersections
        and binding.get("roles_with_unique_stable_candidate")
        == calculated_unique_roles
        and isinstance(binding.get("max_tensor_elements"), int)
        and not isinstance(binding.get("max_tensor_elements"), bool)
        and binding.get("max_tensor_elements") >= largest_expected_tensor > 0
        and binding.get("largest_profiled_tensor_elements")
        == largest_expected_tensor
        and binding.get("largest_profiled_tensor_bytes_bfloat16")
        == largest_expected_tensor * 2
        and binding.get("record_count")
        == (len(records) if isinstance(records, list) else -1)
        and binding.get("exact_nsight_symbol_overlap_record_count") == overlap_count
        and binding.get("all_linear_roles_have_attested_symbol_overlap")
        is linear_overlap
        and binding.get("canonical_eager_attention_symbols_attested_in_deployed_suite")
        is attention_overlap
        and binding.get("controlled_values_equal_recorded_model_tensors") is False
        and binding.get("per_invocation_argument_binding_established") is False
        and binding.get("reduction_order_semantics_established") is False
        and binding.get("tensor_core_accumulator_semantics_established") is False
        and binding.get("hardware_instruction_semantics_established") is False
    )
    valid = all(
        (
            binding_hash_valid,
            records_valid,
            source_bindings_valid,
            claims_consistent,
        )
    )
    return {
        "valid": valid,
        "binding_hash_valid": binding_hash_valid,
        "records_valid": records_valid,
        "source_bindings_valid": source_bindings_valid,
        "claims_consistent": claims_consistent,
        "record_count": len(records) if isinstance(records, list) else 0,
        "exact_nsight_symbol_overlap_record_count": overlap_count,
        "all_linear_roles_have_attested_symbol_overlap": linear_overlap,
        "canonical_eager_attention_symbols_attested_in_deployed_suite": attention_overlap,
        "reduction_order_semantics_established": binding.get(
            "reduction_order_semantics_established"
        ),
    }
