# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare paged decode against explicit PyTorch attention for two requests.

Run from the repository root with a CUDA-enabled PyTorch and FlashInfer:
    uv run --active --no-sync python examples/pytorch/paged_decode_tutorial.py
"""

import math

import torch

import flashinfer


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA-enabled PyTorch and GPU.")
    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16
    page_size, num_qo_heads, num_kv_heads, head_dim = 4, 4, 2, 128
    seq_lens = [6, 9]
    page_ids = [[3, 7], [1, 8, 2]]

    # 1. Construct logical, contiguous K/V independently for the reference.
    keys = [
        torch.randn(n, num_kv_heads, head_dim, device=device, dtype=dtype)
        for n in seq_lens
    ]
    values = [torch.randn_like(k) for k in keys]
    q = torch.randn(2, num_qo_heads, head_dim, device=device, dtype=dtype)

    # 2. Scatter those tokens into physical pages. NHD combines K/V on axis 1.
    # Allocate 9 slots because the largest physical page ID is 8; only 5 are used.
    # NaN padding makes accidentally reading unused slots/tokens visible.
    kv_cache = torch.full(
        (9, 2, page_size, num_kv_heads, head_dim),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    for i, pages in enumerate(page_ids):
        for logical_page, physical_page in enumerate(pages):
            start = logical_page * page_size
            count = min(page_size, seq_lens[i] - start)
            kv_cache[physical_page, 0, :count] = keys[i][start : start + count]
            kv_cache[physical_page, 1, :count] = values[i][start : start + count]

    # 3. Describe the logical-to-physical mapping using CSR-style metadata.
    indptr = torch.tensor([0, 2, 5], device=device, dtype=torch.int32)
    indices = torch.tensor([3, 7, 1, 8, 2], device=device, dtype=torch.int32)
    last_page_len = torch.tensor([2, 1], device=device, dtype=torch.int32)

    # 4. Plan once, then execute. Fix the backend to study the CUDA decode path.
    workspace = torch.zeros(128 * 1024 * 1024, device=device, dtype=torch.uint8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="fa2", use_tensor_cores=False
    )
    def decode(cache, page_indices):
        # Re-plan for each mapping; do not rely on mutation of cached metadata.
        wrapper.plan(
            indptr,
            page_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
            pos_encoding_mode="NONE",
            sm_scale=1.0 / math.sqrt(head_dim),
        )
        return wrapper.run(q, cache)

    actual = decode(kv_cache, indices)

    # 5. Reference: each query is the LAST token and sees every valid KV token,
    # including itself. No additional triangular mask is needed for this row.
    # GQA maps Q heads [0, 1] to KV head 0 and Q heads [2, 3] to KV head 1.
    expected = []
    for i in range(len(seq_lens)):
        group_size = num_qo_heads // num_kv_heads
        k = keys[i].float().repeat_interleave(group_size, dim=1)
        v = values[i].float().repeat_interleave(group_size, dim=1)
        scores = torch.einsum("hd,thd->ht", q[i].float(), k) / math.sqrt(head_dim)
        probs = torch.softmax(scores, dim=-1)
        expected.append(torch.einsum("ht,thd->hd", probs, v))
    expected = torch.stack(expected)

    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-3)
    print(f"Output shape: {tuple(actual.shape)}")
    for i in range(len(seq_lens)):
        error = (actual[i].float() - expected[i]).abs().max().item()
        print(f"Request {i}: length={seq_lens[i]}, max absolute error={error:.6g}")
    print("PASS: paged decode matches contiguous PyTorch attention.")

    # Experiment 1: exchange physical pages 3 and 1 AND update their references.
    # Both pages are full, so the final-page lengths remain unchanged.
    moved_cache = kv_cache.clone()
    moved_cache[1] = kv_cache[3]
    moved_cache[3] = kv_cache[1]
    moved_indices = torch.tensor([1, 7, 3, 8, 2], device=device, dtype=torch.int32)
    moved = decode(moved_cache, moved_indices)
    torch.testing.assert_close(moved.float(), expected, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(moved, actual, rtol=1e-2, atol=1e-3)
    print("PASS: moving KV pages together with their references preserves output.")

    # Experiment 2: use the changed references with the ORIGINAL cache.
    # A now reads B's first four KV tokens, and B reads A's first four KV tokens.
    wrong = decode(kv_cache, moved_indices)
    assert torch.isfinite(wrong).all().item(), "Expected valid but incorrect KV data"
    for i in range(len(seq_lens)):
        error = (wrong[i].float() - expected[i]).abs().max().item()
        print(f"Wrong mapping, request {i}: max absolute error={error:.6g}")
        try:
            torch.testing.assert_close(
                wrong[i].float(), expected[i], rtol=1e-2, atol=1e-3
            )
        except AssertionError:
            print(f"EXPECTED MISMATCH: request {i} reads another request's KV.")
        else:
            raise AssertionError(f"Wrong mapping unexpectedly matched request {i}")


if __name__ == "__main__":
    main()
