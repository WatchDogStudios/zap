# Entropy-mode decoder: measurements and dead ends

Notes from profiling zap's entropy decoder against Oodle, so the experiments below aren't repeated blind.

**Setup.** A 32 MB sample of a Source-engine VPK (Portal Revolution `pak01_010.vpk`, 8 evenly spaced 4 MB pieces), 4 MB
blocks, one thread, clang 18 `-O3 -march=native`, Ryzen 7 5800X (Zen 3) under Windows 11 with Hyper-V. Timings are the
median of 8 interleaved runs on an otherwise idle machine; with a game running, even hardware cycle counts varied by
±20% (the SMT sibling shares the core), so numbers taken under load are not comparable.

## Where the time goes (zap 1.3-dev, entropy hc64, ratio 3.232)

| Part | ms per 32 MB | per unit |
|---|---|---|
| Huffman streams | ~11 | 0.6 ns/symbol plain, ~1 ns/symbol contextual; ~12 M symbols |
| Offset resolution | 6.6 | 3.2 ns/sequence (2.1 M sequences) |
| Copies | 8.2 | 4.0 ns/sequence |
| **Oodle Kraken, whole decode** | **16.5** | ratio 3.300 |

## What didn't help

| Idea | Result | Why |
|---|---|---|
| Two-phase decode (resolve all offsets into an array, then copy) | −2.3% | the offset array and a padded copy of the extra bits cost more than the tighter loops save |
| Prefetching match sources ahead in the copy phase | ~0 | out-of-order execution already overlaps most of the far-offset latency (cache-hot offsets are only 30% faster) |
| Branch-free offset resolution (masks, cmov) | −5% to +5% | the loop is latency-bound, not misprediction-bound; the compiler turned ternaries into branches anyway |
| 8 interleaved Huffman sub-streams instead of 4 | −25% | register spills |
| Bit readers copied to locals (address-taken structs) | 0 | the compiler already kept them in registers |
| Raw (uncompressed) literals | +38% speed, −3.1% ratio | still slower and larger than Kraken |

## What did help

| Change | Decode | Ratio |
|---|---|---|
| Specialised version-2 loop (no per-sequence version tests, 16 nibbles per refill) | +3% | = |
| Next Huffman group folded into the table entry (one dependent load per symbol) | +4% | = |
| Position-based extra-bit reader (the bit position is the only loop-carried value, so loads don't chain) | +1.6% | = |

## What's left

The remaining gap to Kraken is structural: about 5.7 entropy symbols per sequence (token, offset code, half a nibble
pair, ~2.8 literals) at 0.6–1 ns each, plus ~7 ns per sequence of offset resolution and copying. Candidates, none tried yet:
fewer symbols per sequence (a combined command symbol for token and offset class, as Oodle's formats use), a faster
entropy decoder for the contextual streams, and parse-level trade-offs that produce fewer, longer sequences at a
controlled ratio cost.
