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
| One combined "command" symbol per sequence (token + offset code) | +7.1% output for a concrete 1-byte design; even ideal joint coding is +0.6% | the contextual token tables already capture more than the token/offset correlation; bucketing to 256 symbols pushes sub-class and length bits out raw |
| Two symbols per lookup for the contextual streams (tokens) | −14% | the two-symbol tables (16 groups x 2048 x 4 bytes = 128 KB) don't fit in L1/L2 |
| Fewer literal/token groups (smaller contextual tables) | 8 groups: +4.9% speed, −0.5% ratio; 4 groups: +8.2%, −0.8% | a real knob, not applied: ratio is already 2% behind Kraken |
| 10-bit contextual tables (codes limited to 10 bits; 32 KB, L1-resident) | +3.2% speed, −0.28% ratio | cache misses are only part of the contextual decoder's cost |
| Parse penalty per sequence (fewer, longer sequences) | penalty 64 (1/16 bit): +2.3% speed, −0.2% ratio; 256: +7%, −5.4% | sequence count isn't the dominant cost at these settings |

## What did help

| Change | Decode | Ratio |
|---|---|---|
| Specialised version-2 loop (no per-sequence version tests, 16 nibbles per refill) | +3% | = |
| Next Huffman group folded into the table entry (one dependent load per symbol) | +4% | = |
| Position-based extra-bit reader (the bit position is the only loop-carried value, so loads don't chain) | +1.6% | = |
| Two symbols per lookup for plain Huffman streams averaging <= 5.5 bits (offset codes, nibble pairs, lengths) | +7.2% | = |

After these: 1480 MB/s (Kraken 2037). Profile: sequence loop 53%, contextual Huffman (literals, tokens) 31%, slow
match-copy path 7%, plain Huffman 6%.

## What's left

The remaining gap to Kraken is structural: about 5.7 entropy symbols per sequence (token, offset code, half a nibble
pair, ~2.8 literals) at 0.6–1 ns each, plus ~7 ns per sequence of offset resolution and copying. A combined command symbol
and a sequence penalty (above) don't close it. What's left untried is the entropy decoder itself: literals and
tokens are ~8 M contextual-Huffman symbols at ~1 ns each, nearly half of decode time. Candidates: a table-driven
decoder that emits two symbols per lookup for the short-code streams, tANS with interleaved states, or SIMD Huffman.
Each is a substantial rewrite with an uncertain payoff.
