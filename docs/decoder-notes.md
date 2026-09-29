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

# Fast level: turbo blocks against Selkie

Same sample and machine. Oodle Selkie 6 on it: ratio 2.600, 5.9–6.4 GB/s. zap's plain format: 2.616 at about 4 GB/s.
Background load on this machine moved single runs by up to 30%, so the numbers below come from decoders timed in the
same process, interleaved; a comparison between separate runs isn't worth much here.

## Why the plain format can't get there

On identical sequences (LZ4-HC's parse re-emitted in zap's format) zap's decoder ran at 66% of LZ4's speed. It wasn't
the parse or the window. The next token's address depends on a byte loaded inside the current sequence (the far-offset
flag), which puts a load in the loop-carried chain; with a predicted branch instead, 22% far offsets mispredict about as
often. Precomputed limits and 8-byte copies were worth 4–7%, which doesn't close a 35% gap.

## What the turbo format does

One token byte per command, indexing the block's own table of its 253 most frequent (offset kind, literals, match
length) triples; offsets (2 bytes, 3 bytes, or none for "repeat the previous offset") in one stream, escape lengths and
literals in their own. Every pointer advances by an amount the token alone determines, so nothing chains through a load.

| Step | Ratio | Decode |
|---|---|---|
| Split streams, fixed 3/4-bit token split | 2.587 | slower than plain: 23% of tokens escape (literal runs peak at 10–12 on this data, 16-byte records) |
| Per-block (lit, ml) table, 127 pairs + far bit | 2.621 | 1.2x plain; escapes 7% |
| 1-byte token over (kind, lit, ml) triples, repeat offsets | 2.626 | repeat offsets are only 6% of commands but save 2 bytes each |
| Parser priced for the format (bytes + per-command, escape and far-offset penalties) | 2.611–2.631 | 1.3x plain |

## What the decode loop taught

- Counters: at the start the new loop ran at the same IPC as LZ4 (about 2.2) but retired 1.3x the instructions per
  command. It was instruction count, not stalls; the loop has no load-carried chain at all.
- Five stream pointers plus five limits made clang spill a limit per command. One bound per batch (fast commands read at
  most 16 literal and 4 offset bytes and write at most 48) removed every limit but the token pointer's.
- An instruction-pointer sampler (a thread that suspends the decoder and reads RIP) found a third of the time outside
  the fast loop: escapes and offsets < 16 left it through a call, then re-computed the batch. Handling them inline, and
  keeping offsets < 16 out of table commands (the encoder makes them escapes), took it from 0.77x to 0.95x of LZ4 on
  LZ4's own parse.
- On zap's parse 57% of samples wait on the match-source load. 18% of new offsets reach past 512 KB (L3). An oracle
  (prefetch the source recorded from an earlier decode, 16 commands ahead) was worth 9–15%. A real version, far
  matches as absolute positions in their own stream so they can be prefetched without decoding ahead, cost 10 more
  instructions per command and ended at 0.91x of the plain turbo loop. A parse penalty on offsets past 512 KB does
  some of the same job for free.
- Compilers differ: packed 64-bit table entries with a pointer output were fastest under both clang (0.93x Selkie by
  best round) and MSVC (0.96x); a struct of byte fields suited MSVC's register allocation but lost 16% under clang.
- A table command's match copy must be `memmove`: in a corrupt block the offset can be < 16, and overlapping `memcpy` is
  undefined even though it stays in bounds (ASan flags it). A fixed 16-byte `memmove` compiles to the same load and store.

# Kraken tier: entropy v3

Goal: Kraken's ratio (3.300 on the game pak sample, 4 MB blocks) at Kraken's decode speed (~2.0 GB/s in-process).
v2 (the previous entropy mode) was at 3.232 and 0.64x.

## What Kraken does (from its stream headers, no decoding)

Walking the block, quantum and 128 KB chunk headers (layout as documented by the open-source ooz decoder) gives each
stream's symbol count and compressed size. On the sample: 2.10M commands (zap: 2.06M sequences), 5.97M literals (5.82M),
1.70M new offsets (1.88M); repeat offsets on 19% of commands (zap: 9%). Literals: plain Huffman in 231 of 256 chunks,
"sub" (delta) literals in 25, and 28% of literals stored uncoded; 5.59 MB, i.e. 7.49 bits per literal, no better than
zap's contextual literals. Its lead was in commands, offsets and lengths: 4.58 MB against zap's 4.71 MB, most of it from
fewer new offsets. The parse, not the entropy coder.

On synthetic data (matches at L1/L2 distances, random literals) zap's decoder is 12-14% faster than Kraken's; the gap on
real data is contextual coding and data effects.

## What helped

| Step | Ratio | Speed vs Kraken (clang) |
|---|---|---|
| v2 | 3.232 | 0.64x |
| Byte-coded offsets (near: high byte + Huffman low byte; far: three bytes), repeats resolved in their own pass | 3.268 | 0.69x |
| Multi-arrival parse, 4 arrivals (cheapest per repeat-offset state) x 3 passes: repeats 9% -> 15% | 3.287 | |
| Offset symbol = repeat index, or 3 + the near high byte, or far: one symbol instead of two | +0.06% | |
| Parse prices tokens and literals by their coding context (arrivals carry the previous token's and literal's group) | 3.306 | |
| 8-part Huffman decoded symbol-interleaved (all chains side by side in program order), shift by the table entry, bit count from a marker bit (clz) | | 0.85x |
| Literals per 128 KB chunk: raw, plain (own table) or contextual (the block's tables), smallest wins | +0.24% | |
| Parse penalty (2 bits) on offsets past 256 KB | +0.0% | +1.4% |
| Offsets < 16 as a pshufb pattern (two stores), not a byte loop | | +1-2% |

Now: ratio 3.306 against 3.300; decode 0.87x Kraken with clang, 0.67x with MSVC (0.71x with /arch:AVX2).
Time on the sample (clang, TSC Mticks): literals 18.4, tokens 6.6, other streams 8.5, LZ 35.6 (offsets 11, copies 25).

## What didn't help

- Resolving repeat offsets inside the copy loop (Kraken's layout): 7 more live values, clang spilled the near/far
  cursors, a store-to-load chain every sequence. A separate offset pass per 512-sequence batch is 10% faster.
- Branch-free offset pass: the same 11M ticks as the branchy one (repeats mispredict, the cmov chain costs as much).
  Two passes (branch-free new offsets, then repeats found by scanning back for distinct offsets): 14-26M.
- Prefetching match sources: from the copy loop 8 sequences ahead (+7M ticks of instructions), and from the offset pass
  with exact positions (reads the length escapes): both slower. Far sources (17% of sequences past 256 KB) cost ~6.5M
  ticks, but hiding them costs more.
- Kraken-style chunked decoding (every stream resumed per 128 KB chunk so a chunk's streams stay in L2): the copy loop
  didn't get faster; 5% slower overall.
- Context for near low bytes (class of the high byte): -0.2% size but ~10M ticks (the class array needs a compaction pass).
- Literal context from two symbols back (twice the parallelism): loses 70% of the context gain. Fewer context groups
  (smaller tables): 4 groups decode 15% faster but give up 60% of the gain.
- A Kraken-like command byte (literals 0-3, match nibble, offset kind): 7% larger than token + offset symbol on zap's parse.
- Larger per-sequence parse penalties (fewer, longer sequences): no measurable speed, slow ratio loss.

## What's left

- MSVC: the 6- and 8-part contextual decoders need ~20 live registers; MSVC keeps the table pointer, the index and the
  output pointers on the stack (literals 27 vs 18 Mticks, tokens 10 vs 6.6). A lower-pressure layout is the next step.
- The copy loop: far sources and store-forwarding stalls (offsets 16-63: 10 ticks each in isolation).
- Compression speed: each 4-arrival pass costs ~2 s per 4 MB block (match finding plus pricing every length for every
  arrival); ~0.5 MB/s per thread against Kraken's 4.
