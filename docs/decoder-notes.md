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

Kraken's levels on the sample (4 MB blocks, one thread): 5: 3.222 at 5.8 MB/s, 6: 3.300 at 3.5 MB/s, 7: 3.342 at
2.9 MB/s, 8: 3.357 at 1.6 MB/s; all decode at 1.5-1.8 GB/s. Level 8 against 6, by the same header walk: the same
5.97M literals but 178 KB fewer literal bytes (split "multi-array" streams in 51 literal chunks against 13, more delta
literals), slightly more commands. Its option minMatchLen is 3 at level 8; forcing 4 costs it 1.0% on a mixed DXT
block and 2.9% on a grey RGB image block. spaceSpeedTradeoffBytes (256) costs it ~0.7%. Per block, zap was 3.5% ahead
of level 6 on one DXT5 texture and 3-4% behind on mixed DXT1/DXT5 blocks and the RGB block, where 96% of new offsets
are multiples of 3 (the pixel) - invisible to byte-wise offset coding.

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
| 3-byte matches: token match nibble counts from 3; the parse tries the nearest earlier 3 bytes (3-byte hash) below 256 KB, repeat matches from 3 bytes | +0.33% | 0 |
| Offset scale per block: offsets divisible by S (the encoder's pick of 1..64) coded as off / S, the rest in their own symbol range | +0.34% (RGB block -4.9%) | 0 |
| (both) | 3.329 | |
| Matches found once per block into a table (every position in order, tree depth 128) instead of per pass; the first parse is lazy over the table. The per-pass search found nothing at positions a window revisits | 3.345 | 0 |
| A 2^20-bucket 3-byte hash (2^16 lost 3-byte matches to collisions) | 3.347 | |
| Forced long matches (256+) taken whole, and the cheapest path's long repeats too, not in 256-byte pieces | 3.348 | |
| Literal chunk modes priced by decode time (Huffman must save ~1% over raw, contextual 0.5% more) | 3.349 | +2% |
| Delta literals: a chunk may code literal - the byte at the last match's offset (Kraken's "sub" literals), with a second contextual table set; the parse prices literals by region as they'll be coded | 3.362 | -1.5% |

Later: stack rows instead of per-part output pointers (MSVC spilled them), a table layout per ISA (length-low where
shrx shifts by the entry, symbol-low without BMI2), BMI2 / LZCNT / SSSE3 code picked by CPUID in builds that don't
target them, and no library memcpy for short literal runs on the slow path.

Now (in-process, interleaved rounds against Kraken's 3.300 at ~2060 MB/s):

| Build | Default: ratio 3.306 | ZAP_FAST_DECODE: ratio 3.285 |
|---|---|---|
| clang -march=native | ~1820 MB/s, 0.88x | ~2020 MB/s, 0.98x |
| MSVC default | ~0.78x | |

ZAP_FAST_DECODE keeps a 128 KB literal chunk raw unless Huffman saves more than 4% (Kraken stores 28% of its literals
raw). The threshold is the ratio/speed dial: 0% 3.306 / 67.4 Mticks, 3% 3.294 / 63.1, 4% 3.285 / 61.2, 7% 3.259 / 58.6,
10% 3.221 / 54.9. Smaller chunks (64 KB, 32 KB) lose ratio at every threshold. Kraken sits at about 3.300 / 61.7:
just outside the curve (0.93x at its ratio, -0.6% at its speed).

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
- Repeat matches of 2-3 bytes with the match nibble counting from 2 for repeats: -0.26% (the token alphabet mixes two
  meanings; short repeats are rare here).
- Delta ("sub") literals against the byte at the last offset, per 128 KB chunk: on zap's parse ~0 (4 KB); against a
  fixed stride (3, 4, 8, 16) ~0.2%, most of it one mixed DXT block.
- Order-0 literal tables switched per segment (k-means, 4-32 tables): 1% larger than the 16 contextual tables.
- Stream tables per region instead of per 4 MB block: +0.15% at 1 MB regions (+2.3% on the mixed block), lost on the
  others to table costs.
- Extreme parse effort (16 arrivals, 6 passes, 4x match depth) on a mixed DXT block: -0.26%. The parse isn't the gap.

## What's left

- MSVC: the 6- and 8-part contextual decoders need ~20 live registers; MSVC keeps the table pointer, the index and the
  output pointers on the stack (literals 27 vs 18 Mticks, tokens 10 vs 6.6). A lower-pressure layout is the next step.
- The copy loop: far sources and store-forwarding stalls (offsets 16-63: 10 ticks each in isolation).
- Compression speed at the top: depth 64 (arrivals 4, 4, 4) is 3.362 at ~0.8 MB/s per thread (Kraken level 8: 3.357
  at 1.8 MB/s).
- Decode: 0.88x Kraken level 6 on the pak sample, 0.78x on the mixed one. Kraken codes its commands as one symbol
  (zap: token + offset symbol, and contextual tokens: 1.6% of the ratio), decodes literals plain or raw (zap mostly
  contextual: 2.5 against 1.6 ticks a literal), and stores incompressible quanta; its parse trades ~0.7% of ratio
  for decode speed (spaceSpeedTradeoffBytes 256; at 1 it reaches 3.385 on the pak sample, 22% slower).

## A second sample: mixed game data

The pak sample is DXT textures. A 32 MB mix of 4 MB pieces (a UE4 pak's BC7 textures, the game's executable, a
terrain heightmap, a nav mesh, Lua text) told a different story before this round: zap 2.097 against Kraken level
6's 2.135, and 0.58x its decode speed. Kraken stores near-incompressible 256 KB quanta raw (memcpy speed) where zap
Huffman-coded literals for a 0.04% gain, codes every chunk of the heightmap, nav mesh and executable as delta
literals, and zap cut long matches into 256-byte pieces. After the fixes above: 2.192 (Kraken 6: 2.135, 7: 2.172,
8: 2.197), decode 0.78x (textures still ~3x slower than Kraken's stored quanta: the literals are copied twice).

## Compression speed

One thread, same process, best of interleaved rounds (MB/s, game pak sample):

| | ratio | MB/s |
|---|---|---|
| Kraken level 5 | 3.222 | 7.18 |
| Kraken level 6 | 3.300 | 4.54 |
| zap depth 32 (arrivals 1, 1) | 3.313 | 4.66 |
| zap depth 48 (1, 2) | 3.327 | 3.28 |
| Kraken level 7 | 3.342 | 3.46 |
| zap depth 64 (1, 2, 2, 4) | 3.366 | ~1.0 |
| Kraken level 8 | 3.357 | 1.79 |
| zap depth 128 (1, 2, 4, 8) | 3.372 | ~0.5 |

(Ratios after delta literals and the schedules below; the speeds of depth 32 and 48 are from before delta literals,
which cost the parse ~5%.) Arrivals per pass, ratio on the pak sample / the mixed sample: 1, 1: 3.324 / 2.150; 1, 2:
3.343 / 2.172; 2, 2, 4: 3.361 / 2.192; 4, 4, 4: 3.362 / 2.194; 1, 2, 2, 4: 3.366 / 2.193; 4, 8, 8: 3.369 / 2.201;
1, 2, 4, 8: 3.372 / 2.200. More passes pay more than more arrivals.

What got it there (from ~0.5 MB/s at depth 64):
- Matches found once per block into a table instead of once per pass. The tree walks are chains of cache misses
  (7 steps per position on average, 16 on the RGB block); walk length barely depends on the depth limit, so the
  table searches at depth 128 for the same time as 32.
- The table's own 2^20 tree roots instead of the hc state's 2^17: shorter walks, 10-25% faster build.
- The first pass exists for its statistics only: it's a lazy parse over the table, then one sampled pass (every
  other 64 KB, -0.03%). The old first parse (a byte-priced optimal parse) was both slower and a worse seed.
- The length loops: token prices looked up once per arrival (a row by match length), the window extended once per
  match, an early exit on the repeat loop too, single-arrival inserts without the repeat-state check: -20% per pass.

Tried: hash chains for the table (worse ratio, and quadratic on long runs); lower tree "nice" lengths (walks end on
depth or leaves, not length); only the cheapest arrival trying new offsets (-0.4%); matches at interior lengths of
long matches skipped (-0.1%); a copy of the parse per arrival count (noise).

# 1.3.0 release audit

## Fixed

- **v3 encoder wrote undecodable blocks.** `zap__k_lits` chose each literal chunk's mode against table sets built
  from the previous round's contextual chunks, and priced a symbol missing from them at 64 bits instead of ruling the
  mode out. A chunk that turned contextual in the last round could carry a (group, symbol) pair with no code; the
  encoder wrote nothing for it. Seen on one 4 MB block of Linux binaries at depth 64 (the game samples never hit it).
- **Fuzzing** (libFuzzer + ASan/UBSan, ~2M executions per build: runtime CPUID, `-march=native`, and forced non-BMI2
  decoders; every v3/turbo/frame header field at boundary values): no out-of-bounds access. Turbo table commands
  accepted offsets 0-15, so a corrupt block could decode "successfully" with bytes dst held before. Now refused. v3's
  far-offset array was aligned relative to the scratch pointer (misaligned with unaligned scratch). Now absolute.
- `zap__hcodes` reversed codes one bit at a time: 2.2% of v3 decode instructions (~50 tables per 4 MB block). Now a
  branch-free reverse: about +4% decode on mixed data, same output.
- Frames with `ZAP_ENTROPY` at depth >= 32 ran the plain optimal parse before v3's own (15-18% of compression time).
  It now runs only as a fallback and for blocks under 64 KB: 16% faster `zap c -e`, same output.

## Against Kraken on executables (ooz)

ooz (an open-source Kraken implementation; its Huffman length limiter needed a local fix for counts past 64K to
compress 4 MB blocks) on 32 MB of Linux binaries: Kraken 6 3.513, Kraken 8 3.581 against zap depth 64's 3.343
(zstd 19: 3.341). Decoding, zap is 1.5x ooz's decoder. One 4 MB block, by stream:

| | zap depth 64 | Kraken 6 | Kraken 8 |
|---|---|---|---|
| total | 1,353,530 | 1,273,573 | 1,244,902 |
| literals | 856,271 -> 632,516 B (5.9 bits) | 573,582 -> 518,692 B (7.2) | 513,492 -> 452,788 B (7.1) |
| sequences | 317,957, 29% repeats | 395,356, 40% repeats | 426,218 |
| everything else | 721 KB | 755 KB | 792 KB |

Kraken covers ~280K more bytes with matches: twice zap's 4-byte new matches (59K vs 33K), and repeats of 2 bytes
(29K; zap can't code them) and 3 bytes (26K vs 10K). zap codes each literal and each sequence cheaper. Its passes
re-price from the previous parse, and cheap contextual literals keep the parse literal-heavy.

Tried on that block and others (depth 64 unless noted):
- Repeat matches from 2 bytes (match nibble counting from 2): -0.3% to +0.05%. Not the missing piece by itself.
- No penalties (sequence 1.5 bits, far offsets 2 bits, stored regions), depth 32: +1.4% on the binaries block; the
  sequence penalty is most of it (+0.8%); store never triggered there.
- Tree depth 2048 instead of 128: identical parse.
- Literals priced 1 bit dearer in every pass but the last: binaries +0.8%, mixed +0.15%, textures +0.5%; decode 1.5-5%
  slower. With the sequence penalty halved as well: +1.1% / +0.3% / +0.6%, decode 4-7% slower. Neither shipped: they
  are a point further along the same ratio/decode dial, and still 4% short of Kraken 6 on binaries.
- Costlier arrivals trying repeat matches at full length only: 3-14% faster compression, -0.2% to +0.1% ratio.

What's left for executables is the parse's equilibrium, not the coder: a parse that starts match-heavy and holds it
(Kraken-like cheap short repeats, an x86 call/jump filter, or a literal cost that tracks the literals the parse will
leave rather than the ones it left last pass).
