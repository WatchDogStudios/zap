# How Oodle Kraken works, and where zap stands

Kraken is the last Oodle tier zap has to beat. This is what it does, measured against zap's entropy v3, and what to
do next. Kraken's format and encoder are known here from [ooz](https://github.com/powzix/ooz) (an open-source
re-implementation; file and function names below are ooz's), with design background from Charles Bloom's and Fabian
Giesen's blogs. Real Oodle 2.8 numbers come from the game-pak sample in [decoder-notes.md](decoder-notes.md).

## Kraken in one page

**Container.** Output is cut into 256 KB quanta (3-byte header: compressed size, flags; special cases for stored and
memset quanta), each into 128 KB chunks. A chunk is an LZ chunk (mode 0: delta literals, mode 1: raw literals), or
just one entropy-coded byte array. Matches can reach any earlier byte of the stream; the encoder works in up to 4 MB
local rounds plus a long-range matcher for older data. **Every chunk has its own entropy tables.**

**Streams of an LZ chunk** (`Kraken_ReadLzTable`), each an independently coded byte array:

1. literals (raw, or `lit - dst[rep0]` "sub" literals for the whole chunk);
2. commands, one byte each: bits 0-1 literal run 0-2 (3 = from the length stream), bits 2-5 match length 2-16
   (15 = from the length stream), bits 6-7 offset index: 0-2 = one of 3 recent offsets (move to front), 3 = new;
3. new offsets, one byte each: a log2 class with the low nibble (legacy mode) or with 3 mantissa bits (scaled mode),
   plus an optional second array of low parts for a per-chunk stride ("offset scaling", like zap's S);
4. lengths, one byte each (255 escapes to the bitstream);
5. one raw bitstream holding offset and length extra bits, read from both ends by two readers.

**Entropy coding per array** (`Kraken_DecodeBytes`): raw, Huffman in 3 streams or 6 (max code length 11, table
headers of a few dozen bytes), tANS with 5 interleaved states, RLE, or "multi-array" / recursive coding (an array
stitched from up to 63 separately coded arrays: several tables within one array). There is **no context modelling**:
every table is order 0. The encoder picks each array's coder by size + lambda x modelled decode time.

**Decoder.** Two phases per 128 KB chunk: decode every array into scratch and unpack offsets / lengths into int32
arrays, then a copy loop with no entropy decoding in it. 8- and 16-byte unconditional over-copies; the encoder never
emits offsets below 8, so there is no short-offset path. Repeat offsets resolve with a branch-free move-to-front.

**Encoder** (`KrakenOptimal`, levels 5-8). Statistics are seeded by lazy parses that try the repeat offsets first
(a new offset must be 2-3 bytes longer to replace one), smoothed (`count/16 + 1`), and turned into fractional bit
costs; they are updated inside the pass every window. The parse keeps 1 state per position (2 at level 8), tries
repeat matches from 2 bytes at every predecessor, new matches only from the cheapest, and looks ahead for "match,
1-2 literals, rep0". New matches start at 4 bytes (3 at level 7+), longer at long distances. One optimal pass per
128 KB quantum; the result is kept only if it codes smaller.

## Measured: zap entropy v3 against Kraken

Linux binaries (32 MB, 4 MB blocks) and one 4 MB block of them, ooz Kraken level 6 vs zap depth 64:

| | zap v3 | Kraken 6 |
|---|---|---|
| corpus ratio | 3.343 | 3.513 |
| block: literals | 856,271 (632,516 B coded, 5.9 bits) | 573,582 (518,692 B, 7.2 bits) |
| block: sequences | 317,957, 29% repeats | 395,356, 40% repeats (29K of 2 bytes) |
| block: total | 1,353,530 | 1,273,573 |

**It is the coder, not the parse.** Kraken's exact parse (395,356 commands taken from ooz's decoder) coded by zap's
v3 coder is 1,357,489 bytes (with 2-byte matches allowed) - no better than zap's own parse. On identical commands:

| | Kraken 6 | zap v3 | |
|---|---|---|---|
| literals | 518,692 | 528,340 | +1.9% |
| commands, lengths, offsets | 754,667 | 828,997 | +9.8% |

- New offset values cost the same in both (order-0 entropy 16.2 bits per offset under zap's byte coding, Kraken's
  log2 classes, or any bucket scheme).
- The difference is the command: Kraken codes literal run, match length and repeat index as one symbol, with
  tables per 128 KB. zap codes the repeat-vs-new choice in the offset symbol stream (57 KB of the block's entropy;
  30.6 KB if it were conditioned on the token), and its token and offset tables span the 4 MB block (per-128 KB
  tables: about -30 KB on Kraken's parse, but only -7 KB on zap's own, which has adapted to zap's coder).
- On zap's parse the Kraken-style command byte (with a context like zap's tokens) is 1% smaller than zap's token +
  offset symbol on binaries and 4% larger on textures: it suits repeat-heavy executables, not long literal runs.

**Tried this round** (depth 64 unless noted):

| Change | Ratio | Decode | Status |
|---|---|---|---|
| Offset symbol coded with tables by the token's group (16) | +0.76% mixed, +0.88% binaries | -10% | not taken |
| ... by the token's literal-run class (4 tables) | +0.39%, +0.60% | -8% | not taken |
| Repeat matches from 2 bytes (match nibble from 2) | -0.3% to +0.05% | | not taken |
| Literals priced 1 bit dearer in early passes | +0.15% to +0.8% | -1.5% to -5% | not taken |
| Smoothed fractional costs (Kraken-style) | -1.0% binaries, +0.5-0.7% others | | not taken |
| **x86 branch filter, per block by trial** | **+7.6% binaries**, 0 elsewhere | -10% on filtered blocks | **shipped** |

The contextual decoders cost ~6 ticks per symbol against ~2.6 for zap's plain 8-part decoder, which is why the
context experiments lose decode speed faster than they gain ratio.

**The x86 filter** (E8/E9 rel32 -> block-relative absolute, `zap_x86_filter`) is not part of Kraken. Frames apply it
to entropy blocks at depth >= 32 when a fast entropy trial of both versions finds the filtered one >0.5% smaller. On
the binaries it takes zap from 3.343 to 3.596: past Kraken 6 (3.513), Kraken 8 (3.581) and level with Leviathan
(3.595). Texture and other blocks are never filtered (their trial loses). Un-filtering runs at ~3 GB/s.

## Round 2: regional tables

Per 4 MB block of the mixed sample, ooz Kraken 6 was 3-6% smaller on the BC texture blocks. The parse-swap test
again put it on the coding: Kraken resets its tables every 128 KB, zap's token, length and offset streams kept one
set per block. Shipped:

| Change | Mixed sample | Binaries (no filter) | Decode |
|---|---|---|---|
| before | 3.841 | 3.343 | |
| compact table headers (deflate-style code lengths; 20-60 bytes instead of 128 per table) | +0.1-0.3% per block | | |
| regional streams: sub-streams per 256 KB of output, own tables, split only where smaller | 3.925 | 3.378 | -16% |
| table building by doubling, from decoded symbol lists; 64-bit header reads | | | -4.7% net |
| parse priced per region | **3.942** | **3.382** | |

Per block against Kraken 6, the texture blocks went from -3..-6% to +1.1%, +0.7%, -0.4%. With the x86 filter the
binaries reach 3.636 through frames.

Tried and dropped: the offset stage with the 3 recent offsets in one SSE register (one pshufb per sequence, no
branches): 3-7% slower than the scalar code. `ZAP_FAST_DECODE` changes little on these samples (their literal chunks
are already raw or plain).

**What's left on the mixed sample** is block 7, part of the Go binary `containerd`: Kraken 6 1,927,632 against zap's
2,011,423. Kraken's parse is 59% repeat matches (107K of them 2 bytes); on that parse zap's token + offset symbol
cost 53 KB more than Kraken's command + offset symbol, because the repeat index is a separate symbol. That is item 2
below.

## Where to go next

Ranked by expected value for "zap over Kraken, open source":

1. **Decode speed on game data** (0.88x Kraken on the pak sample; the one axis Kraken still wins there).
   - Per-128 KB two-phase decode so a chunk's streams stay in L2 (zap's streams span 4 MB of scratch).
   - Minimum offset 8 (the encoder never emits 1-7) to drop the short-offset path from the copy loop.
   - Plain (order-0) per-128 KB tables instead of contextual block tables where ratio allows: the plain 8-part
     decoder is ~2.3x faster per symbol than the contextual ones.
2. **A Kraken-style command per block** (format, binaries): literal run 0-2 / match length / repeat index in one
   symbol for repeat-heavy blocks, zap's token for the rest; the parse prices whichever the block uses. Rep
   sequences then decode one symbol instead of two. Expected +1-2% on executables (entropy estimate; needs a
   prototype), and faster decode on them.
3. ~~Compact table headers + per-region tables~~ (done, round 2: +2.6% on the mixed sample).
4. **Multi-table arrays / tANS** for skewed streams (delta literals near zero, commands): ratio, at some decode cost.
5. **Matches across blocks** (Kraken's window is the whole stream): frames keep blocks independent for parallel
   decode, so this would be an option for single-threaded streams only.
6. More filters beyond x86: ARM64 BL/ADRP, and stride-delta for vertex / index buffers.
