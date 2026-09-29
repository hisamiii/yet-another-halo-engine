#!/usr/bin/env python3
"""Stage the weight block in LDS for the row-group kStore/swiglu decode.

Format-generic. The row-group decode gathers the block bytes of all 64 rows from
GLOBAL memory with scattered per-element loads, and that address spread is what
makes the row-group form a loss on its own. All 64 rows use the same block for a
given 256-column block, so stage it once into LDS (one row per lane at wave64)
behind a workgroup barrier and read the decode's bytes from LDS instead.

The structure is uniform across the quant formats, which is what makes this
generic rather than per-format:

  * the block size is the constant in %bpr_i = scalar.muli %k_blocks_i, %cNNi and
    in %blk_off0 = scalar.muli %kblk, %cNNi  (98 for IQ3_XXS, 144 for IQ4_XS/Q4_K)
  * %kbase == 0 marks the first k-step of each 256-column block
  * every decode byte reaches %w_view / %w_f16_view as %blk_off + const, where
    %blk_off = %row_off + %blk_off0 carries the global row base AND the block's
    offset inside the row -- neither belongs in an LDS index, so %blk_off is
    ZEROED and only the within-block const survives
  * %row_local = index.cast %r_i is hoisted above the loads to supply the LDS row

Getting the %blk_off part wrong is NOT loud: the kernel still compiles and is
FASTER, and just reads the wrong bytes (that variant returned argmax 163749).

Anchors assume the wave64 + row-group form from tools/widen_rows.py (16 passes of
4 rows, view<64x16xf16> weight tile). Every structural anchor asserts its own
count, so a source of a different shape fails loudly.

usage: lds_stage_rows.py <src.loom> -> the transformed module on stdout
"""
import re
import sys

src = open(sys.argv[1]).read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


def subn(pat, new, n):
    global src
    src, got = re.subn(pat, new, src)
    assert got == n, (got, pat[:70])


m = re.search(r'%bpr_i = scalar\.muli %k_blocks_i, %c(\d+)i : i32', src)
assert m, 'no %bpr_i block-constant anchor'
BS = int(m.group(1))
assert 16 <= BS <= 256 and BS % 2 == 0, 'implausible block size %d' % BS
VEC = BS - (BS % 16)        # whole 16-byte vectors
TAIL = BS % 16              # scalar tail bytes (2 for 98, 0 for 144)
HALF = BS // 2
NL, NH = BS - 1, HALF - 1


def P(s):
    """Fill the @X@ placeholders. Not %-formatting: these strings are full of
    %-prefixed SSA names, and %b/%c etc. are not format specifiers."""
    for k, v in (('@BS@', BS), ('@VEC@', VEC), ('@TAIL@', TAIL), ('@HALF@', HALF),
                 ('@NL@', NL), ('@NH@', NH), ('@BYTES@', 64 * BS)):
        s = s.replace(k, str(v))
    return s


# --- 1. the decode's bounds become the LDS row extent. Before the fill is added
#        these are the only uses of %w_last / %w_half_last, and the rewrite is
#        provably a no-op: every within-block offset is < BS.
nlast = src.count(', %w_last : index')
assert nlast >= 1, 'no %w_last clamp found'
subn(r', %w_last : index', ', %%c%d : index' % NL, nlast)
nhalf = src.count(', %w_half_last : index')
if nhalf:
    subn(r', %w_half_last : index', ', %%c%d : index' % NH, nhalf)

# --- 1b. a 16-byte headroom bound for the fill's VECTOR loads. %w_last proves only
#         origin < w_bytes, which does not bound a 16-byte vector read; iq3s's
#         stager carries the same %w_lim for the same reason.
rep("  %w_last = index.sub %w_bytes, %c1 : index",
    "  %w_last = index.sub %w_bytes, %c1 : index"
    + chr(10) + "  %w_lim = index.sub %w_bytes, %c16 : index")

# --- 2. zero %blk_off: only the within-block offset belongs in an LDS index
rep("      %blk_off = scalar.addi %row_off, %blk_off0 : i32",
    "      %blk_off = scalar.addi %c0i, %c0i : i32")

# --- 3. hoist the LDS row index above the loads; drop the later duplicate
rep("      %r_i = scalar.addi %row0, %r8 : i32",
    "      %r_i = scalar.addi %row0, %r8 : i32"
    + chr(10) + "      %row_local = index.cast %r_i : i32 to index")
rep("      %h = scalar.fptrunc %value : f32 to f16"
    + chr(10) + "      %row_local = index.cast %r_i : i32 to index",
    "      %h = scalar.fptrunc %value : f32 to f16")

# --- 4. block LDS allocation beside the weight-tile LDS, then its views
# The alloca and BOTH views go in as one block, anchored on the %wstage_lds alloca
# so the alloca always precedes the views. Anchoring them separately is wrong:
# iq3xxs declares %wstage_lds_view before %ostage_view and iq4xs declares them the
# other way round, so two independent anchors put the views before their alloca in
# one of the two and the module fails to define %block_lds.
rep("  %wstage_lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer",
    P("  %wstage_lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer"
      + chr(10) + "  %block_bytes = index.constant @BYTES@ : offset"
      + chr(10) + "  %block_lds = buffer.alloca<workgroup> align(16) %block_bytes : buffer"
      + chr(10) + "  %block_view = buffer.view %block_lds[%base] : buffer -> view<64x@BS@xi8>"
      + chr(10) + "  %block_f16_view = buffer.view %block_lds[%base] : buffer -> view<64x@HALF@xf16>"))

ANCHOR = '  %wstage_lds = buffer.alloca'
# VEC // 16 is the fill's vector-loop unroll count, and it must be a declared
# index constant too -- %bpr_i's block size does not imply it.
for name, val in (('%%c%d' % NL, NL), ('%%c%d' % NH, NH), ('%%c%d' % VEC, VEC),
                  ('%%c%d' % TAIL, TAIL), ('%%c%d' % 16, 16), ('%%c%d' % 2, 2),
                  ('%%c%d' % (VEC // 16), VEC // 16)):
    if ('  %s = index.constant' % name) in src or ('      %s = index.constant' % name) in src:
        continue
    src = src.replace(ANCHOR, '  %s = index.constant %d : index' % (name, val)
                      + chr(10) + ANCHOR, 1)

# --- 5. fill the block once per 256-column block, one lane per row
TAILB = ''
if TAIL:
    TAILB = P("""      %ssink = scf.for %sv = [%c0 to %c@TAIL@ step %c1](%mks = %c0 : index) -> (index) unroll(%c@TAIL@) schedule(interleaved) {
        %soff = index.add %c@VEC@, %sv : index
        %soff_i = index.cast %soff : index to i32
        %sfull_i = scalar.addi %fbase, %soff_i : i32
        %sfull_ix = index.cast %sfull_i : i32 to index
        %sfull_lo = index.max %sfull_ix, %c0 : index
        %sfull_c = index.min %sfull_lo, %w_last : index
        %sb = view.load %w_view[%sfull_c] : view<[%w_bytes]xi8> -> i8
        view.store %sb, %block_view[%lane, %soff] : i8, view<64x@BS@xi8>
        scf.yield %mks : index
      }
""")
fill = P("""    // Stage the raw weight block (@BS@ B) of all 64 rows into LDS, once per
    // 256-column block. One lane owns one row, so the global reads are @BS@
    // contiguous bytes per lane instead of the scattered per-element gathers the
    // decode would otherwise issue for every row.
    %is_blk = scalar.cmpi eq, %kbase, %c0i : i32
    scf.if %is_blk {
      %frow_i = scalar.addi %m_origin_i, %lane_i : i32
      %frow_off = scalar.muli %frow_i, %bpr_i : i32
      %fbase = scalar.addi %frow_off, %blk_off0 : i32
      %fsink = scf.for %fv = [%c0 to %c@VEC@ step %c16](%mkf = %c0 : index) -> (index) unroll(%c@VECN@) schedule(interleaved) {
        %fv_i = index.cast %fv : index to i32
        %foff_i = scalar.addi %fbase, %fv_i : i32
        %foff_ix = index.cast %foff_i : i32 to index
        %foff_lo = index.max %foff_ix, %c0 : index
        %foff_c = index.min %foff_lo, %w_lim : index
        %fvec = vector.load %w_view[%foff_c] : view<[%w_bytes]xi8> -> vector<16xi8>
        vector.store %fvec, %block_view[%lane, %fv] : vector<16xi8>, view<64x@BS@xi8>
        scf.yield %mkf : index
      }
@TAILB@    }
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
""".replace('@TAILB@', TAILB).replace('@VECN@', str(VEC // 16)))
# --- 5. redirect every DECODE byte load to the staged block. This must happen
#        BEFORE the fill is inserted: the fill issues its own %w_view loads, they
#        live in the k-loop body outside the j loop where %row_local is defined,
#        and rewriting them yields an undefined %row_local.
#        (An earlier version also wrote '\1' in a NON-raw string, where Python
#        reads it as chr(1) rather than a group reference, so the index silently
#        disappeared and the kernels failed to parse. Use a lambda.)
src, n1 = re.subn(r'view\.load %w_view\[(%\w+)\] : view<\[%w_bytes\]xi8>',
                  lambda m: 'view.load %%block_view[%%row_local, %s] : view<64x%uxi8>'
                  % (m.group(1), BS), src)
assert n1 >= 1, 'no %w_view loads to redirect'
src, n2 = re.subn(r'view\.load %w_f16_view\[(%\w+)\] : view<\[%w_halfs\]xf16>',
                  lambda m: 'view.load %%block_f16_view[%%row_local, %s] : view<64x%uxf16>'
                  % (m.group(1), HALF), src)

# --- 6. fill the block once per 256-column block, one lane per row
rep("    %j_sink = scf.for %j = [%c0 to %c16 step %c1]",
    fill + "    %j_sink = scf.for %j = [%c0 to %c16 step %c1]")
sys.stderr.write('lds_stage_rows: block=%dB, %d byte + %d f16 loads redirected\n' % (BS, n1, n2))
sys.stdout.write(src)
