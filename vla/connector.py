"""
class Idefics3Connector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = Idefics3SimpleMLP(config)

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5) # height, width = sqrt(seq)
        x = x.view(bsz, height, width, embed_dim) # reshape seq dim -> H * W
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor) # reshape -> bsz, H, W/sf, EDIM * sf
        x = x.permute(0, 2, 1, 3) # reorder -> bsz, W/sf, H, EDIM * sf
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2)) # reshape -> bsz, W/sf, H/sf, EDIM * sf^2
        x = x.permute(0, 2, 1, 3) # reorder -> bsz, H/sf, W/sf, EDIM * sf^2
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2)) # reshape seq dim -> (H * W)/sf^2
        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states

class Idefics3SimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)
"""
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import float32, int32, Stream
from allo.memory import Layout
from allo.backend.aie import ExternalModule

S = Layout.Shard
R = Layout.Replicate

BATCH = 1  # fixme: don't care for now
SEQ = 64
EMBD = 768  # 64 * 12
PATCH_SEQ_TILE = 16
SF = 2
TEXT = 512

# ===============================================================================
# Allo Version
# ===============================================================================


Ty = float32  # All tensors use float32
N = BATCH * SEQ  # 16   flattened (batch*seq)
# ----------------------------------------------

norm_io_layout = [S(0), R]

@df.region()
def pixel_shuffle_kernel(
    input_x: Ty[SEQ, EMBD],
    output_x: Ty[SEQ/(SF**2), N_EMBD*(SF**2)],
):

    @df.kernel(mapping=[PATCH_SEQ_TILE], args=[input_x, output_x])
    def shuffle (
        local_input_x: Ty[SEQ, EMBD] @ norm_io_layout,
        local_output_y: Ty[SEQ/(SF**2), N_EMBD*(SF**2)] @ norm_io_layout,
    ):
        """
        NP = PATCH_SEQ_TILE // sf
        for i in range(h_new):
            for j in range(w_new):
                for si in range(sf):
                    for sj in range(sf):
                        # Map input channel block to output
                        local_output_x[i, j, (si*sf + sj)*d//(sf*sf):(si*sf + sj + 1)*d//(sf*sf)] = \
                            local_input_x[i*sf+si, j*sf+sj, :]
                            """

    # ----------------------------------------------------------------
    # Linear
    # ----------------------------------------------------------------
    LINEAR_M, LINEAR_N, LINEAR_K = 64, 64, 64
    linear_A_layout = [S(0), R]
    linear_B_layout = [R, S(1)]
    linear_C_layout = [S(0), S(1)]

    @df.region()
    def linear_matmul_kernel(A: Ty[LINEAR_M, LINEAR_K], B: Ty[LINEAR_K, LINEAR_N], C: Ty[LINEAR_M, LINEAR_N]):
        @df.kernel(mapping=[4, 4], args=[A, B, C])
        def gemm(
            local_A: Ty[LINEAR_M, LINEAR_K] @ linear_A_layout,
            local_B: Ty[LINEAR_K, LINEAR_N] @ linear_B_layout,
            local_C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        ):
            local_C[:, :] = allo.matmul(local_A, local_B)

    @df.region()
    def linear_accumulate_kernel(A: Ty[LINEAR_M, LINEAR_N], B: Ty[LINEAR_M, LINEAR_N], C: Ty[LINEAR_M, LINEAR_N]):
        @df.kernel(mapping=[2, 4], args=[A, B, C])
        def core(
            local_A: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
            local_B: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
            local_C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        ):
            local_C[:, :] = allo.add(local_A, local_B)


# ##############################################################
# BUILD
# ##############################################################
pixel_shuffle_mod = df.build(pixel_shuffle_kernel, target="aie", project="pixel_shuffle.proj")
linear_matmul_mod = df.build(linear_matmul_kernel, target="aie", project="linear_matmul.proj")
linear_accumulate_mod = df.build(linear_accumulate_kernel, target="aie", project="linear_accumulate.proj")

# ##############################################################
# TOOL
# ##############################################################
def linear_projection(A, B, C, M, N, K):
    for i in range(M // LINEAR_M):
        for j in range(N // LINEAR_N):
            C_tmp = np.zeros((LINEAR_M, LINEAR_N)).astype(np.float32)
            for k in range(K // LINEAR_K):
                tile_A = A[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    k * LINEAR_K : (k + 1) * LINEAR_K,
                ]
                tile_B = B[
                    k * LINEAR_K : (k + 1) * LINEAR_K,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ]
                linear_matmul_mod(tile_A, tile_B, C_tmp)
                linear_accumulate_mod(
                    C[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                    C_tmp,
                    C[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                )

# Weight matrix is size [EMBD*(SF**2), TEXT]
def connector_block(x_fp32: np.ndarray, weights: np.ndarray):
    # ##############################################################
    # FORWARD
    # ##############################################################
    x = x_fp32.astype(np.float32)
    x = x.reshape(SEQ, EMBD)
    weights = weights.reshape(EMBD*(SF**2), TEXT)
    S_out = SEQ/(SF**2)
    D_out = EMBD*(SF**2)
    x_shuffled = np.zeros((S_out, D_out), dtype=np.float32)
    pixel_shuffle_mod(x, x_shuffled)
    linear_projection(x_shuffled, weights, out, S_out, TEXT, D_out)
    return out

# ##############################################################
# TEST
# ##############################################################
if __name__ == "__main__":
    x = np.random.randn(SEQ, EMBD).astype(np.float32)
    w = np.random.randn(EMBD*(SF**2), TEXT).astype(np.float32)

    out = connector_block(x, w)
    print(out.shape)