import tensorflow as tf

import keras.backend as K

# pip install tensorflow-models-official


class GLU(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.supports_masking = True

    def call(self, x):
        out, gate = tf.split(x, num_or_size_splits=2, axis=self.dim)
        return out * tf.nn.sigmoid(gate)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, mult=4, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim * mult, activation="swish"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(dim),
                tf.keras.layers.Dropout(dropout),
            ]
        )
        self.supports_masking = True

    def call(self, x):
        return self.net(x)


class ConformerConvModule(tf.keras.layers.Layer):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=17, dropout=0.0, **kwargs
    ):
        super().__init__(**kwargs)

        inner_dim = dim * expansion_factor
        padding = "causal" if causal else "same"

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(inner_dim * 2),
                GLU(2),
                tf.keras.layers.DepthwiseConv1D(kernel_size, padding=padding),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("swish"),
                tf.keras.layers.Dense(dim, use_bias=False),
                tf.keras.layers.Dropout(dropout),
            ]
        )
        self.supports_masking = True

    def call(self, x):
        # K.print_tensor(mask)
        return self.net(x)


def positional_embedding(pos_seq, inv_freq):
    sinusoid_inp = tf.einsum("i,j->ij", pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    return pos_emb[:, None, :]


def rel_shift(x):
    x_size = tf.shape(x)

    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
    x = tf.reshape(x, [x_size[0], x_size[1], x_size[3] + 1, x_size[2]])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)

    return x


def create_positional_embeddings(max_len, dim):
    pos_seq = tf.range(max_len - 1, -1, -1.0)
    inv_freq = 1 / (10000 ** (tf.range(0, dim, 2.0) / dim))
    pos_emb = positional_embedding(pos_seq, inv_freq)
    return pos_emb


class MultiHeadRelativeSelfAttention(tf.keras.layers.Layer):
    def __init__(self, max_len, dim, dim_head, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.max_len = max_len
        self.supports_masking = True
        self.positional_embeddings = create_positional_embeddings(max_len, dim)
        self.qkv = tf.keras.layers.Dense(units=3 * num_heads * dim_head, use_bias=False, name="qkv")
        self.r_head = tf.keras.layers.Dense(units=num_heads * dim_head, use_bias=False, name="r")
        self.project = tf.keras.layers.Dense(dim, use_bias=False, name="o")

    def call(self, inputs):
        bsz = tf.shape(inputs)[0]
        x = tf.transpose(inputs, (1, 0, 2))  # Time first

        scale = 1 / (self.dim_head**0.5)
        w_heads = self.qkv(x)
        r_head_k = self.r_head(self.positional_embeddings)

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-self.max_len :]

        w_head_q = tf.reshape(w_head_q, [self.max_len, bsz, self.num_heads, self.dim_head])
        w_head_k = tf.reshape(w_head_k, [self.max_len, bsz, self.num_heads, self.dim_head])
        w_head_v = tf.reshape(w_head_v, [self.max_len, bsz, self.num_heads, self.dim_head])

        r_head_k = tf.reshape(r_head_k, [self.max_len, self.num_heads, self.dim_head])

        AC = tf.einsum("ibnd,jbnd->bnij", w_head_q, w_head_k)  # query attending to key
        BD = tf.einsum(
            "ibnd,jnd->bnij", w_head_q, r_head_k
        )  # query attending to positional encoding
        BD = rel_shift(BD)  # relative shift trick (appendix B of Transformer XL paper)

        attn_score = (
            AC + BD
        ) * scale  # tensor with shape Batch, num_headss,Time (query),Time (key)

        if hasattr(inputs, "_keras_mask"):
            mask = inputs._keras_mask
            if mask is not None:
                att_mask = tf.cast(mask[:, None, None, :], "float32")
                attn_score = attn_score * att_mask - 1e30 * (1 - att_mask)

        attn_prob = tf.nn.softmax(attn_score, 3)

        attn_vec = tf.einsum("bnij,jbnd->bind", attn_prob, w_head_v)  # Batch,Time, Heads,D
        # size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [bsz, self.max_len, self.num_heads * self.dim_head])

        attn_out = self.project(attn_vec)

        x = attn_out + inputs  # residual connection
        return x


def AttentionBlock(max_len, dim, dim_head, num_heads, dropout=0.0):
    def apply(inputs):
        x = tf.keras.layers.LayerNormalization()(inputs)
        x = MultiHeadRelativeSelfAttention(max_len, dim, dim_head, num_heads)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        return x

    return apply


# Conformer Block


def ConformerBlock(
    max_len,
    dim,
    dim_head=32,
    num_heads=4,
    ff_mult=4,
    conv_expansion_factor=2,
    conv_kernel_size=17,
    dropout=0.0,
    conv_causal=False,
):
    def apply(inputs):
        ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

        ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

        attn = AttentionBlock(
            max_len, dim=dim, dim_head=dim_head, num_heads=num_heads, dropout=dropout
        )
        conv = ConformerConvModule(
            dim=dim,
            kernel_size=conv_kernel_size,
            causal=conv_causal,
            expansion_factor=conv_expansion_factor,
            dropout=dropout,
        )

        x = inputs + 0.5 * ff1(inputs)
        x = x + attn(x)
        x = x + conv(x)
        x = x + 0.5 * ff2(x)
        x = tf.keras.layers.LayerNormalization()(x)
        return x

    return apply
