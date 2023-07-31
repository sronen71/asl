import tensorflow as tf


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
        outputs = self.net(x)
        return outputs


class ConformerConvModule(tf.keras.layers.Layer):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=17, dropout=0.0, **kwargs
    ):
        super().__init__(**kwargs)

        inner_dim = dim * expansion_factor
        padding = "causal" if causal else "same"

        self.ln = tf.keras.layers.LayerNormalization()
        self.dense1 = tf.keras.layers.Dense(inner_dim * 2)
        self.glu = GLU(2)
        self.conv1d = tf.keras.layers.DepthwiseConv1D(kernel_size, padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.swish = tf.keras.layers.Activation("swish")
        self.dense2 = tf.keras.layers.Dense(dim, use_bias=False)
        self.do = tf.keras.layers.Dropout(dropout)

        self.supports_masking = True

    def call(self, inputs, mask=None):
        assert mask is not None
        x = self.ln(inputs)
        x = self.dense1(x)
        x = self.glu(x)
        x = self.conv1d(x)
        x = self.bn(x, mask=mask)
        x = self.swish(x)
        x = self.dense2(x)
        x = self.do(x)

        return x


class MultiHeadRelativeSelfAttention(tf.keras.layers.Layer):
    def __init__(self, max_len, dim, dim_head, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.max_len = max_len
        self.positional_embeddings = self.create_positional_embeddings(max_len, dim)
        self.qkv = tf.keras.layers.Dense(units=3 * num_heads * dim_head, use_bias=False, name="qkv")
        self.r_head = tf.keras.layers.Dense(units=num_heads * dim_head, use_bias=False, name="r")
        self.project = tf.keras.layers.Dense(dim, use_bias=False, name="o")
        self.supports_masking = True

    def positional_embedding(self, pos_seq, inv_freq):
        sinusoid_inp = tf.einsum("i,j->ij", pos_seq, inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
        return pos_emb[:, None, :]

    def rel_shift(self, x):
        x_size = tf.shape(x)

        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, [x_size[0], x_size[1], x_size[3] + 1, x_size[2]])
        x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, x_size)

        return x

    def create_positional_embeddings(self, max_len, dim):
        pos_seq = tf.range(max_len - 1, -1, -1.0)
        inv_freq = 1 / (10000 ** (tf.range(0, dim, 2.0) / dim))
        pos_emb = self.positional_embedding(pos_seq, inv_freq)
        return pos_emb

    def call(self, inputs, mask=None):
        assert mask is not None

        bsz = tf.shape(inputs)[0]
        x = tf.transpose(inputs, (1, 0, 2))  # Change from Batch first to Time first

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
        BD = self.rel_shift(BD)  # relative shift trick (appendix B of Transformer XL paper)

        attn_score = (
            AC + BD
        ) * scale  # tensor with shape Batch, num_headss,Time (query),Time (key)

        if mask is not None:
            att_mask = tf.cast(mask[:, None, None, :], "float32")
            attn_score = attn_score * att_mask - 1e30 * (1 - att_mask)

        attn_prob = tf.nn.softmax(attn_score, 3)

        attn_vec = tf.einsum("bnij,jbnd->bind", attn_prob, w_head_v)  # Batch,Time, Heads,D
        attn_vec = tf.reshape(attn_vec, [bsz, self.max_len, self.num_heads * self.dim_head])

        attn_out = self.project(attn_vec)

        x = attn_out + inputs  # residual connection
        return x


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, max_len, dim, dim_head, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.ln = tf.keras.layers.LayerNormalization()
        self.MHRA = MultiHeadRelativeSelfAttention(max_len, dim, dim_head, num_heads)
        self.do = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        x = self.ln(inputs)
        x = self.MHRA(x, mask=mask)
        x = self.do(x)
        return x


# Conformer Block


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        max_len,
        dim,
        dim_head=32,
        num_heads=4,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=17,
        dropout=0.0,
        conv_causal=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)
        self.attn = AttentionBlock(
            max_len, dim=dim, dim_head=dim_head, num_heads=num_heads, dropout=dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            kernel_size=conv_kernel_size,
            causal=conv_causal,
            expansion_factor=conv_expansion_factor,
            dropout=dropout,
        )
        self.ln = tf.keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        x = inputs + 0.5 * self.ff1(inputs)
        x = x + self.attn(x, mask=mask)
        x = x + self.conv(x, mask=mask)
        x = x + 0.5 * self.ff2(x)
        x = self.ln(x)

        return x
