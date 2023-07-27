import math
import numpy as np
import tensorflow as tf

# Code From Squeezeformer repo


def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        head_size,
        output_size=None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
        **kwargs,
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.dropout = tf.keras.layers.Dropout(dropout, name="dropout")
        self._droput_rate = dropout

    def build(self, input_shape):
        # num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        output_size = self.output_size if self.output_size is not None else num_value_features
        input_max = (self.num_heads * self.head_size) ** -0.5
        self.query = tf.keras.layers.Dense(
            self.num_heads * self.head_size,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-input_max, maxval=input_max
            ),
            bias_initializer=tf.keras.initializers.RandomUniform(
                minval=-input_max, maxval=input_max
            ),
        )
        self.key = tf.keras.layers.Dense(
            self.num_heads * self.head_size,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-input_max, maxval=input_max
            ),
            bias_initializer=tf.keras.initializers.RandomUniform(
                minval=-input_max, maxval=input_max
            ),
        )
        self.value = tf.keras.layers.Dense(
            self.num_heads * self.head_size,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-input_max, maxval=input_max
            ),
            bias_initializer=tf.keras.initializers.RandomUniform(
                minval=-input_max, maxval=input_max
            ),
        )
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=tf.keras.initializers.RandomUniform(minval=-input_max, maxval=input_max),
        )
        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=[output_size],
                initializer=tf.keras.initializers.RandomUniform(
                    minval=-input_max, maxval=input_max
                ),
            )
        else:
            self.projection_bias = None

    def call_qkv(self, query, key, value, training=False):
        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to "
                "the same as the number of elements in 'value'"
            )
        # Linear transformations
        query = self.query(query)
        B, T, E = shape_list(query)
        query = tf.reshape(query, [B, T, self.num_heads, self.head_size])

        key = self.key(key)
        B, T, E = shape_list(key)
        key = tf.reshape(key, [B, T, self.num_heads, self.head_size])

        value = self.value(value)
        B, T, E = shape_list(value)
        value = tf.reshape(value, [B, T, self.num_heads, self.head_size])

        return query, key, value

    def call_attention(self, query, key, value, logits, training=False, mask=None):
        # mask = attention mask with shape [B, Tquery, Tkey] with 1 is for positions we want to attend, 0 for masked
        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have at least 2 dimensions")
            if query.shape[-3] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to "
                    "the number of elements in 'query'"
                )
            if key.shape[-3] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )
        # apply mask
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = tf.expand_dims(mask, -3)

            logits += -10e9 * (1.0 - mask)

        attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum("...NHI,HIO->...NO", multihead_output, self.projection_kernel)

        if self.projection_bias is not None:
            output += self.projection_bias

        return output, attn_coef

    def call(self, inputs, training=False, mask=None, **kwargs):
        query, key, value = inputs

        query, key, value = self.call_qkv(query, key, value, training=training)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=tf.float32)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        output, attn_coef = self.call_attention(
            query, key, value, logits, training=training, mask=mask
        )

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def compute_output_shape(self, input_shape):
        num_value_features = input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]
        output_size = self.output_size if self.output_size is not None else num_value_features

        output_shape = input_shape[0][:-1] + (output_size,)

        if self.return_attn_coef:
            num_query_elements = input_shape[0][-2]
            num_key_elements = input_shape[1][-2]
            attn_coef_shape = input_shape[0][:-2] + (
                self.num_heads,
                num_query_elements,
                num_key_elements,
            )

            return output_shape, attn_coef_shape
        else:
            return output_shape

    def get_config(self):
        config = super().get_config()

        config.update(
            head_size=self.head_size,
            num_heads=self.num_heads,
            output_size=self.output_size,
            dropout=self._droput_rate,
            use_projection_bias=self.use_projection_bias,
            return_attn_coef=self.return_attn_coef,
        )

        return config


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def __init__(self, kernel_sizes=None, strides=None, **kwargs):
        super(RelPositionMultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        num_pos_features = input_shape[-1][-1]
        input_max = (self.num_heads * self.head_size) ** -0.5
        self.pos_kernel = self.add_weight(
            name="pos_kernel",
            shape=[self.num_heads, num_pos_features, self.head_size],
            initializer=tf.keras.initializers.RandomUniform(minval=-input_max, maxval=input_max),
        )
        self.pos_bias_u = self.add_weight(
            name="pos_bias_u",
            shape=[self.num_heads, self.head_size],
            initializer=tf.keras.initializers.Zeros(),
        )
        self.pos_bias_v = self.add_weight(
            name="pos_bias_v",
            shape=[self.num_heads, self.head_size],
            initializer=tf.keras.initializers.Zeros(),
        )
        super(RelPositionMultiHeadAttention, self).build(input_shape[:-1])

    @staticmethod
    def relative_shift(x):
        x_shape = tf.shape(x)
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])
        x = tf.reshape(x[:, :, 1:, :], x_shape)
        return x

    def call(self, inputs, training=False, mask=None, **kwargs):
        query, key, value, pos = inputs

        query, key, value = self.call_qkv(query, key, value, training=training)

        pos = tf.einsum("...MI,HIO->...MHO", pos, self.pos_kernel)

        query_with_u = query + self.pos_bias_u
        query_with_v = query + self.pos_bias_v

        logits_with_u = tf.einsum("...NHO,...MHO->...HNM", query_with_u, key)
        logits_with_v = tf.einsum("...NHO,...MHO->...HNM", query_with_v, pos)
        logits_with_v = self.relative_shift(logits_with_v)

        logits = logits_with_u + logits_with_v[:, :, :, : tf.shape(logits_with_u)[3]]

        depth = tf.constant(self.head_size, dtype=tf.float32)
        logits /= tf.sqrt(depth)

        output, attn_coef = self.call_attention(
            query, key, value, logits, training=training, mask=mask
        )

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output


class FFModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        dropout=0.0,
        ff_expansion_rate=4,
        name="ff_module",
        **kwargs,
    ):
        super(FFModule, self).__init__(name=name, **kwargs)

        self.scale = tf.Variable([1.0] * input_dim, trainable=True, name=f"{name}_scale")
        self.bias = tf.Variable([0.0] * input_dim, trainable=True, name=f"{name}_bias")
        ffn1_max = input_dim**-0.5
        ffn2_max = (ff_expansion_rate * input_dim) ** -0.5
        self.ffn1 = tf.keras.layers.Dense(
            ff_expansion_rate * input_dim,
            name=f"{name}_dense_1",
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-ffn1_max, maxval=ffn1_max
            ),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-ffn1_max, maxval=ffn1_max),
        )
        self.act = tf.keras.layers.Activation(tf.nn.swish, name=f"{name}_act")
        self.do1 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            input_dim,
            name=f"{name}_dense_2",
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-ffn2_max, maxval=ffn2_max
            ),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-ffn2_max, maxval=ffn2_max),
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_2")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, training=False, **kwargs):
        scale = tf.reshape(self.scale, (1, 1, -1))
        bias = tf.reshape(self.bias, (1, 1, -1))
        outputs = inputs * scale + bias
        outputs = self.ffn1(outputs, training=training)
        outputs = self.act(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs


class MHSAModule(tf.keras.layers.Layer):
    def __init__(
        self,
        head_size,
        num_heads,
        dropout=0.0,
        mha_type="relmha",
        name="mhsa_module",
        **kwargs,
    ):
        super(MHSAModule, self).__init__(name=name, **kwargs)

        input_dim = num_heads * head_size
        self.scale = tf.Variable([1.0] * input_dim, trainable=True, name=f"{name}_scale")
        self.bias = tf.Variable([0.0] * input_dim, trainable=True, name=f"{name}_bias")

        if mha_type == "relmha":
            self.mha = RelPositionMultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size,
                num_heads=num_heads,
            )
        elif mha_type == "mha":
            self.mha = MultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size,
                num_heads=num_heads,
            )
        else:
            raise ValueError("mha_type must be either 'mha' or 'relmha'")
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")
        self.mha_type = mha_type

    def call(self, inputs, training=False, mask=None, pos=False, **kwargs):
        if pos is False:
            inputs, pos = inputs  # pos is positional encoding

        scale = tf.reshape(self.scale, (1, 1, -1))
        bias = tf.reshape(self.bias, (1, 1, -1))
        outputs = inputs * scale + bias
        if self.mha_type == "relmha":
            outputs = self.mha([outputs, outputs, outputs, pos], training=training, mask=mask)
        else:
            outputs = outputs + pos
            outputs = self.mha([outputs, outputs, outputs], training=training, mask=mask)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs


class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        kernel_size=31,
        dropout=0.0,
        depth_multiplier=1,
        conv_expansion_rate=2,
        name="conv_module",
        **kwargs,
    ):
        super(ConvModule, self).__init__(name=name, **kwargs)

        self.scale = tf.Variable([1.0] * input_dim, trainable=True, name=f"{name}_scale")
        self.bias = tf.Variable([0.0] * input_dim, trainable=True, name=f"{name}_bias")
        pw1_max = input_dim**-0.5
        dw_max = kernel_size**-0.5
        pw2_max = input_dim**-0.5
        self.pw_conv_1 = tf.keras.layers.Conv2D(
            filters=conv_expansion_rate * input_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name=f"{name}_pw_conv_1",
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-pw1_max, maxval=pw1_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-pw1_max, maxval=pw1_max),
        )
        self.act1 = tf.keras.layers.Activation(tf.nn.swish, name=f"{name}_act_1")
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, 1),
            strides=1,
            padding="same",
            name=f"{name}_dw_conv",
            depth_multiplier=depth_multiplier,
            depthwise_initializer=tf.keras.initializers.RandomUniform(
                minval=-dw_max, maxval=dw_max
            ),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-dw_max, maxval=dw_max),
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name=f"{name}_bn", momentum=0.985, synchronized=True
        )
        self.act2 = tf.keras.layers.Activation(tf.nn.swish, name=f"{name}_act_2")
        self.pw_conv_2 = tf.keras.layers.Conv2D(
            filters=input_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name=f"{name}_pw_conv_2",
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-pw2_max, maxval=pw2_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-pw2_max, maxval=pw2_max),
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, training=False, pad_mask=None, **kwargs):
        scale = tf.reshape(self.scale, (1, 1, -1))
        bias = tf.reshape(self.bias, (1, 1, -1))
        outputs = inputs * scale + bias
        B, T, E = shape_list(outputs)
        outputs = tf.reshape(outputs, [B, T, 1, E])
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.act1(outputs)
        pad_mask = tf.expand_dims(tf.expand_dims(pad_mask, -1), -1)
        outputs = outputs * tf.cast(pad_mask, "float32")
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.act2(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = tf.reshape(outputs, [B, T, E])
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs


class MHSAFFModule(tf.keras.layers.Layer):
    """
    Wrapper class for a MHSA layer followed by a FF layer
    """

    def __init__(
        self,
        input_dim,
        head_size,
        num_heads,
        dropout=0.0,
        mha_type="relmha",
        ff_expansion_rate=4,
        name="mhsaff_module",
        **kwargs,
    ):
        super(MHSAFFModule, self).__init__(name=name, **kwargs)
        assert input_dim == head_size * num_heads
        self.mhsa = MHSAModule(
            mha_type=mha_type,
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            name=f"{name}_mhsa",
        )
        self.ln_mid = tf.keras.layers.LayerNormalization(name=f"{name}_ln_mid")
        self.ff = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            ff_expansion_rate=ff_expansion_rate,
            name=f"{name}_ff",
        )
        self.ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln")

    def call(self, inputs, training=False, *args, **kwargs):
        outputs = self.mhsa(inputs, training=training, *args, **kwargs)
        outputs = self.ln_mid(outputs, training=training)
        outputs = self.ff(outputs, training=training, *args, **kwargs)
        outputs = self.ln(outputs, training=training)
        return outputs


class ConvFFModule(tf.keras.layers.Layer):
    """
    Wrapper class for a Conv layer followed by a FF layer
    """

    def __init__(
        self,
        input_dim,
        kernel_size=31,
        dropout=0.0,
        conv_expansion_rate=2,
        ff_expansion_rate=4,
        name="convff_module",
        **kwargs,
    ):
        super(ConvFFModule, self).__init__(name=name, **kwargs)
        self.conv = ConvModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            conv_expansion_rate=conv_expansion_rate,
            dropout=dropout,
            name=f"{name}_conv",
        )
        self.ln_mid = tf.keras.layers.LayerNormalization(name=f"{name}_ln_mid")
        self.ff = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            ff_expansion_rate=ff_expansion_rate,
            name=f"{name}_ff",
        )
        self.ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln")

    def call(self, inputs, training=False, *args, **kwargs):
        outputs = self.conv(inputs, training=training, *args, **kwargs)
        outputs = self.ln_mid(outputs, training=training)
        outputs = self.ff(outputs, training=training, *args, **kwargs)
        outputs = self.ln(outputs, training=training)
        return outputs


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        dropout=0.0,
        head_size=36,
        num_heads=4,  # num_heads*head_size should equal to input_dim
        mha_type="relmha",
        kernel_size=31,
        name="conformer_block",
        **kwargs,
    ):
        assert input_dim == num_heads * head_size
        super(ConformerBlock, self).__init__(name=name, **kwargs)

        def get_fixed_arch(arch_type, name):
            if arch_type == "f":
                return FFModule(
                    input_dim=input_dim,
                    dropout=dropout,
                    name=name,
                )
            elif arch_type == "m":
                return MHSAModule(
                    mha_type=mha_type,
                    head_size=head_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    name=name,
                )
            elif arch_type == "c":
                return ConvModule(
                    input_dim=input_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    name=name,
                )
            elif arch_type == "M":
                return MHSAFFModule(
                    mha_type=mha_type,
                    head_size=head_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    input_dim=input_dim,
                    name=name,
                )
            elif arch_type == "C":
                return ConvFFModule(
                    input_dim=input_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    name=name,
                )

            raise ValueError(f"fised architecture type '{arch_type}' is not defined")

        # Layer 1: MHSA+FF
        self.layer1 = get_fixed_arch("M", name + "_layer1")
        # Layer 2: CONV+FF
        self.layer2 = get_fixed_arch("C", name + "_layer2")

    def call(self, inputs, training=False, mask=None, pad_mask=None, **kwargs):
        inputs, pos = inputs  # pos is positional encoding
        outputs = self.layer1(
            inputs, training=training, mask=mask, pos=pos, pad_mask=pad_mask, **kwargs
        )
        outputs = self.layer2(
            outputs, training=training, mask=mask, pos=pos, pad_mask=pad_mask, **kwargs
        )
        return outputs


class TimeReductionLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size=5,
        stride=2,
        dropout=0.0,
        name="time_reduction",
        **kwargs,
    ):
        super(TimeReductionLayer, self).__init__(name=name, **kwargs)
        self.stride = stride
        self.kernel_size = kernel_size
        dw_max = kernel_size**-0.5
        pw_max = input_dim**-0.5
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, 1),
            strides=self.stride,
            padding="valid",
            name=f"{name}_dw_conv",
            depth_multiplier=1,
            depthwise_initializer=tf.keras.initializers.RandomUniform(
                minval=-dw_max, maxval=dw_max
            ),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-dw_max, maxval=dw_max),
        )
        # self.swish = tf.keras.layers.Activation(tf.nn.swish, name=f"{name}_swish_activation")
        self.pw_conv = tf.keras.layers.Conv2D(
            filters=output_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name=f"{name}_pw_conv_2",
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-pw_max, maxval=pw_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-pw_max, maxval=pw_max),
        )

    def call(self, inputs, training=False, mask=None, pad_mask=None, **kwargs):
        B, T, E = shape_list(inputs)
        outputs = tf.reshape(inputs, [B, T, 1, E])
        _pad_mask = tf.expand_dims(tf.expand_dims(pad_mask, -1), -1)
        outputs = outputs * tf.cast(_pad_mask, "float32")
        padding = max(0, self.kernel_size - self.stride)
        outputs = tf.pad(outputs, [[0, 0], [0, padding], [0, 0], [0, 0]])
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.pw_conv(outputs, training=training)
        B, T, _, E = shape_list(outputs)
        outputs = tf.reshape(outputs, [B, T, E])

        mask = mask[:, :: self.stride, :: self.stride]
        pad_mask = pad_mask[:, :: self.stride]
        _, L = shape_list(pad_mask)
        outputs = tf.pad(outputs, [[0, 0], [0, L - T], [0, 0]])

        return outputs, mask, pad_mask


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Same positional encoding method as NeMo library
    """

    def __init__(self, d_model, max_len=1000, name="positional_encoding_nemo", **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.max_len = max_len
        positions = tf.expand_dims(
            tf.range(self.max_len - 1, -max_len, -1.0, dtype=tf.float32), axis=1
        )
        pos_length = tf.shape(positions)[0]
        pe = np.zeros([pos_length, d_model], "float32")
        div_term = np.exp(
            tf.range(0, d_model, 2, dtype=tf.float32) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        pe = tf.convert_to_tensor(pe)
        self.pe = tf.expand_dims(pe, 0)

    def call(self, inputs, **kwargs):
        # inputs shape [B, T, V]
        _, length, dmodel = shape_list(inputs)
        center_pos = tf.shape(self.pe)[1] // 2
        start_pos = center_pos - length + 1
        end_pos = center_pos + length
        pos_emb = self.pe[:, start_pos:end_pos]
        return tf.cast(pos_emb, dtype=inputs.dtype)


class Conv1dSubsampling(tf.keras.layers.Layer):
    def __init__(self, filters, name="Conv1dSubsampling", **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv = tf.keras.layers.SeparableConv1D(
            filters=filters,
            kernel_size=3,
            strides=2,
            padding="valid",
        )

    def call(self, inputs, training=False, **kwargs):
        # shape (B,T,F)

        x = tf.pad(inputs, [[0, 0], [0, 1], [0, 0]])
        x = self.conv(x)
        x = tf.nn.relu(x)
        return x


class ConformerEncoder(tf.keras.Model):
    # input with shape [B,T,F]
    def __init__(
        self,
        dmodel=144,
        num_blocks=16,
        head_size=36,
        num_heads=4,  # make sure that num_heads*head_size == dmodel
        kernel_size=31,
        dropout=0.1,
        name="conformer_encoder",
        time_reduce_idx=[7],
        time_recover_idx=[15],
    ):
        super().__init__()
        if time_reduce_idx is None:
            self.time_reduce = None
        else:
            if time_recover_idx is None:
                self.time_reduce = "normal"
            else:
                self.time_reduce = "recover"
                assert len(time_reduce_idx) == len(time_recover_idx)
            self.reduce_idx = time_reduce_idx
            self.recover_idx = time_reduce_idx
            self.reduce_stride = 2

        self.dmodel = dmodel
        self.xscale = dmodel**0.5
        self.conv_subsampling = Conv1dSubsampling(filters=dmodel)
        self.linear = tf.keras.layers.Dense(
            dmodel,
            name=f"{name}_linear",
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.pre_ln = tf.keras.layers.LayerNormalization(name=f"{name}_preln")
        self.pe = PositionalEncoding(dmodel, name=f"{name}_pe")
        self.conformer_blocks = []
        recover_dmodels = []
        recover_head_sizes = []
        self.pe_time_reduction = []
        self.time_reduction_layers = []
        self.time_recover_layers = []

        for i in range(num_blocks):
            if self.time_reduce is not None and i in self.reduce_idx:
                recover_dmodel = dmodel
                recover_dmodels.append(recover_dmodel)  # push dmodel to recover later
                recover_head_sizes.append(head_size)  # push head size to recover later
                self.time_reduction_layers.append(
                    TimeReductionLayer(
                        recover_dmodel,
                        dmodel,
                        stride=self.reduce_stride,
                        name=f"{name}_timereduce",
                    )
                )
                self.pe_time_reduction.append(PositionalEncoding(dmodel, name=f"{name}_pe2"))
            if self.time_reduce == "recover" and i in self.recover_idx:
                dmodel = recover_dmodels[-1]  # pop dmodel for recovery
                head_size = recover_head_sizes[-1]  # pop head size for recovery

                self.time_recover_layers.append(tf.keras.layers.Dense(dmodel))
                recover_dmodels = recover_dmodels[:-1]
                recover_head_sizes = recover_head_sizes[:-1]

            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                name=f"{name}_block_{i}",
            )
            self.conformer_blocks.append(conformer_block)

    def call(self, inputs, length, training=False, mask=None, **kwargs):
        # input with shape [B, T, F]
        outputs = self.conv_subsampling(inputs, training=training)
        outputs = self.linear(outputs, training=training)
        padding, kernel_size, stride, num_subsample = 1, 3, 2, 2  # TODO: set these in __init__
        for _ in range(num_subsample):
            length = tf.math.ceil(
                (tf.cast(length, tf.float32) + (2 * padding) - (kernel_size - 1) - 1)
                / float(stride)
                + 1
            )
        pad_mask = tf.sequence_mask(length, maxlen=tf.shape(outputs)[1])
        mask = tf.expand_dims(pad_mask, 1)
        mask = tf.repeat(mask, repeats=[tf.shape(mask)[-1]], axis=1)
        mask = tf.math.logical_and(tf.transpose(mask, perm=[0, 2, 1]), mask)
        pe = self.pe(outputs)
        outputs = outputs * self.xscale
        outputs = self.do(outputs, training=training)
        # pe_org, mask_org = pe, mask

        recover_activations = []
        index = 0  # index to point the queues for pe, recover activations, etc.

        outputs = self.pre_ln(outputs, training=training)
        for i, cblock in enumerate(self.conformer_blocks):
            if self.time_reduce is not None and i in self.reduce_idx:
                recover_activations.append((outputs, mask, pad_mask, pe))
                outputs, mask, pad_mask = self.time_reduction_layers[index](
                    outputs,
                    training=training,
                    mask=mask,
                    pad_mask=pad_mask,
                    **kwargs,
                )
                pe = self.pe_time_reduction[index](outputs)
                index += 1

            if self.time_reduce == "recover" and i in self.recover_idx:
                index -= 1
                recover_activation, mask, pad_mask, pe = recover_activations[index]
                _, T, _ = shape_list(outputs)
                outputs = tf.repeat(outputs, [self.reduce_stride] * T, axis=1)
                _, T, _ = shape_list(recover_activation)
                outputs = self.time_recover_layers[index](outputs[:, :T, :], training=training)
                outputs = outputs + recover_activation

            outputs = cblock(
                [outputs, pe], training=training, mask=mask, pad_mask=pad_mask, **kwargs
            )
        return outputs


batch_size, seq_len, dim = 3, 12, 80
inputs = tf.random.uniform((batch_size, seq_len, dim), minval=-10, maxval=10)

encoder = ConformerEncoder()
lengths = [4, 8, 12]
outputs = encoder(inputs, lengths)
print(outputs.shape)
encoder.summary()
