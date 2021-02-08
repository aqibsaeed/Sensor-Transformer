""" Adapted from: https://github.com/lucidrains/vit-pytorch/tree/main/vit_pytorch """

import tensorflow as tf
from einops.layers.tensorflow import Rearrange

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim) 
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation=tf.keras.activations.gelu), 
            tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm_a = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_b = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_a = tf.keras.layers.Dropout(dropout)
        self.dropout_b = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout_a(attn_output, 
            training=training)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output, 
            training=training)
        return self.layernorm_b(out_a + ffn_output)

class SensorTransformer(tf.keras.Model):
    def __init__(
        self,
        signal_length,
        segment_size,
        channels,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        dropout,
    ):
        super(SensorTransformer, self).__init__()
        num_patches = (signal_length // segment_size)

        self.segment_size = segment_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_emb = self.add_weight("pos_emb", 
            shape=(1, num_patches + 1, d_model))
        self.class_emb = self.add_weight("class_emb", 
            shape=(1, 1, d_model))
        self.patch_proj = tf.keras.layers.Dense(d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(mlp_dim, 
                    activation=tf.keras.activations.gelu),
                tf.keras.layers.Dense(num_classes),
            ]
        )

    def call(self, input, training):        
        batch_size = tf.shape(input)[0]
        patches = Rearrange("b (w p1) c-> b w (p1 c)", 
            p1=self.segment_size)(input)
        x = self.patch_proj(patches)

        class_emb = tf.broadcast_to(self.class_emb, 
            [batch_size, 1, self.d_model])
        
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb
        
        for layer in self.enc_layers:
            x = layer(x, training)

        return self.mlp_head(x[:, 0])