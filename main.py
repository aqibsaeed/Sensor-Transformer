import argparse
import numpy as np
import tensorflow as tf
from sensortransformer import set_network
from data import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--signal-length", type=int)
parser.add_argument("--segment-size", type=int)
parser.add_argument("--num_channels", type=int)
parser.add_argument("--num_classes", type=int)
parser.add_argument("--num-layers", default=4, type=int)
parser.add_argument("--d-model", default=64, type=int)
parser.add_argument("--num-heads", default=4, type=int)
parser.add_argument("--mlp-dim", default=64, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--weight-decay", default=0.0001, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--epochs", default=50, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    ds_train, ds_test = load_data(args.batch_size) 
    model = set_network.SensorTransformer(
        signal_length=args.signal_length,
        segment_size=args.segment_size,
        channels=args.num_channels,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), 
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    model.fit(ds_train, epochs=args.epochs, verbose=1)
    model.evaluate(ds_test)
