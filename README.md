## Sensor Transformer (SeT)
Adaptation of <a href="https://arxiv.org/pdf/2010.11929.pdf">Vision Transformer (ViT)</a> for Time-Series and Sensor Data in Tensorflow. 

#### Problems/Datasets
* <a href="https://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition">Human Activity Recognition with Wearable Device</a>
* <a href="https://www.isip.piconepress.com/projects/tuh_eeg/">Seizure Detection</a>
* <a href="https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/">Predictive Maintenance</a>

#### Tools

* <a href="https://www.tensorflow.org/">Tensorflow 2.4</a>
* <a href="https://github.com/arogozhnikov/einops">einops</a>

#### Usage

```python
import argparse
import tensorflow as tf
import transformer

parser = argparse.ArgumentParser()
parser.add_argument("--signal-length", type=int)
parser.add_argument("--segment-size", type=int)
parser.add_argument("--num_channels", type=int)
parser.add_argument("--num_classes", type=int)
args = parser.parse_args()

"""
TF-Data objects, see data.load_data function.
Instances must be of shape x = (batch, signal_length, num_channels)
y = (batch, num_classes)
"""
ds_train, ds_test = ...

model = transformer.SensorTransformer(
        signal_length=args.signal_length,
        segment_size=args.segment_size,
        channels=args.num_channels,
        num_classes=args.num_classes,
        num_layers=4,
        d_model=64,
        num_heads=4,
        mlp_dim=64,
        dropout=0.1,
)
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.Adam(),
      metrics=[tf.keras.metrics.CategoricalAccuracy()],
)
model.fit(ds_train, epochs=50, verbose=1)
model.evaluate(ds_test)
```

Thanks to Phil Wang for open-sourcing <a href="https://github.com/lucidrains/vit-pytorch">Pytorch implementation</a> of ViT



