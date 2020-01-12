# GCP-Cloud-Vision

## Contents

* Serving vision models on GCP
* Dealing with data scarcity
* Transfer Learning & Reinforcement learning for architecture designing
* Cloud Vision API / AutoML Vision

## Serving_CNN

The models are stored in the `mnistmodel/trainer/model.py`  file.

Two other important `task.py` under `mnistmodel/trainer` and a `setup.py` file under `mnistmodel` directory.

#### Step 1

To serve, we are going to run the code **locally** to test the code before training on Cloud ML Engine’s GPU:

```python
%%bash
rm -rf mnistmodel.tar.gz mnist_trained
gcloud ml-engine local train \
    --module-name=trainer.task \
    --package-path=${PWD}/mnistmodel/trainer \
    -- \
    --output_dir=${PWD}/mnist_trained \
    --train_steps=100 \
    --learning_rate=0.01 \
    --model=$MODEL_TYPE
```

#### Step 2

Once the model is working on the local machine, let’s train on the Cloud ML Engine with GPU:

```python
%%bash
OUTDIR=gs://${BUCKET}/mnist/trained_${MODEL_TYPE}
JOBNAME=mnist_${MODEL_TYPE}_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME \
    --region=$REGION \
    --module-name=trainer.task \
    --package-path=${PWD}/mnistmodel/trainer \
    --job-dir=$OUTDIR \
    --staging-bucket=gs://$BUCKET \
    --scale-tier=BASIC_GPU \
    --runtime-version=$TFVERSION \
    -- \
    --output_dir=$OUTDIR \
    --train_steps=10000 --learning_rate=0.01 --train_batch_size=512 \
    --model=$MODEL_TYPE --batch_norm
```

While training, we can launch `tensorboard` to monitor the training process:

```python
from google.datalab.ml import TensorBoard

TensorBoard().start("gs://{}/mnist/trained_{}".format(BUCKET, MODEL_TYPE))

for pid in TensorBoard.list()["pid"]:
  TensorBoard().stop(pid)
  print("Stopped TensorBoard with pid {}".format(pid))
```

#### Step 3

Once training finished, deploy the model on GCloud ML-Engine:

```python
%%bash
MODEL_NAME="mnist"
MODEL_VERSION=${MODEL_TYPE}
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/mnist/trained_${MODEL_TYPE}/export/exporter | tail -1)
echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"
#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=$TFVERSION
```

#### Step 4

For **hyper-parameter tuning**, create a `hyperparam.yaml` :

```python
%%writefile hyperparam.yaml
trainingInput:
  scaleTier: CUSTOM
  masterType: complex_model_m_gpu
  hyperparameters:
    goal: MAXIMIZE
    maxTrials: 30
    maxParallelTrials: 2
    hyperparameterMetricTag: accuracy
    params:
    - parameterName: train_batch_size
      type: INTEGER
      minValue: 32
      maxValue: 512
      scaleType: UNIT_LINEAR_SCALE
    - parameterName: learning_rate
      type: DOUBLE
      minValue: 0.001
      maxValue: 0.1
      scaleType: UNIT_LOG_SCALE
    - parameterName: nfil1
      type: INTEGER
      minValue: 5
      maxValue: 20
      scaleType: UNIT_LINEAR_SCALE
    - parameterName: nfil2
      type: INTEGER
      minValue: 10
      maxValue: 30
      scaleType: UNIT_LINEAR_SCALE
    - parameterName: dprob
      type: DOUBLE
      minValue: 0.1
      maxValue: 0.6
      scaleType: UNIT_LINEAR_SCALE
```

## Dealing with data scarcity

### Data Augmentation

```python
def make_input_fn(csv_of_filenames, batch_size, mode, augment):
  def _input_fn():
    def decode_csv(csv_row):
      filename, labe = tf.decode_csv(
      csv_row, record_defaults = [[ ],[ ]])
    image_bytes = tf.read_file(filename)
    return image_bytes, label
  
  dataset = tf.data.TextLineDataset(csv_of_filenames).map(decode_csv).map(decode_jpeg).map(resize_image)
  if augment:
    dataset = dataset.map(augment_image)
  dataset = dataset.map(postprocess_image)
  
  
  return dataset.make_one_shot_iterator().get_next()
  return _input_fn  
```

To define `decode_jpeg` function:

```python
import tensorflow.image as tfi

def decode_jpeg(image_bytes, label):
  image = tfi.decode_jpeg(image_bytes, channels=NUM_CHANNELS)
  image = tfi.convert_image_dtype(image, dtype=tf.float32)
  return image, label
```

To define `augment_image` function:

```python
def augment_image(image_dict, label=None):
  image = image_dict['image']
  image = tf.expand_dims(image, 0) # resize_bilinear needs batches
  image = tfi.resize_bilinear(image, [HEIGHT+10, WIDTH+10], align_corners=False)
  image = tfi.squeeze(image) # remove batch dimension
  image = tfi.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])
  image = tfi.random_flip_left_right(image)
  image = tfi.random_brightness(image, max_delta=63.0/255.0)
  image = tfi.random_contrast(image, lower=0.2, upper=1.8)
  
  return image, label
```

## Transfer learning

![image-20200109120518992](images/transfer_learning.jpeg)

This is another way of thinking about Transfer learning. When we don’t have enough data to close the distance between **random initialization** and **target optimum**, we could go for the **related optimum** first by training the network for another related but different task. Then start from **related optimum** and train for the **target optimum**.

 ## Reinforcement Learning for designing architectures

![reinforcement_learning](images\reinforcement_learning.png)

This is a technique that Google use to search for potential neural network architectures for certain tasks in an **non-manual** manner.

The working principle is similar to a **GAN** but simpler on the **discriminator** side: The **controller(RNN)** propose an architecture (it’s sampled from it’s architecture pool with probability **p**). The **”discriminator”** trains the network and evaluate it’s performance(accuracy **R**). Thirdly, compute gradient of **p** and scale it by **R** to update the controller.

Eventually, the **controller** learns to assign **high** probabilities to the areas of the *architecture space*  that **achieve better accuracy** and **low** probabilities to areas that performed poorly. This lays the foundation for the **AutoML** that Google builds.

## Cloud Vision API / AutoML Vision

![different_learning_tools](images\different_learning_tools.png)

These are the four categories of approaches we can take to deploy ML models on GCP.

With the most flexible customized models on the left to **AutoML** on the right where we barely have to write and code.

**BigQuery** ML has power ML structure to deal with **structured datasets** and only **structured**.

