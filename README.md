# GCP-Cloud-Vision

## Contents

* Serving vision models on GCP
* Dealing with data scarcity
* Transfer Learning
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





