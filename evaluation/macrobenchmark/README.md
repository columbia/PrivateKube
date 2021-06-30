
# Macrobenchmark


## Requirements

The commands in this section have to be run from the `macrobenchmark` directory. You can jump there with:

```bash
cd evaluation/macrobenchmark
```

You should have the `privatekube` Python package installed, as described in the [main README](https://github.com/columbia/PrivateKube).

Please note that the steps below can take several days to run, depending on your hardware. If you want to examine the experimental data without running the preprocessing or the training yourself, you can download some artifacts from this [public bucket](https://storage.googleapis.com/privatekube-public-artifacts). 

Training will be faster with a Nvidia GPU, but you can also use your CPU by specifying `--device=cpu` in the script arguments.

## Download and preprocess the dataset

To download a preprocessed and truncated (140Mb instead of 7Gb) version of the dataset, run the following:

```bash
python dataset.py getmini
```

Alternatively, you can download the raw dataset from the [source](https://nijianmo.github.io/amazon/index.html) and preprocess it yourself as follows:

1. First, download the dataset with:

```bash
python dataset.py download
```

2. Then, preprocess the dataset:
```bash
python dataset.py preprocess
```

3. Finally, merge the preprocessed dataset into a single `reviews.h5` file that can be used by the machine learning pipelines:

```bash
python dataset.py merge
```

4. You can also change the tokenizer from Bert to a custom vocabulary (e.g. the 10,000 most common English words) with:

```bash
python dataset.py convert
```

Use `python dataset.py --help` or `python dataset.py preprocess --help` to have more details about the options.

If you wish, you can examine the preprocessed data blocks with `h5dump`:
```bash
h5dump -d 999-99 Books_5.h5
```

## Run individual DP machine learning models

Once you have a preprocessed dataset (either the full dataset or a truncated version), you can run the neural networks with the `classification.py` script, which accepts a lot of options defined at the top of the file. The default flags are set to reasonable values.

Here is an example:
```bash
python workload/models/classification.py --model="bert" --task="product" --n_blocks=200 --n_epochs=3 --dp=1 --epsilon=1.0 --delta=1e-9 --batch_size=32 --user_level=1
```

And another example:
```
python workload/models/classification.py --model="lstm" --task="sentiment" --n_blocks=500 --n_epochs=15 --dp=1 --epsilon=2.0 --delta=1e-9 --batch_size=64 --user_level=0
```

Similarly, you can run statistics:
```bash
python workload/models/statistics.py --dp="event" --n_blocks=1 --model="avg_rating" --epsilon=0.1 
```


## Run large experiments

To create a workload from the individual models, we have to run them for various parameters.

First, you can generate the experiments (with nice parameters and a local path to store the results) with:

```bash
python models.py generate
```

Then, you can run all the experiments with:

```bash
python models.py all
```

## End-to-end macrobenchmark

The `evaluation/macrobenchmark/scheduling` folder contains a Go stub that mimicks the Kubernetes API. We can then deploy the privacy resource and run the real scheduler over workload traces The traces are generated with `models.py` as above. This is much faster than using a full cluster.

To compile the stub, please go into the `scheduling` folder and run:

```bash
go build
```

The `evaluation/macrobenchmark/scheduler.py` file provides a command-line interface to interact with the stub. You can run `python scheduler.py --help` for more information.

For instance, you can run a workload (in fast-forward) with pipelines drawn from the distribution described in the paper, for different values of N by running:

```bash
python scheduler.py schedulers --config-path config.yaml --keep-raw-logs True
```

Where `config.yaml` is:

```yaml
N:
  - 1
  - 100
  - 200
  - 300
  - 400
  - 500
timeout: 5
epsilon: 10
delta: 1e-7
rdp:
  - 1
  - 0
n: 50
t: 30_000
elephants: "/home/<USERNAME>/PrivateKube/evaluation/macrobenchmark/workload/runs/event/elephants"
mice: "/home/<USERNAME>/PrivateKube/evaluation/macrobenchmark/workload/runs/event/mice-laplace"
mice_ratio: 0.75
m: 300
b: 10
```



