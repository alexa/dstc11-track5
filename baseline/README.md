[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Neural Baseline Models for DSTC11 Track 5

This directory contains the official baseline codes for [DSTC11 Track 5](../README.md).

## Getting started

* Clone this repository into your working directory.

``` shell
$ git clone https://github.com/alexa/dstc11-track5.git
$ cd dstc11-track5
```

* Install the required python packages.

``` shell
$ pip3 install -r requirements.txt
$ python -m nltk.downloader 'punkt'
$ python -m nltk.downloader 'wordnet
```

* Train the baseline models.

``` shell
$ ./bin/run_baseline_training.sh
```

* Run the baseline models.

``` shell
$ ./bin/run_baseline_eval.sh
```

* Validate the structure and contents of the tracker output.

``` shell
$ python -m scripts.check_results --dataset val --dataroot data/ --outfile pred/val/baseline.rg.bart-base.json
Found no errors, output file is valid
```

* Evaluate the output.

``` shell
$ python -m scripts.scores --dataset val --dataroot data/ --outfile pred/val/baseline.rg.bart-base.json --scorefile pred/val/baseline.rg.bart-base.score.json
```

* Print out the scores.

``` shell
$ cat pred/val/baseline.rg.bart-base.score.json | jq
{
  "detection": {
    "prec": 1,
    "rec": 0.9990605918271489,
    "f1": 0.9995300751879699
  },
  "selection": {
    "prec": 0.7950632648828044,
    "rec": 0.8843003806667435,
    "f1": 0.8373109060127791,
    "em_acc": 0.40488492249882574
  },
  "generation": {
    "bleu": 0.10419026648096004,
    "meteor": 0.1810290467362776,
    "rouge_1": 0.36509971645462036,
    "rouge_2": 0.15059917821384866,
    "rouge_l": 0.2874932052481134
  }
}
```

