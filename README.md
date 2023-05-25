# DSTC11 Track 5 - Task-oriented Conversational Modeling with Subjective Knowledge

This repository contains the data, scripts and baseline codes for [DSTC11](https://dstc11.dstc.community/) Track 5.

This challenge track aims to support more informative and engaging task-oriented conversations by utilizing the subjective knowledge from review posts.
Track participants will develop dialogue systems to understand relevant review posts, and generate system responses grounded on the selected review knowledge snippets.

**Organizers:** Seokhwan Kim, Spandana Gella, Chao Zhao, Di Jin, Alexandros Papangelis, Behnam Hedayatnia, Yang Liu, Dilek Hakkani-Tur

## News
* **April 17, 2023** - The human evaluation scores for each finalist entry are released at [results/](results/).
* **April 17, 2023** - The system outputs submitted by the participants are released at [results/](results/).
* **April 17, 2023** - The ground-truth labels/responses for the evaluation data are released at [data/test/labels.json](data/test/labels.json). 
* **April 13, 2023** - The human evaluation results are now available: [See Results](https://docs.google.com/spreadsheets/d/1cgUWr6h2PHvZa1Ez0bLhInkQbO-GCp9RJtUbTumdjoY/edit?usp=sharing).
* **April 7, 2023** - The objective evaluation results are now available: [See Results](https://docs.google.com/spreadsheets/d/1cgUWr6h2PHvZa1Ez0bLhInkQbO-GCp9RJtUbTumdjoY/edit?usp=sharing).
* **March 24, 2023** - The evaluation data is released. Please find the data and participation details from [data/](data/README.md).

## Important Links
* [Track Proposal](https://drive.google.com/file/d/1wHZdlz8JecDWiiJiwhP3VsKnbApdL6_e/view)
* [Challenge Registration](https://forms.gle/e2qVGPPAhpp8Upt8A)
* [Data Formats](data/README.md)
* [Baseline Details](baseline/README.md)
* [Objective Evaluation Results](https://docs.google.com/spreadsheets/d/1cgUWr6h2PHvZa1Ez0bLhInkQbO-GCp9RJtUbTumdjoY/edit?usp=sharing)
* [Human Evaluation Results](https://docs.google.com/spreadsheets/d/1cgUWr6h2PHvZa1Ez0bLhInkQbO-GCp9RJtUbTumdjoY/edit?usp=sharing)

If you want to publish experimental results with this dataset or use the baseline models, please cite [this article](https://arxiv.org/abs/2305.12091):
```
@misc{zhao2023what,
      title={"What do others think?": Task-Oriented Conversational Modeling with Subjective Knowledge}, 
      author={Chao Zhao and Spandana Gella and Seokhwan Kim and Di Jin and Devamanyu Hazarika and Alexandros Papangelis and Behnam Hedayatnia and Mahdi Namazifar and Yang Liu and Dilek Hakkani-Tur},
      year={2023},
      eprint={2305.12091},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Tasks

This challenge track distinguishes between turns that could be handled by the existing task-oriented conversational models with no extra knowledge and turns that require external subjective knowledge to be answered by the dialogue system.
We focus on the turns that require knowledge access as the evaluation target in this track by the following three tasks:

| Task #1 | Knowledge-seeking Turn Detection                                                                                                      |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal    | To decide whether to continue the existing scenario or trigger the knowledge access branch for a given utterance and dialogue history |
| Input   | Current user utterance, Dialogue context, Knowledge snippets                                                                          |
| Output  | Binary class (requires knowledge access or not)                                                                                       |

| Task #2 | Knowledge Selection                                                                                                                   |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal    | To select proper subjective knowledge sources given a dialogue state at each turn with knowledge access                               |
| Input   | Current user utterance, Dialogue context, Knowledge snippets                                                                          |
| Output  | List of relevant knowledge candidates                                                                                                 |

| Task #3 | Knowledge-grounded Response Generation                                                                                                |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal    | To take a triple of input utterance, dialog context, and the selected knowledge snippets and generate a system response               |
| Input   | Current user utterance, Dialogue context, and Selected knowledge snippets                                                             |
| Output  | Generated system response                                                                                                             |

Participants will develop systems to generate the outputs for each task.
They can leverage the annotations and the ground-truth responses available in the training and validation datasets.

In the test phase, participants will be given a set of unlabeled test instances.
And they will submit **up to 5** system outputs for **all three tasks**.

**NOTE**: For teams who are interested in only one or two of the tasks, we recommend to use our baseline system for the remaining tasks to complete the system outputs.

## Evaluation

Each submission will be evaluated in the following task-specific automated metrics first:

| Task                                   | Automated Metrics                    |
|----------------------------------------|--------------------------------------|
| Knowledge-seeking Turn Detection       | Precision/Recall/F-measure           |
| Knowledge Selection                    | Precision/Recall/F-measure, Accuracy |
| Knowledge-grounded Response Generation | BLEU, ROUGE, METEOR                  |

To consider the dependencies between the tasks, the scores for knowledge selection and knowledge-grounded response generation are weighted by knowledge-seeking turn detection performances. Please find more details from [scores.py](scripts/scores.py).

The final ranking will be based on **human evaluation results** only for selected systems according to automated evaluation scores.
It will address the following aspects: appropriateness and relevance to given knowledge.

## Data

In this challenge track, participants will use an augmented version of [MultiWoz 2.1](https://github.com/budzianowski/multiwoz) which includes newly introduced subjective knowledge-seeking turns.
All the ground-truth annotations for Knowledge-seeking Turn Detection and Knowledge Selection tasks as well as the agent's responses for Knowledge-grounded Response Generation task are available to develop the components on the [training](data/train) and [validation](data/val) sets.
In addition, relevant knowledge snippets for each domain and entity are also provided in [knowledge.json](data/knowledge.json).

In the test phase, participants will be evaluated on the results generated by their models for the unlabeled test set.
To evaluate the generalizability and the portability of each model, the unseen test set may include different domains, entities and locales than MultiWoz.

Data and system output format details can be found from [data/README.md](data/README.md).

## Timeline

* Training data released: Dec 19, 2022 
* Test data released: Mar 24, 2023
* Entry submission deadline: Mar 31, 2023
* Objective evaluation completed: Apr 7, 2023
* Human evaluation completed: Apr 14, 2023

## Rules

* Participation is welcome from any team (academic, corporate, non profit, government).
* The identity of participants will NOT be published or made public. In written results, teams will be identified as team IDs (e.g. team1, team2, etc). The organizers will verbally indicate the identities of all teams at the workshop chosen for communicating results.
* Participants may identify their own team label (e.g. team5), in publications or presentations, if they desire, but may not reveal the identities of other teams.
* Participants are allowed to use any external datasets, resources or pre-trained models.

## Contact

### Join the DSTC mailing list to get the latest updates about DSTC11
* To join the mailing list: visit https://groups.google.com/a/dstc.community/forum/#!forum/list/join

* To post a message: send your message to list@dstc.community

* To leave the mailing list: visit https://groups.google.com/a/dstc.community/forum/#!forum/list/unsubscribe

### For specific enquiries about DSTC11 Track 5

Please feel free to contact: seokhwk (at) amazon (dot) com

## License

The code is licensed under Apache 2.0 (see [SOFTWARELICENSE](SOFTWARELICENSE)) and the data files are licensed under CDLA-Sharing 1.0 (see [DATALICENSE](DATALICENSE)).

