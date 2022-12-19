# DSTC11 Track 5 Dataset

This directory contains the official training/validation datasets for [DSTC11 Track 5](../README.md).

## Data

We are releasing the data divided into the following two subsets:

* Training set
  * [logs.json](train/logs.json): training instances
  * [labels.json](train/labels.json): ground-truths for training instances
* Validation set:
  * [logs.json](val/logs.json): validation instances
  * [labels.json](val/labels.json): ground-truths for validation instances

The ground-truth labels for Knowledge Selection task refer to the knowledge snippets in [knowledge.json](knowledge.json).

Participants will develop systems to take *logs.json* as an input and generates outputs following the **same format** as *labels.json*.

## JSON Data Formats

### Log Objects

The *logs.json* file includes a list of instances each of which is a partial conversation from the beginning to the target user turn.
Each instance is a list of the following turn objects:

* speaker: the speaker of the turn (string: "U" for user turn/"S" for system turn)
* text: utterance text (string)

### Label Objects

The *labels.json* file provides the ground-truth labels and human responses for the final turn of each instance in *logs.json*.
It includes the list of the following objects in the same order as the input instances:

* target: whether the turn is knowledge-seeking or not (boolean: true/false)
* knowledge: [
  * domain: the domain identifier referring to a relevant knowledge snippet in *knowledge.json* (string)
  * entity\_id: the entity identifier referring to a relevant knowledge snippet in *knowledge.json* (integer)
  * doc\_type: the document type identifier referring to a relevant knowledge snippet in *knowledge.json* (string: 'review' or 'faq')
  * doc\_id: the document identifier referring to a relevant knowledge snippet in *knowledge.json* (integer)
  * sent\_id: the sentence identifier (only for reviews) referring to a relevant knowledge snippet in *knowledge.json* (integer)
  ]
* response: knowledge-grounded system response (string)

NOTE: *knowledge* and *response* exist only for the target instances with *true* for the *target* value.

### Knowledge Objects

The *knowledge.json* contains the unstructured knowledge sources to be selected and grounded in the tasks.
It includes the domain/entity-specific reviews and FAQs in the following format:

* domain: domain identifier (string: "hotel", "restaurant")
  * entity\_id: entity identifier (integer)
      * name: entity name (string; only exists for entity-specific knowledge)
      * reviews
          * review\_id: review document identifier (integer)
              * sentences
                  * sent\_id: review sentence identifier (interger)
                    * sent: review sentence (string)
              * metadata (incl. traveler type, dishes, drinks)
      * faqs
          * faq\_id: faq document identifier (integer)
              * question: question (string)
              * answer: answer (string)
