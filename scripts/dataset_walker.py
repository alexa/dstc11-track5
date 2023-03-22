import os
import json

from scripts.knowledge_reader import KnowledgeReader

class DatasetWalker(object):
    def __init__(self, dataset, dataroot, labels=False, labels_file=None, incl_knowledge=False):
        path = os.path.join(os.path.abspath(dataroot))
            
        if dataset not in ['train', 'val', 'test']:
            raise ValueError('Wrong dataset name: %s' % (dataset))

        logs_file = os.path.join(path, dataset, 'logs.json')
        with open(logs_file, 'r') as f:
            self.logs = json.load(f)

        self.labels = None

        if labels is True:
            if labels_file is None:
                labels_file = os.path.join(path, dataset, 'labels.json')

            with open(labels_file, 'r') as f:
                self.labels = json.load(f)

        self._incl_knowledge = incl_knowledge
        if self._incl_knowledge is True:
            self._knowledge = KnowledgeReader(dataroot)

    def __iter__(self):
        if self.labels is not None:
            for log, label in zip(self.logs, self.labels):
                if self._incl_knowledge is True and label['target'] is True:
                    for idx, snippet in enumerate(label['knowledge']):
                        domain = snippet['domain']
                        entity_id = snippet['entity_id']
                        doc_type = snippet['doc_type']
                        doc_id = snippet['doc_id']

                        if doc_type == 'review':
                            sent_id = snippet['sent_id']                            
                            sent = self._knowledge.get_review_sent(domain, entity_id, doc_id, sent_id)
                            label['knowledge'][idx]['sent'] = sent
                            
                        elif doc_type == 'faq':
                            doc = self._knowledge.get_faq_doc(domain, entity_id, doc_id)
                            question = doc['question']
                            answer = doc['answer']

                            label['knowledge'][idx]['question'] = question
                            label['knowledge'][idx]['answer'] = answer
                
                yield(log, label)
        else:
            for log in self.logs:
                yield(log, None)

    def __len__(self, ):
        return len(self.logs)
