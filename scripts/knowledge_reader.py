import os
import json

class KnowledgeReader(object):
    def __init__(self, dataroot, knowledge_file='knowledge.json'):
        path = os.path.join(os.path.abspath(dataroot))

        with open(os.path.join(path, knowledge_file), 'r') as f:
            self.knowledge = json.load(f)

    def get_domain_list(self):
        return list(self.knowledge.keys())

    def get_entity_list(self, domain):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name")

        entity_ids = []
        for entity_id in self.knowledge[domain].keys():
            entity_ids.append(int(entity_id))

        result = []
        for entity_id in sorted(entity_ids):
            entity_name = self.knowledge[domain][str(entity_id)]['name']
            result.append({'id': entity_id, 'name': entity_name})

        return result

    def get_entity_name(self, domain, entity_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        result = self.knowledge[domain][str(entity_id)]['name'] or None

        return result

    def get_faq_doc_ids(self, domain, entity_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)
        
        result = []

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        entity_obj = self.knowledge[domain][str(entity_id)]
        for doc_id, doc_obj in entity_obj['faqs'].items():
            result.append(doc_id)

        return result

    def get_faq_doc(self, domain, entity_id, doc_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        entity_name = self.get_entity_name(domain, entity_id)

        if str(doc_id) not in self.knowledge[domain][str(entity_id)]['faqs']:
            raise ValueError("invalid doc id: %s" % str(doc_id))

        doc_obj = self.knowledge[domain][str(entity_id)]['faqs'][str(doc_id)]
        result = {'domain': domain, 'entity_id': entity_id, 'entity_name': entity_name, 'doc_id': doc_id, 'question': doc_obj['question'], 'answer': doc_obj['answer']}

        return result

    def get_review_doc_ids(self, domain, entity_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        result = []
        
        entity_obj = self.knowledge[domain][str(entity_id)]
        for doc_id, doc_obj in entity_obj['reviews'].items():
            result.append(doc_id)

        return result

    def get_review_doc(self, domain, entity_id, doc_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        entity_name = self.get_entity_name(domain, entity_id)

        if str(doc_id) not in self.knowledge[domain][str(entity_id)]['reviews']:
            raise ValueError("invalid doc id: %s" % str(doc_id))
        
        doc_obj = self.knowledge[domain][str(entity_id)]['reviews'][str(doc_id)]
        
        result = {'domain': domain, 'entity_id': entity_id, 'entity_name': entity_name, 'doc_id': doc_id, 'sentences': doc_obj['sentences']}
        if 'traveler_type' in doc_obj:
            result['traveler_type'] = doc_obj['traveler_type']
        
        if 'dishes' in doc_obj:
            result['dishes'] = doc_obj['dishes']

        if 'drinks' in doc_obj:
            result['drinks'] = doc_obj['drinks']

        return result
    
    def get_review_sent(self, domain, entity_id, doc_id, sent_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))
        
        if str(doc_id) not in self.knowledge[domain][str(entity_id)]['reviews']:
            raise ValueError("invalid doc id: %s" % str(doc_id))

        if str(sent_id) not in self.knowledge[domain][str(entity_id)]['reviews'][str(doc_id)]['sentences']:
            raise ValueError("invalid sentence id: %s" % str(sent_id))

        result = self.knowledge[domain][str(entity_id)]['reviews'][str(doc_id)]['sentences'][str(sent_id)]

        return result
