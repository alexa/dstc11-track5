import os
import sys
import argparse
import json
import warnings

from jsonschema import validate

from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader


def main(argv):
    parser = argparse.ArgumentParser(description='Check the validity of system outputs.')
    
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', choices=['train', 'val', 'test'], required=True, help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store', metavar='PATH', required=True,
                        help='Will look for corpus in <dataroot>/<dataset>/...')

    parser.add_argument('--outfile',dest='outfile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing output JSON')
    parser.add_argument('--knowledge',dest='knowledge_file',action='store',metavar='JSON_FILE',required=False,
                        default='knowledge.json', help='Knowledge JSON file')
    parser.add_argument('--schema',dest='schema_file',action='store',metavar='JSON_FILE',required=False,
                        default="output_schema.json", help='Output schema JSON file')

    args = parser.parse_args()
    
    data = DatasetWalker(dataset=args.dataset, dataroot=args.dataroot)

    knowledge_reader = KnowledgeReader(dataroot=args.dataroot, knowledge_file=args.knowledge_file)

    try:
        with open(os.path.join(args.dataroot, args.schema_file), 'r') as f:
            schema = json.load(f)
    except FileNotFoundError:
        sys.exit('Schema file does not exist at %s' % os.path.join(args.dataroot, args.schema_file))

    try:
        with open(args.outfile, 'r') as f:
            output = json.load(f)
    except FileNotFoundError:
        sys.exit('Output file does not exist at %s' % args.outfile)

    # initial syntax check with the schema
    validate(instance=output, schema=schema)

    # check the number of labels
    if len(data) != len(output):
        raise ValueError("the number of instances between ground truth and output does not match")

    error_msg = []
    warning_msg = []
    
    for idx in range(len(output)):
        item = output[idx]

        if item['target'] is True:
            for knowledge_label in item['knowledge']:
                domain = knowledge_label['domain']
                entity_id = str(knowledge_label['entity_id'])
                doc_type = knowledge_label['doc_type']
                doc_id = str(knowledge_label['doc_id'])

                if doc_type == 'review':
                    if 'sent_id' in knowledge_label:
                        sent_id = str(knowledge_label['sent_id'])
                        
                        # check the knowledge
                        try:
                            doc = knowledge_reader.get_review_sent(domain, entity_id, doc_id, sent_id)
                        except ValueError as err:
                            error_msg.append("On instance[%d]: %s" % (idx, err))
                    else:
                        error_msg.append("On instance[%d]: found no sentence ID" % (idx,))
                elif doc_type == 'faq':
                    if 'sent_id' in knowledge_label:
                        warning_msg.append("On instance[%d]: sentence ID is given for FAQ knowledge" % (idx,))

                    # check the knowledge
                    try:
                        doc = knowledge_reader.get_faq_doc(domain, entity_id, doc_id)
                    except ValueError as err:
                        error_msg.append("On instance[%d]: %s" % (idx, err))
        
        # check the additional properties for non-target instances
        elif len(item) > 1:
            warning_msg.append("On instance[%d]: additional properties for non-target instance" % idx)

    if len(error_msg) > 0:
        raise ValueError('Found %d errors:\n' % len(error_msg) + '\n'.join(error_msg))
    elif len(warning_msg) > 0:
        warnings.warn('Found %d warnings:\n' % len(warning_msg) + '\n'.join(warning_msg))
    else:
        print("Found no error, output file is valid.")

if __name__ =="__main__":
    main(sys.argv)        
