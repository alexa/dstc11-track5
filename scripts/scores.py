import argparse
import json
import os
import sys

import requests
from rouge_score import rouge_scorer
import summ_eval
from summ_eval.bleu_metric import BleuMetric
from summ_eval.meteor_metric import MeteorMetric

from scripts.dataset_walker import DatasetWalker


class Metric:
    def __init__(self):
        self.reset()
        self.set_meteor()

    def reset(self):
        self._detection_tp = 0.0
        self._detection_fp = 0.0
        self._detection_fn = 0.0
        
        self._selection_tp = 0.0
        self._selection_fp = 0.0
        self._selection_fn = 0.0
        self._selection_exact_matched = 0.0
        self._selection_total = 0.0

        self._generation_rouge_l = 0.0
        self._generation_rouge_1 = 0.0
        self._generation_rouge_2 = 0.0

        self._ref_responses = []
        self._pred_responses = []

    def set_meteor(self):
        file_path = summ_eval.__file__
        dir = os.path.dirname(file_path)
        if not os.path.exists(os.path.join(dir, "data")):
            os.mkdir(os.path.join(dir, "data"))
        if not os.path.exists(os.path.join(dir, "data", "paraphrase-en.gz")):
            paraphrase_en_gz_url = "https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/meteor/data/paraphrase-en.gz?raw=true"
            r = requests.get(paraphrase_en_gz_url)
            with open(os.path.join(dir, "data", "paraphrase-en.gz"), "wb") as outputf:
                outputf.write(r.content)

    def _match_knowledge_obj(self, obj1, obj2):
        matched = False
        if obj1['doc_type'] == 'review' and obj2['doc_type'] == 'review':
            if obj2['domain'] == obj1['domain'] and obj2['entity_id'] == obj1['entity_id'] and obj2['doc_id'] == obj1['doc_id'] and obj2['sent_id'] == obj1['sent_id']:
                matched = True
        elif obj1['doc_type'] == 'faq' and obj2['doc_type'] == 'faq':
            if obj2['domain'] == obj1['domain'] and obj2['entity_id'] == obj1['entity_id'] and obj2['doc_id'] == obj1['doc_id']:
                matched = True
        return matched

    def _remove_duplicate_knowledge(self, objs):
        result = []
        for obj_i in objs:
            duplicated = False
            for obj_j in result:
                if self._match_knowledge_obj(obj_i, obj_j) is True:
                    duplicated = True
            if duplicated is False:
                result.append(obj_i)
        return result

    def _match_knowledge(self, ref_objs, pred_objs):
        num_matched = 0
        for ref in ref_objs:
            for pred in pred_objs:
                if self._match_knowledge_obj(ref, pred):
                    num_matched += 1

        tp = num_matched
        fp = len(pred_objs) - num_matched
        fn = len(ref_objs) - num_matched

        if len(ref_objs) == len(pred_objs) and len(ref_objs) == tp:
            exact_matched = 1
        else:
            exact_matched = 0

        return (tp, fp, fn, exact_matched)
    
    def _rouge(self, ref_response, hyp_response):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        scores = scorer.score(ref_response, hyp_response)

        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure

        return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}

                    
    def update(self, ref_obj, hyp_obj):
        if ref_obj['target'] is True:
            if hyp_obj['target'] is True:
                self._ref_responses.append(ref_obj['response'])
                self._pred_responses.append(hyp_obj['response'])
                
                self._detection_tp += 1

                rouge_scores = self._rouge(ref_obj['response'], hyp_obj['response'])
                self._generation_rouge_l += rouge_scores['rougeL']
                self._generation_rouge_1 += rouge_scores['rouge1']
                self._generation_rouge_2 += rouge_scores['rouge2']
            else:
                self._detection_fn += 1
        else:
            if hyp_obj['target'] is True:
                self._detection_fp += 1

        ref_knowledge = ref_obj['knowledge'] if 'knowledge' in ref_obj else []
        hyp_knowledge = self._remove_duplicate_knowledge(hyp_obj['knowledge']) if 'knowledge' in hyp_obj else []

        if len(ref_knowledge) > 0 or len(hyp_knowledge) > 0:
            self._selection_total += 1.0
            
            tp, fp, fn, exact_matched = self._match_knowledge(ref_knowledge, hyp_knowledge)

            self._selection_tp += float(tp)
            self._selection_fp += float(fp)
            self._selection_fn += float(fn)

            self._selection_exact_matched += float(exact_matched)


    def _compute(self, score_sum):
        if self._detection_tp + self._detection_fp > 0.0:
            score_p = score_sum/(self._detection_tp + self._detection_fp)
        else:
            score_p = 0.0

        if self._detection_tp + self._detection_fn > 0.0:
            score_r = score_sum/(self._detection_tp + self._detection_fn)
        else:
            score_r = 0.0

        if score_p + score_r > 0.0:
            score_f = 2*score_p*score_r/(score_p+score_r)
        else:
            score_f = 0.0

        return (score_p, score_r, score_f)
        
    def scores(self):
        detection_p, detection_r, detection_f = self._compute(self._detection_tp)

        if self._selection_tp + self._selection_fp > 0:
            selection_p = self._selection_tp / (self._selection_tp + self._selection_fp)
        else:
            selection_p = 0.0

        if self._selection_tp + self._selection_fn > 0:
            selection_r = self._selection_tp / (self._selection_tp + self._selection_fn)
        else:
            selection_r = 0.0

        if selection_p + selection_r > 0.0:
            selection_f = 2 * selection_p * selection_r / (selection_p + selection_r)
        else:
            selection_f = 0.0

        selection_em_acc = self._selection_exact_matched / self._selection_total

        bleu_metric = BleuMetric()
        bleu_score = bleu_metric.evaluate_batch(self._pred_responses, self._ref_responses)['bleu'] / 100.0 * self._detection_tp

        meteor_metric = MeteorMetric()
        meteor_score = meteor_metric.evaluate_batch(self._pred_responses, self._ref_responses)['meteor'] * self._detection_tp

        generation_bleu_p, generation_bleu_r, generation_bleu_f = self._compute(bleu_score)
        generation_meteor_p, generation_meteor_r, generation_meteor_f = self._compute(meteor_score)

        generation_rouge_l_p, generation_rouge_l_r, generation_rouge_l_f = self._compute(self._generation_rouge_l)
        generation_rouge_1_p, generation_rouge_1_r, generation_rouge_1_f = self._compute(self._generation_rouge_1)
        generation_rouge_2_p, generation_rouge_2_r, generation_rouge_2_f = self._compute(self._generation_rouge_2)

        scores = {
            'detection': {
                'prec': detection_p,
                'rec': detection_r,
                'f1': detection_f
            },
            'selection': {
                'prec': selection_p,
                'rec': selection_r,
                'f1': selection_f,
                'em_acc': selection_em_acc
            },
            'generation': {
                'bleu': generation_bleu_f,
                'meteor': generation_meteor_f,
                'rouge_1': generation_rouge_1_f,
                'rouge_2': generation_rouge_2_f,
                'rouge_l': generation_rouge_l_f
            }
        }

        return scores
        
def main(argv):
    parser = argparse.ArgumentParser(description='Evaluate the system outputs.')

    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', choices=['train', 'val', 'test'], required=True, help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store', metavar='PATH', required=True,
                        help='Will look for corpus in <dataroot>/<dataset>/...')
    parser.add_argument('--outfile',dest='outfile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing output JSON')
    parser.add_argument('--scorefile',dest='scorefile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing scores')

    args = parser.parse_args()

    try:
        with open(args.outfile, 'r') as f:
            output = json.load(f)
    except FileNotFoundError:
        sys.exit('Output file does not exist at %s' % args.outfile)
    
    data = DatasetWalker(dataroot=args.dataroot, dataset=args.dataset, labels=True)

    metric = Metric()

    for (instance, ref), pred in zip(data, output):
        metric.update(ref, pred)
        
    scores = metric.scores()

    with open(args.scorefile, 'w') as out:
        json.dump(scores, out, indent=2)
    

if __name__ =="__main__":
    main(sys.argv)        
