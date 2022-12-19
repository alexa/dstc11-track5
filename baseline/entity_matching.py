import argparse
import os, json
import re
from difflib import SequenceMatcher as SM
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import string
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


def fuzzy_extract(qs, ls, threshold):
    """ Match an entity name (qs) with an utterance (ls) using fuzzy matching """
    qs_length = len(qs.split())
    max_sim_val = 0
    max_sim_string = u""

    for ngram in ngrams(ls.split(), qs_length + int(.2 * qs_length)):
        ls_ngram = u" ".join(ngram)
        similarity = SM(None, ls_ngram, qs).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = ls_ngram

    if max_sim_val > threshold:
        return max_sim_string, max_sim_val
    else:
        return None, max_sim_val


def check_substring_exist(qs, ls):
    """ Check if an entity mention (qs) is in an utterance (ls) """
    if len(qs.split()) > 1:
        return qs in ls
    else:
        return qs in word_tokenize(ls)


def entity_matching(entity, log):
    """ Match a single entity with a dialogue history """
    result = None
    max_fuzzy_score = 0

    for turn_id, obj in enumerate(log):
        flag = False

        utter = obj['text'].lower()
        if check_substring_exist(entity, utter):
            flag = True

        entity_names = all_entity_names_norm.get(entity, []) + [entity]
        for entity_name in entity_names:
            if check_substring_exist(entity_name, utter):
                flag = True

        # if substring exist, fuzzy_match_score = 1, otherwise fuzzy_match_score < 1
        for entity_name in entity_names:
            fuzzy_match_res, fuzzy_match_score = fuzzy_extract(entity_name, utter, 0.95)
            max_fuzzy_score = max(max_fuzzy_score, fuzzy_match_score)
            if fuzzy_match_res is not None:
                flag = True

        if flag is True:
            result = turn_id

    return result, max_fuzzy_score


def run_entity_matching(args):
    """ Run entity matching for a single instance """
    idx_, (log, label) = args
    if label['target'] is False:
        return None, None

    matching_res_ls = set()
    entity_scores = []
    for entity_tup in all_entity_names:
        entity_domain, entity_id, entity_name = entity_tup
        match_res, match_score = entity_matching(entity_name.lower(), log)
        entity_scores.append((entity_tup, match_score))
        if match_res is not None:
            matching_res_ls.add((entity_domain, entity_id, entity_name, match_res))
    matching_res_ls = sorted(list(matching_res_ls), key=lambda x: x[-1])

    result = []
    if len(matching_res_ls) > 0:
        latest_turn_w_entity = matching_res_ls[-1][-1]
        for entity_domain, entity_id, entity_name, turn_id in matching_res_ls:
            if turn_id == latest_turn_w_entity:
                result.append({'domain': entity_domain, 'entity_id': int(entity_id), 'entity_name': entity_name})
    else:
        entity_scores = sorted(entity_scores, key=lambda x: -x[1])
        entity_tup, match_score = entity_scores[0]
        entity_domain, entity_id, entity_name = entity_tup
        result.append({'domain': entity_domain, 'entity_id': int(entity_id), 'entity_name': entity_name})

    pred_entity_set = set([str(r['entity_id']) for r in result])
    return result, pred_entity_set


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--norm_dict", type=str, default="baseline/resources/entity_mapping.json",
                        help="Path to the normalization dictionary")
    parser.add_argument("--labels_file", type=str, default="data/train/labels.json",
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                             "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")
    parser.add_argument("--output_file", type=str, default="",
                        help="Predictions will be written to this file.")
    args = parser.parse_args()

    # read data
    logs_file = os.path.join(args.dataroot, args.eval_dataset, 'logs.json')
    knowledge_file = os.path.join(args.dataroot, 'knowledge.json')
    labels_file = args.labels_file
    with open(logs_file, 'r') as f:
        logs = json.load(f)
    with open(knowledge_file, 'r') as f:
        knowledges = json.load(f)
    with open(labels_file, 'r') as f:
        labels = json.load(f)

    # load entities and normalized entity mentions
    all_entity_names = []
    for domain, domain_dict in knowledges.items():
        if domain in ['train', 'taxi']:
            continue
        for doc_id, docs in domain_dict.items():
            all_entity_names.append((domain, doc_id, docs['name']))

    with open(args.norm_dict, 'r') as fr:
        all_entity_names_norm = json.load(fr)

    # match entities in parallel
    results = []
    pred_entity_sets = []
    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(logs), desc='entity matching') as pbar:
            for result, pred_entity_set in p.imap(run_entity_matching, enumerate(zip(logs, labels))):
                if result is not None:
                    results.append(result)
                    pred_entity_sets.append(pred_entity_set)
                else:
                    results.append(None)
                pbar.update()

    # write the matched entities in output file
    for label, result in zip(labels, results):
        if label['target'] is False:
            assert result is None
        else:
            label['knowledge'] = result
    with open(args.output_file, 'w') as fw:
        json.dump(labels, fw, indent=4)
