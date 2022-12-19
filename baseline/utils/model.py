import copy

import torch
import logging

logger = logging.getLogger(__name__)


def run_batch_detection_train(args, model, batch, **kwargs):
    """ Run batch knowledge turn detection during training time """
    cls_loss, cls_logits, labels = run_batch_detection_eval(args, model, batch, **kwargs)
    yield cls_loss, cls_logits, None


def run_batch_detection_eval(args, model, batch, **kwargs):
    """ Run batch knowledge turn detection during evaluation time """
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, attention_mask, labels = batch
    model_outputs = model(
        input_ids=input_ids,
        token_type_ids=None if model.base_model_prefix in ['roberta'] else token_type_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    cls_loss = model_outputs.loss
    cls_logits = model_outputs.logits
    return cls_loss, cls_logits, labels


def run_batch_selection_train(args, model, batch, **kwargs):
    """ Run batch knowledge selection during training time """
    candidates_per_forward = args.max_candidates_per_forward_train
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, attention_mask, labels = batch
    for index in range(0, input_ids.size(0), candidates_per_forward):
        model_outputs = model(
            input_ids=input_ids[index:index + candidates_per_forward],
            token_type_ids=None if model.base_model_prefix in ['roberta'] else
                           token_type_ids[index:index + candidates_per_forward],
            attention_mask=attention_mask[index:index + candidates_per_forward],
            labels=labels[index:index + candidates_per_forward],
        )
        loss, logits = model_outputs[0], model_outputs[1]
        yield loss, logits, None


def run_batch_selection_eval(args, model, batch, **kwargs):
    """ Run batch knowledge selection during evaluation time """
    # return: loss, logits, labels
    candidates_per_forward = args.max_candidates_per_forward_eval
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, attention_mask, labels = batch
    original_labels = copy.deepcopy(labels)

    all_logits = []
    eval_loss = 0
    for index in range(0, input_ids.size(0), candidates_per_forward):
        model_outputs = model(
            input_ids=input_ids[index:index + candidates_per_forward],
            token_type_ids=None if model.base_model_prefix in ['roberta'] else
                           token_type_ids[index:index + candidates_per_forward],
            attention_mask=attention_mask[index:index + candidates_per_forward],
            labels=labels[index:index + candidates_per_forward]
        )
        eval_loss += model_outputs.loss * len(input_ids[index:index + candidates_per_forward])
        logits = model_outputs.logits
        all_logits.append(logits.detach())
    all_logits = torch.cat(all_logits, dim=0)
    return eval_loss, all_logits, original_labels


def run_batch_generation_train(args, model, batch, **kwargs):
    """ Run batch generation during training time """
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch[:4])
    input_ids, attention_mask, lm_labels = batch
    model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels)
    loss = model_outputs[0]
    lm_logits = model_outputs[1]
    yield loss, lm_logits, torch.tensor([])


def run_batch_generation_eval(args, model, batch, **kwargs):
    """ Run batch generation during evaluation time """
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch[:4])
    input_ids, attention_mask, lm_labels = batch
    model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels)
    loss = model_outputs[0]
    lm_logits = model_outputs[1]
    return loss, lm_logits, torch.tensor([])


def run_batch_generation_sample(args, model, tokenizer, batch, dataset):
    """ Run batch generation during test time
        Responses are decoded using beam search + sampling
    """
    current_output = []

    example = batch[0]
    knowledge, history = example["knowledge"], example["history"]
    response_text = example["response_text"]
    dialog_id = example["dialog_id"]

    instance, sequence = dataset.build_input_from_segments(
        knowledge, history, current_output
    )

    input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
    current_output = model.generate(input_ids=input_ids, num_beams=args.num_beams,
                                    min_length=args.min_length, max_length=args.max_length,
                                    eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                                    pad_token_id=tokenizer.pad_token_id, do_sample=args.do_sample, num_return_sequences=1)

    return current_output, response_text, dialog_id
