# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from collections import OrderedDict

import torch

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.ctc import CtcCriterion
from fairseq.criterions.label_smoothed_cross_entropy_with_rdrop import (
    RdropLabelSmoothedCrossEntropyCriterion,
    RdropLabelSmoothedCrossEntropyCriterionConfig,
    duplicate_input,
)
from fairseq.criterions.tacotron2_loss import (
    Tacotron2Criterion,
    Tacotron2CriterionConfig,
)

from fairseq.criterions.speech_to_speech_criterion import (
    MultitaskCriterion, 

)

logger = logging.getLogger(__name__)


@register_criterion(
    "speech_to_unit_v2", dataclass=RdropLabelSmoothedCrossEntropyCriterionConfig
)
class SpeechToUnitMultitaskTaskCriterion(
    RdropLabelSmoothedCrossEntropyCriterion, MultitaskCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        rdrop_alpha=0.0,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy,
            rdrop_alpha,
        )
        MultitaskCriterion.__init__(self, task.multitask_tasks, rdrop_alpha)

    def forward(self, model, sample, reduce=True):
        net_input_concat = {
            "src_tokens": sample["net_input"]["src_tokens"],
            "src_lengths": sample["net_input"]["src_lengths"],
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "tgt_speaker": sample["net_input"].get("tgt_speaker", None),
            "return_all_hiddens": True,
        }
        for item in sample["net_input"]:
            if item not in net_input_concat:
                net_input_concat[item] = sample["net_input"][item]

        if self.rdrop_alpha > 0 or self.rdrop_alpha_mtl > 0:
            net_input_concat = duplicate_input(net_input_concat)

        net_output, extra = model(**net_input_concat)
        loss, nll_loss, rdrop_kl_loss = self.compute_loss(
            model, [net_output], sample, reduce=reduce
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, [net_output], sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        if self.rdrop_alpha > 0:
            logging_output["rdrop_kl_loss"] = utils.item(rdrop_kl_loss.data)

        if len(self.multitask_criterion) == 0:
            return loss, sample_size, logging_output

        # multitask
        multitask_loss, multitask_log = self.get_multitask_loss(model, sample, extra)
        loss += multitask_loss
        logging_output["multitask"] = multitask_log

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)

        # inference metrics
        if "targ_frames" in logging_outputs[0]:
            n = sum(log.get("norm_frames", 0) for log in logging_outputs)
            for key, new_key in [
                ("mcd_loss", "mcd_loss"),
                ("pred_frames", "pred_ratio"),
                ("nins", "ins_rate"),
                ("ndel", "del_rate"),
            ]:
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(new_key, val / n, n, round=3)

        if "multitask" not in logging_outputs[0]:
            return

        MultitaskCriterion.reduce_metrics(logging_outputs)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
