import logging

import os
import shutil
from collections import Counter
from theano import tensor
from toolz import merge
import numpy
import pickle
from subprocess import Popen, PIPE
import codecs

from blocks.algorithms import (GradientDescent, StepClipping,
                               CompositeRule, Adam, AdaDelta)
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector
from blocks.search import BeamSearch
from blocks_extras.extensions.plot import Plot

from machine_translation.checkpoint import CheckpointNMT, LoadNMT
from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.sampling import BleuValidator, Sampler, SamplingBase
from machine_translation.stream import (get_tr_stream, get_dev_stream,
                                        _ensure_special_tokens)

from mmmt.model import InitialContextDecoder

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)

# TODO: add the multimodal NMT __main__ function here, need to slice up the prototype script


def load_params_and_get_beam_search(exp_config):

    encoder = BidirectionalEncoder(
        exp_config['src_vocab_size'], exp_config['enc_embed'], exp_config['enc_nhids'])

    decoder = InitialContextDecoder(
        exp_config['trg_vocab_size'], exp_config['dec_embed'], exp_config['dec_nhids'],
        exp_config['enc_nhids'] * 2, exp_config['context_dim'])

    # Create Theano variables
    logger.info('Creating theano variables')
    sampling_input = tensor.lmatrix('source')
    sampling_context = tensor.matrix('context_input')

    logger.info("Building sampling model")
    sampling_representation = encoder.apply(
        sampling_input, tensor.ones(sampling_input.shape))

    generated = decoder.generate(sampling_input, sampling_representation, sampling_context)
    _, samples = VariableFilter(
        bricks=[decoder.sequence_generator], name="outputs")(
            ComputationGraph(generated[1]))  # generated[1] is next_outputs

    beam_search = BeamSearch(samples=samples)

    # Set the parameters
    logger.info("Creating Model...")
    model = Model(generated)
    logger.info("Loading parameters from model: {}".format(exp_config['saved_parameters']))

    # load the parameter values from an .npz file
    param_values = LoadNMT.load_parameter_values(exp_config['saved_parameters'])
    LoadNMT.set_model_parameters(model, param_values)

    return beam_search, sampling_input, sampling_context


class NMTPredictor:
    """"Uses a trained NMT model to do prediction"""

    sutils = SamplingBase()

    def __init__(self, exp_config):

        search_vars = load_params_and_get_beam_search(exp_config)
        self.beam_search, self.sampling_input, self.sampling_context = search_vars

        self.exp_config = exp_config
        # how many hyps should be output (only used in file prediction mode)
        self.n_best = exp_config.get('n_best', 1)

        self.source_lang = exp_config.get('source_lang', 'en')
        self.target_lang = exp_config.get('target_lang', 'es')

        tokenize_script = exp_config.get('tokenize_script', None)
        detokenize_script = exp_config.get('detokenize_script', None)
        if tokenize_script is not None and detokenize_script is not None:
            self.tokenizer_cmd = [tokenize_script, '-l', self.source_lang, '-q', '-', '-no-escape', '1']
            self.detokenizer_cmd = [detokenize_script, '-l', self.target_lang, '-q', '-']
        else:
            self.tokenizer_cmd = None
            self.detokenizer_cmd = None

        # this index will get overwritten with the EOS token by _ensure_special_tokens
        # IMPORTANT: the index must be created in the same way it was for training,
        # otherwise the predicted indices will be nonsense
        # Make sure that src_vocab_size and trg_vocab_size are correct in your configuration
        self.src_eos_idx = exp_config['src_vocab_size'] - 1
        self.trg_eos_idx = exp_config['trg_vocab_size'] - 1

        self.unk_idx = exp_config['unk_id']

        # Get vocabularies and inverse indices
        self.src_vocab = _ensure_special_tokens(
            pickle.load(open(exp_config['src_vocab'])), bos_idx=0,
            eos_idx=self.src_eos_idx, unk_idx=self.unk_idx)
        self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        self.trg_vocab = _ensure_special_tokens(
            pickle.load(open(exp_config['trg_vocab'])), bos_idx=0,
            eos_idx=self.trg_eos_idx, unk_idx=self.unk_idx)
        self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}

        self.unk_idx = self.unk_idx

    def map_idx_or_unk(self, sentence, index, unknown_token='<UNK>'):
        if type(sentence) is str:
            sentence = sentence.split()
        return [index.get(w, unknown_token) for w in sentence]

    @staticmethod
    def get_numpy_array(filename):
        import ipdb;ipdb.set_trace()
        return numpy.load(filename)['arr_0']

    # WORKING: add the contexts into prediction
    # Contexts are currently *.npz files (need to fit into memory)
    def predict_files(self, source_input_file, context_input_file, output_file=None):
        tokenize = self.tokenizer_cmd is not None
        detokenize = self.detokenizer_cmd is not None

        if output_file is not None:
            ftrans = codecs.open(output_file, 'wb', encoding='utf8')
        else:
            # cut off the language suffix to make output file name
            output_file = '.'.join(source_input_file.split('.')[:-1]) + '.trans.out'
            ftrans = codecs.open(output_file, 'wb', encoding='utf8')

        logger.info("Started translation, will output {} translations for each segment"
                    .format(self.n_best))
        total_cost = 0.0

        # TODO: the tokenizer throws an error when the input file is opened with encoding='utf8'
        # TODO: why would that error happen?
        # TODO: WORKING: send both source and context to beam search
        with codecs.open(source_input_file) as source_inp:
            source_lines = source_inp.read().strip().split('\n')
            context_features = self.get_numpy_array(context_input_file)
            assert len(source_lines) == len(context_features), 'lens {} and {} do not match'.format(
                len(source_lines), len(context_features)
            )

            for i, inputs in enumerate(zip(source_lines, context_features)):
                source, context = inputs
                logger.info("Translating segment: {}".format(i))

                translations, costs = self.predict_segment(source, context, n_best=self.n_best,
                                                           tokenize=tokenize, detokenize=detokenize)

                # predict_segment returns a list of hyps, we just take the best one
                nbest_translations = translations[:self.n_best]
                nbest_costs = costs[:self.n_best]

                if self.n_best == 1:
                    ftrans.write((nbest_translations[0] + '\n').decode('utf8'))
                    total_cost += nbest_costs[0]
                else:
                    # one blank line to separate each nbest list
                    ftrans.write('\n'.join(nbest_translations) + '\n\n')
                    total_cost += sum(nbest_costs)

                if i != 0 and i % 100 == 0:
                    logger.info("Translated {} lines of test set...".format(i))

        logger.info("Saved translated output to: {}".format(ftrans.name))
        logger.info("Total cost of the test: {}".format(total_cost))
        ftrans.close()

        return output_file

    def predict_segment(self, segment, context, n_best=1, tokenize=False, detokenize=False):
        """
        Do prediction for a single segment, which is a list of token idxs

        Parameters
        ----------
        segment: list[int] : a list of int indexes representing the input sequence in the source language
        n_best: int : how many hypotheses to return (must be <= beam_size)
        tokenize: bool : does the source segment need to be tokenized first?
        detokenize: bool : do the output hypotheses need to be detokenized?

        Returns
        -------
        trans_out: str : the best translation according to beam search
        cost: float : the cost of the best translation

        """

        if tokenize:
            tokenizer = Popen(self.tokenizer_cmd, stdin=PIPE, stdout=PIPE)
            segment, _ = tokenizer.communicate(segment)

        segment = self.map_idx_or_unk(segment, self.src_vocab, self.unk_idx)
        segment += [self.src_eos_idx]

        seq = NMTPredictor.sutils._oov_to_unk(
            segment, self.exp_config['src_vocab_size'], self.unk_idx)

        input_ = numpy.tile(seq, (self.exp_config['beam_size'], 1))
        context_input_ = numpy.tile(context, (self.exp_config['beam_size'], 1))

        # WORKING: change beam search to use the context as well
        # draw sample, checking to ensure we don't get an empty string back
        trans, costs = \
            self.beam_search.search(
                input_values={self.sampling_input: input_,
                              self.sampling_context: context_input_},
                max_length=3*len(seq), eol_symbol=self.trg_eos_idx,
                ignore_first_eol=True)

        # normalize costs according to the sequence lengths
        if self.exp_config['normalized_bleu']:
            lengths = numpy.array([len(s) for s in trans])
            costs = costs / lengths

        best_n_hyps = []
        best_n_costs = []
        best_n_idxs = numpy.argsort(costs)[:n_best]
        for j, idx in enumerate(best_n_idxs):
            try:
                trans_out = trans[idx]
                cost = costs[idx]

                # convert idx to words
                # `line` is a tuple with one item
                try:
                    assert trans_out[-1] == self.trg_eos_idx, 'Target hypothesis should end with the EOS symbol'
                    trans_out = trans_out[:-1]
                    src_in = NMTPredictor.sutils._idx_to_word(segment, self.src_ivocab)
                    trans_out = NMTPredictor.sutils._idx_to_word(trans_out, self.trg_ivocab)
                except AssertionError as e:
                    src_in = NMTPredictor.sutils._idx_to_word(segment, self.src_ivocab)
                    trans_out = NMTPredictor.sutils._idx_to_word(trans_out, self.trg_ivocab)
                    logger.error("ERROR: {} does not end with the EOS symbol".format(trans_out))
                    logger.error("I'm continuing anyway...")
            # TODO: why would this error happen?
            except ValueError:
                logger.info("Can NOT find a translation for line: {}".format(src_in))
                trans_out = '<UNK>'
                cost = 0.

            if detokenize:
                detokenizer = Popen(self.detokenizer_cmd, stdin=PIPE, stdout=PIPE)
                trans_out, _ = detokenizer.communicate(trans_out)
                # strip off the eol symbol
                trans_out = trans_out.strip()

            # TODO: remove this quick hack
            trans_out = trans_out.replace('<UNK>', 'UNK')

            logger.info("Source: {}".format(src_in))
            logger.info("Target Hypothesis: {}".format(trans_out))

            best_n_hyps.append(trans_out)
            best_n_costs.append(cost)

        return best_n_hyps, best_n_costs

