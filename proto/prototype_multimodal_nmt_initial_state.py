# coding: utf-8

from __future__ import print_function
from theano import tensor
from toolz import merge
import os

import numpy

from fuel.datasets import IterableDataset
from fuel.transformers import Merge
from fuel.streams import DataStream
from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

from six.moves import cPickle

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter,
    SequenceGenerator)
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans

from machine_translation.models import MinRiskSequenceGenerator

from mmmt.stream import get_tr_stream_with_context_features, get_dev_stream_with_context_features

from picklable_itertools.extras import equizip


import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create an NMT decoder which has access to image features via the target-side initial state

# IDEA: subclass attention recurrent, and add one more context
# -- could directly push the context onto attention_recurrent.context_names?

# it's ok to add directly to the contexts of the recurrent transition, since that's what will be using them anyway,
# TEST 1: what happens when we directly add the image features to the kwargs that we pass to sequence_generator.cost?
# note this is similar to IMT, since we're trying to modify the decoder initial state

# The kwargs do get passed through to the recurrent transition, so this should work

# AttentionRecurrent gets created in the SequenceGenerator init(), which then calls BaseSequenceGenerator
# Subclass SequenceGenerator


# add one more source for the images

# get the MT datastream in the standard way, then add the new source using Merge
# -- the problem with this is all the operations we do on the stream beforehand

# as long as the arrays fit in memory, we should be able to use iterable dataset

# TODO: move configuration to yaml ASAP
TRAIN_IMAGE_FEATURES = '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/train.npz'
DEV_IMAGE_FEATURES = '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/dev.npz'
TEST_IMAGE_FEATURES = '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/test.npz'

# the prototype config for NMT experiment with images

BASEDIR = '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BERTHA-TEST_Adam_wmt-multimodal_internal_data_dropout'+\
          '0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/'
#best_bleu_model_1455464992_BLEU31.61.npz

exp_config = {
    'src_vocab_size': 20000,
    'trg_vocab_size': 20000,
    'enc_embed': 300,
    'dec_embed': 300,
    'enc_nhids': 800,
    'dec_nhids': 800,
    'src_vocab': os.path.join(BASEDIR, 'vocab.en-de.en.pkl'),
    'trg_vocab': os.path.join(BASEDIR, 'vocab.en-de.de.pkl'),
    'src_data': os.path.join(BASEDIR, 'training_data/train.en.tok.shuf'),
    'trg_data': os.path.join(BASEDIR, 'training_data/train.de.tok.shuf'),
    'unk_id':1,
    # Bleu script that will be used (moses multi-perl in this case)
    'bleu_script': '/home/chris/projects/neural_mt/test_data/sample_experiment/tiny_demo_dataset/multi-bleu.perl',

    # Optimization related ----------------------------------------------------
    # Batch size
    'batch_size': 40,
    # This many batches will be read ahead and sorted
    'sort_k_batches': 10,
    # Optimization step rule
    'step_rule': 'AdaDelta',
    # Gradient clipping threshold
    'step_clipping': 1.,
    # Std of weight initialization
    'weight_scale': 0.01,
    'seq_len': 40,
    # Beam-size
    'beam_size': 10,
    'dropout': 0.3,
    'weight_noise_ff': False,

    # Maximum number of updates
    'finish_after': 1000000,

    # Reload model from files if exist
    'reload': False,

    # Save model after this many updates
    'save_freq': 500,

    # Show samples from model after this many updates
    'sampling_freq': 1000,

    # Show this many samples at each sampling
    'hook_samples': 5,

    # Validate bleu after this many updates
    'bleu_val_freq': 200,
    # Normalize cost according to sequence length after beam-search
    'normalized_bleu': True,
    
    'saveto': '/media/1tb_drive/test_min_risk_model_save',
    'model_save_directory': 'test_image_context_features_model_save',
    
    # Validation set source file
    'val_set': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/dev.en.tok',

    # Validation set gold file
    'val_set_grndtruth': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/dev.de.tok',

    # Print validation output to file
    'output_val_set': True,

    # Validation output file
    'val_set_out': '/media/1tb_drive/test_min_risk_model_save/validation_out.txt',
    'val_burn_in': 5000,

    #     'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BERTHA-TEST_wmt-multimodal_internal_data_dropout0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/best_bleu_model_1455410311_BLEU30.38.npz',

    # NEW PARAMS FOR ADDING CONTEXT FEATURES
    'context_features': '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/train.npz', 
    'val_context_features': '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/dev.npz',
    # the dimensionality of the context features
    'context_dim': 4096
    
    # NEW PARAM FOR MIN RISK
#     'n_samples': 100

}


# setting up the experiment
    
# args = parser.parse_args()
# arg_dict = vars(args)
# configuration_file = arg_dict['exp_config']
# mode = arg_dict['mode']

mode = 'train'
logger.info('Running Neural Machine Translation in mode: {}'.format(mode))
# config_obj = configurations.get_config(configuration_file)
config_obj = exp_config

# add the config file name into config_obj
# config_obj['config_file'] = configuration_file
# logger.info("Model Configuration:\n{}".format(pprint.pformat(config_obj)))

train_stream, source_vocab, target_vocab = get_tr_stream_with_context_features(**config_obj)
dev_stream = get_dev_stream_with_context_features(**config_obj)


# WORKING: make beam search and sampling work nicely with the new context

import logging
import numpy
import operator
import os
import re
import signal
import time
import theano

from blocks.extensions import SimpleExtension
from blocks.search import BeamSearch
from machine_translation.checkpoint import SaveLoadUtils

from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

# this is to let us use all of the sources in the fuel dev stream
# without needing to explicitly filter them
theano.config.on_unused_input = 'warn'


class SamplingBase(object):
    """Utility class for BleuValidator and Sampler."""

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</S>']) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq, vocab_size, unk_idx):
        return [x if x < vocab_size else unk_idx for x in seq]

    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])

    def _initialize_dataset_info(self):
        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
#         if not hasattr(self, 'source_dataset'):
#             self.source_dataset = sources.data_streams[0].dataset
#         if not hasattr(self, 'target_dataset'):
#             self.target_dataset = sources.data_streams[1].dataset
        if not hasattr(self, 'src_vocab'):
            self.src_vocab = self.source_dataset.dictionary
        if not hasattr(self, 'trg_vocab'):
            self.trg_vocab = self.target_dataset.dictionary
        if not hasattr(self, 'src_ivocab'):
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not hasattr(self, 'trg_ivocab'):
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not hasattr(self, 'src_vocab_size'):
            self.src_vocab_size = len(self.src_vocab)


class Sampler(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self, model, data_stream, hook_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, src_vocab_size=None, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.model = model
        self.hook_samples = hook_samples
        self.data_stream = data_stream
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False

        self.sampling_fn = model.get_theano_function()

    def do(self, which_callback, *args):
                # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)

        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]
        batch_size = batch['source'].shape[0]
        hook_samples = min(batch_size, self.hook_samples)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(
            batch_size, hook_samples, replace=False)
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]
        context_batch = batch[self.main_loop.data_stream.sources[-1]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]
        context_ = context_batch[sample_idx, :]


        # Sample
        print()
        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)

            inp = input_[i, :input_length]
            context = context_[i]

            # outputs of self.sampling_fn:
            _1, outputs, _2, _3, costs = (self.sampling_fn(inp[None, :], context[None, :]))
            outputs = outputs.flatten()
            costs = costs.T

            sample_length = self._get_true_length(outputs, self.trg_vocab)

            print("Input : ", self._idx_to_word(input_[i][:input_length],
                                                self.src_ivocab))
            print("Target: ", self._idx_to_word(target_[i][:target_length],
                                                self.trg_ivocab))
            print("Sample: ", self._idx_to_word(outputs[:sample_length],
                                                self.trg_ivocab))
            print("Sample cost: ", costs[:sample_length].sum())
            print()


class BleuValidator(SimpleExtension, SamplingBase):
    # TODO: a lot has been changed in NMT, sync respectively
    # TODO: there is a mistake here when the source and target vocabulary sizes are different -- fix the ""Helpers" section below
    """Implements early stopping based on BLEU score."""

    def __init__(self, source_sentence, initial_state_context, samples, model, data_stream,
                 config, src_vocab=None, trg_vocab=None, n_best=1, track_n_models=1,
                 normalize=True, **kwargs):
        # TODO: change config structure
        super(BleuValidator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.initial_context = initial_state_context
        
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.verbose = config.get('val_set_out', None)

        # Helpers
        self.best_models = []
        self.val_bleu_curve = []
        self.beam_search = BeamSearch(samples=samples)
        self.multibleu_cmd = ['perl', self.config['bleu_script'],
                              self.config['val_set_grndtruth'], '<']

        # Create saving directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

        if self.config['reload']:
            try:
                bleu_score = numpy.load(os.path.join(self.config['saveto'],
                                        'val_bleu_scores.npz'))
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()

                # Track n best previous bleu scores
                for i, bleu in enumerate(
                        sorted(self.val_bleu_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(bleu))
                logger.info("BleuScores Reloaded")
            except:
                logger.info("BleuScores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <=                 self.config['val_burn_in']:
            return

        # Evaluate the model
        bleu_score = self._evaluate_model()
        # add an entry to the log
        self.main_loop.log.current_row['validation_set_bleu_score'] = bleu_score
        # save if necessary
        self._save_model(bleu_score)

    def _evaluate_model(self):
        # Set in the superclass -- SamplingBase
        if not hasattr(self, 'target_dataset'):
            self._initialize_dataset_info()
        
#         self.unk_sym = self.target_dataset.unk_token
#         self.eos_sym = self.target_dataset.eos_token
        
        self.unk_sym = '<UNK>'
        self.eos_sym = '</S>'
        self.unk_idx = self.trg_vocab[self.unk_sym]
        self.eos_idx = self.trg_vocab[self.eos_sym]

        logger.info("Started Validation: ")
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0

        if self.verbose:
            ftrans = open(self.config['val_set_out'], 'w')

        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = self._oov_to_unk(
                line[0], self.config['src_vocab_size'], self.unk_idx)
            initial_state_context = line[-1]
            
            input_ = numpy.tile(seq, (self.config['beam_size'], 1))
            context_input_ = numpy.tile(initial_state_context, (self.config['beam_size'], 1))

            # draw sample, checking to ensure we don't get an empty string back
            # beam search param names come from WHERE??
            trans, costs = self.beam_search.search(
                    input_values={self.source_sentence: input_,
                                  self.initial_context: context_input_},
                    max_length=3*len(seq), eol_symbol=self.eos_idx,
                    ignore_first_eol=True)

            # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]

                    # convert idx to words
                    trans_out = self._idx_to_word(trans_out, self.trg_ivocab)

                except ValueError:
                    logger.info(
                        "Can NOT find a translation for line: {}".format(i+1))
                    trans_out = '<UNK>'

                if j == 0:
                    # Write to subprocess and file if it exists
                    print(trans_out, file=mb_subprocess.stdin)
                    if self.verbose:
                        print(trans_out, file=ftrans)

            if i != 0 and i % 100 == 0:
                logger.info(
                    "Translated {} lines of validation set...".format(i))

            mb_subprocess.stdin.flush()

        logger.info("Total cost of the validation: {}".format(total_cost))
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        logger.info(stdout)
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logger.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        logger.info(bleu_score)
        mb_subprocess.terminate()

        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score):
        if self._is_valid_to_save(bleu_score):
            model = ModelInfo(bleu_score, self.config['saveto'])

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('bleu_score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))

            SaveLoadUtils.save_parameter_values(self.main_loop.model.get_parameter_values(), model.path)
            numpy.savez(
                os.path.join(self.config['saveto'], 'val_bleu_scores.npz'),
                bleu_scores=self.val_bleu_curve)
            signal.signal(signal.SIGINT, s)


class ModelInfo:
    """Utility class to keep track of evaluated models."""

    def __init__(self, bleu_score, path=None):
        self.bleu_score = bleu_score
        self.path = self._generate_path(path)

    def _generate_path(self, path):
        gen_path = os.path.join(
            path, 'best_bleu_model_%d_BLEU%.2f.npz' %
            (int(time.time()), self.bleu_score) if path else None)
        return gen_path

# END SAMPLING AND VALIDATION

# MODEL CREATION AND TRAINING

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
# we reimplement sampling for the context NMT
# from machine_translation.sampling import BleuValidator, Sampler, SamplingBase
from machine_translation.stream import (get_tr_stream, get_dev_stream,
                                        _ensure_special_tokens)

from mmmt.model import InitialContextSequenceGenerator, InitialContextDecoder

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)


def main(config, tr_stream, dev_stream, use_bokeh=False):

    # Create Theano variables
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')
    initial_context = tensor.matrix('initial_context')

    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])

    decoder = InitialContextDecoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2, config['context_dim'])
    
    cost = decoder.cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask, initial_context)

    cost.name = 'decoder_cost'

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # Initialize model
    logger.info('Initializing model')
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(
        config['weight_scale'])
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()

    # apply dropout for regularization
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        # this is the probability of dropping out, so you probably want to make it <=0.5
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

    # Apply weight noise for regularization
    if config['weight_noise_ff'] > 0.0:
        logger.info('Applying weight noise to ff layers')
        enc_params = Selector(encoder.lookup).get_parameters().values()
        enc_params += Selector(encoder.fwd_fork).get_parameters().values()
        enc_params += Selector(encoder.back_fork).get_parameters().values()
        dec_params = Selector(
            decoder.sequence_generator.readout).get_parameters().values()
        dec_params += Selector(
            decoder.sequence_generator.fork).get_parameters().values()
        dec_params += Selector(decoder.transition.initial_transformer).get_parameters().values()
        cg = apply_noise(cg, enc_params+dec_params, config['weight_noise_ff'])

    # TODO: weight noise for recurrent params isn't currently implemented -- see config['weight_noise_rec']
    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
    logger.info("Total number of parameters: {}".format(len(shapes)))

    # Print parameter names
    enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                               Selector(decoder).get_parameters())
    logger.info("Parameter names: ")
    for name, value in enc_dec_param_dict.items():
        logger.info('    {:15}: {}'.format(value.get_value().shape, name))
    logger.info("Total number of parameters: {}"
                .format(len(enc_dec_param_dict)))

    # Set up training model
    logger.info("Building model")
    training_model = Model(cost)

    # create the training directory, and copy this config there if directory doesn't exist
    if not os.path.isdir(config['saveto']):
        os.makedirs(config['saveto'])
        shutil.copy(config['config_file'], config['saveto'])

    # Set extensions

    # TODO: add checking for existing model and loading
    logger.info("Initializing extensions")
    extensions = [
        FinishAfter(after_n_batches=config['finish_after']),
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
        CheckpointNMT(config['saveto'],
                      every_n_batches=config['save_freq'])
    ]

    # Create the theano variables that we need for the sampling graph
    sampling_input = tensor.lmatrix('input')
    sampling_context = tensor.matrix('context_input')
    
    # WORKING: change this part to account for the new initial context for decoder
    # Set up beam search and sampling computation graphs if necessary
    if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
        logger.info("Building sampling model")
        sampling_representation = encoder.apply(
            sampling_input, tensor.ones(sampling_input.shape))
        
        generated = decoder.generate(sampling_input, sampling_representation, sampling_context)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs

    # Add sampling
    # TODO: currently commented because we need to modify the sampler to use the contexts
    if config['hook_samples'] >= 1:
        logger.info("Building sampler")
        extensions.append(
            Sampler(model=search_model, data_stream=tr_stream,
                    hook_samples=config['hook_samples'],
                    every_n_batches=config['sampling_freq'],
                    src_vocab=source_vocab,
                    trg_vocab=target_vocab,
                    src_vocab_size=config['src_vocab_size'],
                   ))


    # TODO: add sampling_context to BleuValidator and Sampler
    # Add early stopping based on bleu
    if config['bleu_script'] is not None:
        logger.info("Building bleu validator")
        extensions.append(
            BleuValidator(sampling_input, sampling_context, samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          src_vocab=source_vocab,
                          trg_vocab=target_vocab,
                          normalize=config['normalized_bleu'],
                          every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot(config['model_save_directory'], channels=[['decoder_cost', 'validation_set_bleu_score']],
                 every_n_batches=10))

    # Set up training algorithm
    logger.info("Initializing training algorithm")
    # if there is dropout or random noise, we need to use the output of the modified graph
    if config['dropout'] < 1.0 or config['weight_noise_ff'] > 0.0:
        algorithm = GradientDescent(
            cost=cg.outputs[0], parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()])
        )
    else:
        algorithm = GradientDescent(
            cost=cost, parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()])
        )

    # enrich the logged information
    extensions.append(
        Timing(every_n_batches=100)
    )

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=training_model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )

    # Train!
    main_loop.run()


# In[ ]:

main(exp_config, train_stream, dev_stream, use_bokeh=True)


