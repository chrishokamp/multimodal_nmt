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
from mmmt.sample import Sampler, BleuValidator

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
# TRAIN_IMAGE_FEATURES = '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/train.npz'
# DEV_IMAGE_FEATURES = '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/dev.npz'
# TEST_IMAGE_FEATURES = '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/test.npz'

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


