# coding: utf-8

import os
import codecs
import subprocess
from pprint import pprint
from subprocess import Popen, PIPE, STDOUT

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

import numpy
import codecs
import tempfile
import cPickle
import copy
from collections import OrderedDict
import itertools
from theano import tensor
import shutil
import yaml

from fuel.datasets import IterableDataset
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

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
from blocks.roles import WEIGHT
from blocks.search import BeamSearch
from blocks_extras.extensions.plot import Plot

from machine_translation.checkpoint import CheckpointNMT, LoadNMT
from machine_translation.model import BidirectionalEncoder, Decoder

from machine_translation.stream import (_ensure_special_tokens,
                                        get_textfile_stream, _too_long, _length, PaddingWithEOS,
                                        _oov_to_unk, FlattenSamples)

# TODO: these need to be implemented for multimodal min_risk as well
from machine_translation.evaluation import sentence_level_bleu, sentence_level_meteor

from mmmt.sample import SampleFunc, BleuValidator, MeteorValidator
from mmmt.model import GRUInitialStateWithInitialStateSumContext, GRUInitialStateWithInitialStateConcatContext, InitialContextDecoder
from mmmt.stream import MMMTSampleStreamTransformer, CopySourceAndContextNTimes, get_dev_stream_with_context_features


try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

# build the training and sampling graphs for minimum risk training
# Intialize the MTSampleStreamTransformer with the sampling function

# load a model that's already trained, and start tuning it with minimum-risk
# mock-up training using the blocks main loop

# TODO: Integrate configuration so min-risk training is a single line in the config file
# TODO: this requires handling both the data stream and the Model cost function


# BASEDIR = '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BERTHA-TEST_Adam_wmt-multimodal_internal_data_dropout'+\
#           '0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/'
BASEDIR = '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed'
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
    'src_data': os.path.join(BASEDIR, 'train.en.tok.shuf'),
    'trg_data': os.path.join(BASEDIR, 'train.de.tok.shuf'),
    'unk_id':1,
    # Bleu script that will be used (moses multi-perl in this case)
    # 'bleu_script': os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                             '../test_data/sample_experiment/tiny_demo_dataset/multi-bleu.perl'),

    # Optimization related ----------------------------------------------------
    # Batch size
    'batch_size': 15,
    # This many batches will be read ahead and sorted
    'sort_k_batches': 10,
    # Optimization step rule
    'step_rule': 'AdaDelta',
    # Gradient clipping threshold
    'step_clipping': 1.,
    # Std of weight initialization
    'weight_scale': 0.1,
    'seq_len': 40,
    # Beam-size
    'beam_size': 10,
    # dropout
    'dropout': 0.5,

    # l2_reg
    'l2_regularization': True,
    'l2_regularization_alpha': 0.0001,

    # Maximum number of updates
    'finish_after': 10000,

    # Reload model from files if exist
    'reload': True,

    # Save model after this many updates
    'save_freq': 500,

    # Show samples from model after this many updates
    'sampling_freq': 1000,

    # Show this many samples at each sampling
    'hook_samples': 5,

    # Validate bleu after this many updates
    'bleu_val_freq': 20,

    # Normalize cost according to sequence length after beam-search
    'normalized_bleu': True,

    # Validation set source file
    # 'val_set': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/dev.en.tok',
    'val_set': '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational/dev.en.tok',


    # Validation set gold file
    # 'val_set_grndtruth': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/dev.de.tok',
    'val_set_grndtruth': '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational/dev.de.tok',

    # Print validation output to file
    'output_val_set': True,

    # Validation output file
    'val_set_out': '/media/1tb_drive/test_min_risk_model_save/validation_out.txt',
    'val_burn_in': 0,

    'source_lang': 'en',
    'target_lang': 'de',

    # NEW PARAM FOR MIN RISK
    'n_samples': 25,

    'min_risk_score_func': 'bleu',

    'target_transition': 'GRUInitialStateWithInitialStateSumContext',
    # 'target_transition': 'GRUInitialStateWithInitialStateConcatContext',

    'meteor_directory': '/home/chris/programs/meteor-1.5',
    # 'brick_delimiter': '-'

    # Multimodal
    # NEW PARAMS FOR ADDING CONTEXT FEATURES
    'context_features': '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/train.npz',
    # 'val_context_features': '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/dev.npz',
    'val_context_features': '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational/dev.npz',
    'context_dim': 4096,

    # 'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BERTHA-TEST_wmt-multimodal_internal_data_dropout0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/best_bleu_model_1455410311_BLEU30.38.npz',
    #'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BASELINE_WITH_REGULARIZATION0.0001_WEIGHT_SCALE_0.1_wmt-multimodal_internal_data_dropout0.5_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/best_bleu_model_1461853863_BLEU32.80.npz',
    # 'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BASELINE_METEOR_CHECKPOINT_WITH_REGULARIZATION0.0001_WEIGHT_SCALE_0.1_wmt-multimodal_internal_data_dropout0.5_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/best_model_1462312300_METEOR0.52.npz',
    # 'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/wmt16-multimodal_SUM_INITIAL_STATE_BASELINE_SANITY_internal_data_dropout0.5_src_vocab20000_trg_vocab20000_emb300_rec800_batch40/best_model_1462352198_METEOR0.51.npz',
    # 'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/pilot_multimodal_min_risk/best_model_1462409560_METEOR0.48.npz',
    #'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/pilot_multimodal_min_risk/best_model_CONCAT_INITIAL_METEOR0.52.npz',
    'saved_parameters': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/pilot_multimodal_min_risk/best_model_TRUE_SUM_METEOR0.51.npz',

    'saveto': '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/MIN-RISK-FINAL-BEST-4-L2REG_DROPOUT0.5-n_samples30-batch10',
    'model_save_directory': 'MIN-RISK-FINAL-BEST-4-L2REG_DROPOUT0.5-n_samples50-batch15',
}

# TODO: load config from .yaml file -- this will let us be dynamic with the run names, which is a big advantage

# WORKING: change the model to use initial image feature context
def get_sampling_model_and_input(exp_config):
    # Create Theano variables
    encoder = BidirectionalEncoder(
        exp_config['src_vocab_size'], exp_config['enc_embed'], exp_config['enc_nhids'])

    # TODO: modify InitialContextDecoder to support expected cost
    # Note: the 'min_risk' kwarg tells the decoder which sequence_generator and cost_function to use

    transition = eval(exp_config['target_transition'])

    decoder = InitialContextDecoder(
        exp_config['trg_vocab_size'], exp_config['dec_embed'], exp_config['dec_nhids'],
        exp_config['enc_nhids'] * 2, exp_config['context_dim'], transition,
        loss_function='min_risk')

    # Create Theano variables
    logger.info('Creating theano variables')
    sampling_source_input = tensor.lmatrix('source')
    sampling_context_input = tensor.matrix('context')

    # Get beam search
    logger.info("Building sampling model")
    sampling_source_representation = encoder.apply(
        sampling_source_input, tensor.ones(sampling_source_input.shape))

    generated = decoder.generate(sampling_source_input,
                                 sampling_source_representation,
                                 sampling_context_input)

    # build the model that will let us get a theano function from the sampling graph
    logger.info("Creating Sampling Model...")
    sampling_model = Model(generated)

    # TODO: update clients with sampling_context_input
    return sampling_model, sampling_source_input, sampling_context_input, encoder, decoder

sample_model, theano_sampling_source_input, theano_sampling_context_input, train_encoder, train_decoder = \
    get_sampling_model_and_input(exp_config)

trg_vocab = cPickle.load(open(exp_config['trg_vocab']))
trg_vocab_size = exp_config['trg_vocab_size'] - 1
src_vocab = cPickle.load(open(exp_config['src_vocab']))
src_vocab_size = exp_config['src_vocab_size'] - 1

src_vocab = _ensure_special_tokens(src_vocab, bos_idx=0,
                                   eos_idx=src_vocab_size, unk_idx=exp_config['unk_id'])
trg_vocab = _ensure_special_tokens(trg_vocab, bos_idx=0,
                                   eos_idx=trg_vocab_size, unk_idx=exp_config['unk_id'])

theano_sample_func = sample_model.get_theano_function()
sampling_func = SampleFunc(theano_sample_func, trg_vocab)

src_stream = get_textfile_stream(source_file=exp_config['src_data'], src_vocab=exp_config['src_vocab'],
                                         src_vocab_size=exp_config['src_vocab_size'])

trg_stream = get_textfile_stream(source_file=exp_config['trg_data'], src_vocab=exp_config['trg_vocab'],
                                         src_vocab_size=exp_config['trg_vocab_size'])


# text file stream
training_stream = Merge([src_stream,
                         trg_stream],
                         ('source', 'target'))

# add in the context features
train_features = numpy.load(exp_config['context_features'])['arr_0']
train_feature_dataset = IterableDataset(train_features)
train_image_stream = DataStream(train_feature_dataset)

# TODO: get the stream for initial context features
training_stream = Merge([training_stream, train_image_stream],
                        ('source', 'target', 'initial_context'))



# Filter sequences that are too long
# TODO: the logic would need to be modified to use this filter w/ context features
# training_stream = Filter(training_stream,
#                          predicate=_too_long(seq_len=exp_config['seq_len']))

# TODO: configure min-risk score func from the yaml config
# BLEU

# METEOR
trg_ivocab = {v:k for k,v in trg_vocab.items()}

min_risk_score_func = exp_config.get('min_risk_score_func', 'bleu')

if min_risk_score_func == 'meteor':
    sampling_transformer = MMMTSampleStreamTransformer(sampling_func,
                                                       sentence_level_meteor,
                                                       num_samples=exp_config['n_samples'],
                                                       trg_ivocab=trg_ivocab,
                                                       lang=exp_config['target_lang'],
                                                       meteor_directory=exp_config['meteor_directory']
                                                      )
else:
    sampling_transformer = MMMTSampleStreamTransformer(sampling_func,
                                                       sentence_level_bleu,
                                                       num_samples=exp_config['n_samples'])


training_stream = Mapping(training_stream, sampling_transformer, add_sources=('samples', 'scores'))


# Build a batched version of stream to read k batches ahead
training_stream = Batch(training_stream,
                        iteration_scheme=ConstantScheme(
                        exp_config['batch_size']*exp_config['sort_k_batches']))

# TODO: add read-ahead shuffling Mapping similar to SortMapping
# Sort all samples in the read-ahead batch
training_stream = Mapping(training_stream, SortMapping(_length))

# Convert it into a stream again
training_stream = Unpack(training_stream)

# Construct batches from the stream with specified batch size
training_stream = Batch(
    training_stream, iteration_scheme=ConstantScheme(exp_config['batch_size']))

# Pad sequences that are short
# IDEA: add a transformer which flattens the target samples before we add the mask
flat_sample_stream = FlattenSamples(training_stream)

expanded_source_stream = CopySourceAndContextNTimes(flat_sample_stream, n_samples=exp_config['n_samples'])

# Note: some sources can be excluded from the padding Op, but since blocks matches sources with input variable
# Note: names, it's not critical
# TODO: add mask sources?
masked_stream = PaddingWithEOS(
    expanded_source_stream, [exp_config['src_vocab_size'] - 1, exp_config['trg_vocab_size'] - 1])

# create the model for training
# TODO: implement the expected_cost multimodal decoder
def create_model(encoder, decoder):

    # Create Theano variables
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')

#     target_samples = tensor.tensor3('samples').astype('int64')
#     target_samples_mask = tensor.tensor3('target_samples_mask').astype('int64')
    samples = tensor.lmatrix('samples')
    samples_mask = tensor.matrix('samples_mask')

    initial_context = tensor.matrix('initial_context')

    # scores is (batch, samples)
    scores = tensor.matrix('scores')
    # We don't need a scores mask because there should be the same number of scores for each instance
    # num samples is a hyperparameter of the model

    # the name is important to make sure pre-trained params get loaded correctly
#     decoder.name = 'decoder'

    # This is the part that is different for the MinimumRiskSequenceGenerator

    cost = decoder.expected_cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, samples, samples_mask, scores,
        initial_state_context=initial_context,
        smoothing_constant=0.005
    )

    return cost


def main(model, cost, config, tr_stream, dev_stream, use_bokeh=False):

    # Set the parameters from a trained models (.npz file)
    logger.info("Loading parameters from model: {}".format(exp_config['saved_parameters']))
    # Note the brick delimeter='-' is here for legacy reasons because blocks changed the serialization API
    param_values = LoadNMT.load_parameter_values(exp_config['saved_parameters'], brick_delimiter=exp_config.get('brick_delimiter', None))
    LoadNMT.set_model_parameters(model, param_values)

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # GRAPH TRANSFORMATIONS FOR BETTER TRAINING
    if config.get('l2_regularization', False) is True:
        l2_reg_alpha = config['l2_regularization_alpha']
        logger.info('Applying l2 regularization with alpha={}'.format(l2_reg_alpha))
        model_weights = VariableFilter(roles=[WEIGHT])(cg.variables)

        for W in model_weights:
            cost = cost + (l2_reg_alpha * (W ** 2).sum())

        # why do we need to rename the cost variable? Where did the original name come from?
        cost.name = 'decoder_cost_cost'

    cg = ComputationGraph(cost)

    # apply dropout for regularization
    # Note dropout variables are hard-coded here
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        # this is the probability of dropping out, so you probably want to make it <=0.5
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

    # create the training directory, and copy this config there if directory doesn't exist
    if not os.path.isdir(config['saveto']):
        os.makedirs(config['saveto'])
        # TODO: mv the actual config file once we switch to .yaml for min-risk
        # shutil.copy(config['config_file'], config['saveto'])
        # shutil.copy(config['config_file'], config['saveto'])

        # TODO: this breaks when we directly reference a class in the config obj instead of using reflection
        with codecs.open(os.path.join(config['saveto'], 'config.yaml'), 'w', encoding='utf8') as yaml_out:
            yaml_out.write(yaml.dump(config))

    # Set extensions
    logger.info("Initializing extensions")
    extensions = [
        FinishAfter(after_n_batches=config['finish_after']),
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
         CheckpointNMT(config['saveto'],
                       every_n_batches=config['save_freq'])
    ]

    # Set up beam search and sampling computation graphs if necessary
    # TODO: change the if statement here
    if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
        logger.info("Building sampling model")
        sampling_representation = train_encoder.apply(
            theano_sampling_source_input, tensor.ones(theano_sampling_source_input.shape))
        # TODO: the generated output actually contains several more values, ipdb to see what they are
        generated = train_decoder.generate(theano_sampling_source_input, sampling_representation,
                                           theano_sampling_context_input)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[train_decoder.sequence_generator], name="outputs")(
            ComputationGraph(generated[1]))  # generated[1] is next_outputs

    # Add sampling -- TODO: sampling is broken for min-risk
    #if config['hook_samples'] >= 1:
    #    logger.info("Building sampler")
    #    extensions.append(
    #        Sampler(model=search_model, data_stream=tr_stream,
    #                hook_samples=config['hook_samples'],
    #                every_n_batches=config['sampling_freq'],
    #                src_vocab_size=config['src_vocab_size']))

    # Add early stopping based on bleu
    # TODO: use multimodal meteor and BLEU validator
    # Add early stopping based on bleu
    if config.get('bleu_script', None) is not None:
        logger.info("Building bleu validator")
        extensions.append(
            BleuValidator(theano_sampling_source_input, theano_sampling_context_input,
                          samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          src_vocab=src_vocab,
                          trg_vocab=trg_vocab,
                          normalize=config['normalized_bleu'],
                          every_n_batches=config['bleu_val_freq']))


    # Add early stopping based on Meteor
    if config.get('meteor_directory', None) is not None:
        logger.info("Building meteor validator")
        extensions.append(
            MeteorValidator(theano_sampling_source_input, theano_sampling_context_input,
                            samples=samples,
                            config=config,
                            model=search_model, data_stream=dev_stream,
                            src_vocab=src_vocab,
                            trg_vocab=trg_vocab,
                            normalize=config['normalized_bleu'],
                            every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot(config['model_save_directory'], channels=[['decoder_cost_cost', 'validation_set_bleu_score', 'validation_set_meteor_score']],
                 every_n_batches=10))

    # Set up training algorithm
    logger.info("Initializing training algorithm")

    # if there is l2_regularization, dropout or random noise, we need to use the output of the modified graph
    if config['dropout'] < 1.0:
        algorithm = GradientDescent(
            cost=cg.outputs[0], parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()]),
            on_unused_sources='warn'
        )
    else:
        algorithm = GradientDescent(
            cost=cost, parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()]),
            on_unused_sources='warn'
        )

    #algorithm = GradientDescent(
    #    cost=cost, parameters=cg.parameters,
    #    step_rule=CompositeRule([StepClipping(config['step_clipping']),
    #                             eval(config['step_rule'])()],
    #                           ),
    #    on_unused_sources='warn'
    #)

    # enrich the logged information
    extensions.append(
        Timing(every_n_batches=100)
    )

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )

    # Train!
    main_loop.run()


training_cost = create_model(train_encoder, train_decoder)

# Set up training model
logger.info("Building model")
train_model = Model(training_cost)

dev_stream = get_dev_stream_with_context_features(**exp_config)

main(train_model, training_cost, exp_config, masked_stream, dev_stream=dev_stream, use_bokeh=True)


