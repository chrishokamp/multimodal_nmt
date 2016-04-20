# coding: utf-8

from __future__ import print_function
from theano import tensor
from toolz import merge
import os

import numpy

from fuel.datasets import IterableDataset
from fuel.transformers import Merge
from fuel.streams import DataStream

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

TRAIN_IMAGE_FEATURES = '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/train.npz'
DEV_IMAGE_FEATURES = '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/dev.npz'
TEST_IMAGE_FEATURES = '/media/1tb_drive/multilingual-multimodal/flickr30k/img_features/f30k-translational-newsplits/test.npz'

# the prototype config for NMT experiment with images

BASEDIR = '/media/1tb_drive/multilingual-multimodal/flickr30k/train/processed/BERTHA-TEST_Adam_wmt-multimodal_internal_data_dropout'+          '0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/'
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


from machine_translation.stream import _ensure_special_tokens, _length, PaddingWithEOS, _oov_to_unk, _too_long

def get_tr_stream_with_context_features(src_vocab, trg_vocab, src_data, trg_data, context_features,
                  src_vocab_size=30000, trg_vocab_size=30000, unk_id=1,
                  seq_len=50, batch_size=80, sort_k_batches=12, **kwargs):
    """Prepares the training data stream."""

    def _get_np_array(filename):
        return numpy.load(filename)['arr_0']
    
    # Load dictionaries and ensure special tokens exist
    src_vocab = _ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict)
        else cPickle.load(open(src_vocab)),
        bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
    trg_vocab = _ensure_special_tokens(
        trg_vocab if isinstance(trg_vocab, dict) else
        cPickle.load(open(trg_vocab)),
        bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

    # Get text files from both source and target
    src_dataset = TextFile([src_data], src_vocab, None)
    trg_dataset = TextFile([trg_data], trg_vocab, None)

    # Merge them to get a source, target pair
    stream = Merge([src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   ('source', 'target'))

    # Filter sequences that are too long
    stream = Filter(stream,
                    predicate=_too_long(seq_len=seq_len))
    
  
    # Replace out of vocabulary tokens with unk token
    # TODO: doesn't the TextFile stream do this anyway?
    stream = Mapping(stream,
                     _oov_to_unk(src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

    # now add the source with the image features
    # create the image datastream (iterate over a file line-by-line)
    train_features = _get_np_array(context_features)
    train_feature_dataset = IterableDataset(train_features)
    train_image_stream = DataStream(train_feature_dataset)

    stream = Merge([stream, train_image_stream], ('source', 'target', 'initial_context'))
    
    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(
                       batch_size*sort_k_batches))

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length))

    # Convert it into a stream again
    stream = Unpack(stream)

    # Construct batches from the stream with specified batch size
    stream = Batch(
        stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    masked_stream = PaddingWithEOS(
        stream, [src_vocab_size - 1, trg_vocab_size - 1], mask_sources=('source', 'target'))

    return masked_stream, src_vocab, trg_vocab


# Remember that the BleuValidator does hackish stuff to get target set information from the main_loop data_stream
# using all kwargs here makes it more clear that this function is always called with get_dev_stream(**config_dict)
def get_dev_stream_with_context_features(val_context_features=None, val_set=None, src_vocab=None,
                                         src_vocab_size=30000, unk_id=1, **kwargs):
    """Setup development set stream if necessary."""
    
    def _get_np_array(filename):
        return numpy.load(filename)['arr_0']
    
    
    dev_stream = None
    if val_set is not None and src_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab)),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        
        # TODO: how is the dev dataset used without the context features?
        dev_dataset = TextFile([val_set], src_vocab, None)
        
        # now add the source with the image features
        # create the image datastream (iterate over a file line-by-line)
        con_features = _get_np_array(val_context_features)
        con_feature_dataset = IterableDataset(con_features)
        valid_image_stream = DataStream(con_feature_dataset)
        
        dev_stream = DataStream(dev_dataset)
        dev_stream = Merge([dev_dataset.get_example_stream(),
                            valid_image_stream], ('source', 'initial_context'))
#         dev_stream = dev_stream.get_example_stream()

    return dev_stream


# In[8]:

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

from six.moves import cPickle

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


class GRUInitialStateWithInitialStateContext(GatedRecurrent):
    """Gated Recurrent with special initial state.

    Initial state of Gated Recurrent is set by an MLP that conditions on the
    last hidden state of the bidirectional encoder, applies an affine
    transformation followed by a tanh non-linearity to set initial state.

    """
    def __init__(self, attended_dim, context_dim, **kwargs):
        super(GRUInitialStateWithInitialStateContext, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.context_dim = context_dim

        self.initial_transformer = MLP(activations=[Tanh(),Tanh(),Tanh()],
                                       dims=[attended_dim + context_dim, 1000, 500, self.dim],
                                       name='state_initializer')
        self.children.append(self.initial_transformer)
  
    # WORKING: add the images as another context to the recurrent transition
    # THINKING: how to best combine the image info with the source info?
    @application
    def initial_states(self, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        context = kwargs['initial_state_context']
        attended_reverse_final_state = attended[0, :, -self.attended_dim:]
        concat_attended_and_context = tensor.concatenate([attended_reverse_final_state, context], axis=1)
        initial_state = self.initial_transformer.apply(concat_attended_and_context)
        return initial_state

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                               name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)


# In[15]:

from abc import ABCMeta, abstractmethod

from theano import tensor
from six import add_metaclass

from blocks.bricks import (Brick, Initializable, Sequence,
                           Feedforward, Linear, Tanh)
from blocks.bricks.base import lazy, application
from blocks.bricks.parallel import Parallel, Distribute
from blocks.bricks.recurrent import recurrent, BaseRecurrent
from blocks.utils import dict_union, dict_subset, pack

from blocks.bricks.attention import AttentionRecurrent

class InitialContextAttentionRecurrent(AttentionRecurrent):
    
    def __init__(self, *args, **kwargs):
        super(InitialContextAttentionRecurrent, self).__init__(*args, **kwargs)
#         print('CONTEXT NAMES:')
#         print(self._context_names)
#         self._context_names.append('initial_state_context')
#         print('CONTEXT NAMES:')
#         print(self._context_names)
        
#     @application
#     def compute_states(self, **kwargs):
#         r"""Compute current states when glimpses have already been computed.

#         Combines an application of the `distribute` that alter the
#         sequential inputs of the wrapped transition and an application of
#         the wrapped transition. All unknown keyword arguments go to
#         the wrapped transition.

#         Parameters
#         ----------
#         \*\*kwargs
#             Should contain everything what `self.transition` needs
#             and in addition the current glimpses.

#         Returns
#         -------
#         current_states : list of :class:`~tensor.TensorVariable`
#             Current states computed by `self.transition`.

#         """
#         # make sure we are not popping the mask
#         normal_inputs = [name for name in self._sequence_names
#                          if 'mask' not in name]
#         sequences = dict_subset(kwargs, normal_inputs, pop=True)
#         glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
#         if self.add_contexts:
#             kwargs.pop(self.attended_name)
#             # attended_mask_name can be optional
#             kwargs.pop(self.attended_mask_name, None)
            
        

#         sequences.update(self.distribute.apply(
#             as_dict=True, **dict_subset(dict_union(sequences, glimpses),
#                                         self.distribute.apply.inputs)))
#         current_states = self.transition.apply(
#             iterate=False, as_list=True,
#             **dict_union(sequences, kwargs))
#         return current_states
    
#     @recurrent
#     def do_apply(self, **kwargs):
#         r"""Process a sequence attending the attended context every step.

#         In addition to the original sequence this method also requires
#         its preprocessed version, the one computed by the `preprocess`
#         method of the attention mechanism. Unknown keyword arguments
#         are passed to the wrapped transition.

#         Parameters
#         ----------
#         \*\*kwargs
#             Should contain current inputs, previous step states, contexts,
#             the preprocessed attended context, previous step glimpses.

#         Returns
#         -------
#         outputs : list of :class:`~tensor.TensorVariable`
#             The current step states and glimpses.

#         """
#         attended = kwargs[self.attended_name]
#         preprocessed_attended = kwargs.pop(self.preprocessed_attended_name)
#         attended_mask = kwargs.get(self.attended_mask_name)
#         sequences = dict_subset(kwargs, self._sequence_names, pop=True,
#                                 must_have=False)
#         states = dict_subset(kwargs, self._state_names, pop=True)
#         glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)

#         current_glimpses = self.take_glimpses(
#             as_dict=True,
#             **dict_union(
#                 states, glimpses,
#                 {self.attended_name: attended,
#                  self.attended_mask_name: attended_mask,
#                  self.preprocessed_attended_name: preprocessed_attended}))
#         current_states = self.compute_states(
#             as_list=True,
#             **dict_union(sequences, states, current_glimpses, kwargs))
#         return current_states + list(current_glimpses.values())
    
    
#     @do_apply.property('contexts')
#     def do_apply_contexts(self):
#         return self._context_names + [self.preprocessed_attended_name] + ['initial_state_context']
#         return self._context_names + [self.preprocessed_attended_name]


# In[16]:

# WORKING: sequence generator which uses the contexts properly

from abc import ABCMeta, abstractmethod

from six import add_metaclass
from theano import tensor

from blocks.bricks import Initializable, Random, Bias, NDimensionalSoftmax
from blocks.bricks.base import application, Brick, lazy
from blocks.bricks.parallel import Fork, Merge
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import recurrent
from blocks.bricks.attention import (
    AbstractAttentionRecurrent, AttentionRecurrent)
from blocks.roles import add_role, COST
from blocks.utils import dict_union, dict_subset

from blocks.bricks.sequence_generators import BaseSequenceGenerator

class InitialContextSequenceGenerator(BaseSequenceGenerator):
    
    def __init__(self, readout, transition, attention=None,
                 add_contexts=True, **kwargs):
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        if attention:
            transition = InitialContextAttentionRecurrent(
#             transition = AttentionRecurrent(
                transition, attention,
                add_contexts=add_contexts, name="att_trans")
        else:
            transition = FakeAttentionRecurrent(transition,
                                                name="with_fake_attention")
        super(InitialContextSequenceGenerator, self).__init__(
            readout, transition, **kwargs)
    
#     def __init__(self, *args, **kwargs):
#         self.softmax = NDimensionalSoftmax()
#         super(InitialContextSequenceGenerator, self).__init__(*args, **kwargs)
#         self.children.append(self.softmax)

    @application
    def cost_matrix(self, application_call, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.

        See Also
        --------
        :meth:`cost` : Scalar cost.

        """
        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
#         contexts = dict_subset(kwargs, self._context_names, must_have=False)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        contexts['initial_state_context'] = kwargs['initial_state_context']
    
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(batch_size)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
        costs = self.readout.cost(readouts, outputs)
        if mask is not None:
            costs *= mask

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)

        # This variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        for name in self._state_names + self._glimpse_names:
            application_call.add_auxiliary_variable(
                results[name][-1].copy(), name=name+"_final_value")

        return costs


from theano import tensor
from toolz import merge

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

from picklable_itertools.extras import equizip

from machine_translation.model import LookupFeedbackWMT15, InitializableFeedforwardSequence

# TODO: a lot of code was duplicated here during speedy prototyping -- CLEAN UP
class InitialContextDecoder(Initializable):
    """
    Decoder which incorporates context features into the target-side initial state

    Parameters:
    -----------
    vocab_size: int
    embedding_dim: int
    representation_dim: int
    theano_seed: int
    loss_function: str : {'cross_entropy'(default) | 'min_risk'}

    """

    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, context_dim, theano_seed=None, loss_function='cross_entropy', **kwargs):
        super(InitialContextDecoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.theano_seed = theano_seed

        # Initialize gru with special initial state
        self.transition = GRUInitialStateWithInitialStateContext(
            attended_dim=state_dim, context_dim=context_dim, dim=state_dim,
            activation=Tanh(), name='decoder')

        # Initialize the attention mechanism
        self.attention = SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=representation_dim,
            match_dim=state_dim, name="attention")

        # Initialize the readout, note that SoftmaxEmitter emits -1 for
        # initial outputs which is used by LookupFeedBackWMT15
        readout = Readout(
            source_names=['states', 'feedback',
                          # Chris: it's key that we're taking the first output of self.attention.take_glimpses.outputs
                          # Chris: the first output is the weighted avgs, the second is the weights in (batch, time)
                          self.attention.take_glimpses.outputs[0]],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=-1, theano_seed=theano_seed),
            feedback_brick=LookupFeedbackWMT15(vocab_size, embedding_dim),
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=state_dim, name='maxout_bias').apply,
                 Maxout(num_pieces=2, name='maxout').apply,
                 Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
                        use_bias=False, name='softmax0').apply,
                 Linear(input_dim=embedding_dim, name='softmax1').apply]),
            merged_dim=state_dim)

        # Build sequence generator accordingly
        if loss_function == 'cross_entropy':
            self.sequence_generator = InitialContextSequenceGenerator(
                readout=readout,
                transition=self.transition,
                attention=self.attention,
                fork=Fork([name for name in self.transition.apply.sequences
                           if name != 'mask'], prototype=Linear())
            )
#         elif loss_function == 'min_risk':
#             self.sequence_generator = MinRiskSequenceGenerator(
#                 readout=readout,
#                 transition=self.transition,
#                 attention=self.attention,
#                 fork=Fork([name for name in self.transition.apply.sequences
#                            if name != 'mask'], prototype=Linear())
#             )
            # the name is important, because it lets us match the brick hierarchy names for the vanilla SequenceGenerator
            # to load pretrained models
            self.sequence_generator.name = 'sequencegenerator'
        else:
            raise ValueError('The decoder does not support the loss function: {}'.format(loss_function))

        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence', 'initial_state_context'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask, initial_state_context):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended': representation,
            'attended_mask': source_sentence_mask,
            'initial_state_context': initial_state_context
            }
        )

        return (cost * target_sentence_mask).sum() /             target_sentence_mask.shape[1]

    # Note: this requires the decoder to be using sequence_generator which implements expected cost
#     @application(inputs=['representation', 'source_sentence_mask',
#                          'target_samples_mask', 'target_samples', 'scores'],
#                  outputs=['cost'])
#     def expected_cost(self, representation, source_sentence_mask, target_samples, target_samples_mask, scores,
#                       **kwargs):
#         return self.sequence_generator.expected_cost(representation,
#                                                      source_sentence_mask,
#                                                      target_samples, target_samples_mask, scores, **kwargs)


    @application
    def generate(self, source_sentence, representation, initial_state_context, **kwargs):
        return self.sequence_generator.generate(
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            initial_state_context=initial_state_context,
            **kwargs)


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
        return self._get_attr_rec(getattr(obj, attr), attr)             if hasattr(obj, attr) else obj

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
            trans, costs =                 self.beam_search.search(
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


# In[ ]:

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
        
        # TODO: decoder generate function also needs to include the new initial contexts in the kwargs
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


