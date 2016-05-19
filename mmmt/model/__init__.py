
from abc import ABCMeta, abstractmethod

from theano import tensor
from six import add_metaclass

from blocks.bricks import (Brick, Initializable, Sequence,
                           Feedforward, Linear, Tanh)
from blocks.bricks.base import lazy, application
from blocks.bricks.parallel import Parallel, Distribute
from blocks.bricks.recurrent import recurrent, BaseRecurrent
from blocks.utils import dict_union, dict_subset, pack


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
from machine_translation.models import MinRiskSequenceGenerator

from picklable_itertools.extras import equizip

from machine_translation.model import LookupFeedbackWMT15, InitializableFeedforwardSequence


# This is copied from machine translation so we can pop the 'context_dim' kwarg, and use it with the same
# interface as the other transitions
class GRUInitialState(GatedRecurrent):
    """Gated Recurrent with special initial state.

    Initial state of Gated Recurrent is set by an MLP that conditions on the
    last hidden state of the bidirectional encoder, applies an affine
    transformation followed by a tanh non-linearity to set initial state.

    """
    def __init__(self, attended_dim, context_dim, **kwargs):
        super(GRUInitialState, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.initial_transformer = MLP(activations=[Tanh()],
                                       dims=[attended_dim, self.dim],
                                       name='state_initializer')
        self.children.append(self.initial_transformer)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        initial_state = self.initial_transformer.apply(
            attended[0, :, -self.attended_dim:])
        return initial_state

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                               name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)


class GRUInitialStateWithInitialStateSumContext(GatedRecurrent):
    """Gated Recurrent with special initial state.

    Initial state of Gated Recurrent is set by an MLP that conditions on the
    last hidden state of the bidirectional encoder, applies an affine
    transformation followed by a tanh non-linearity to set initial state.

    """
    def __init__(self, attended_dim, context_dim, **kwargs):
        super(GRUInitialStateWithInitialStateSumContext, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.context_dim = context_dim

        # two MLPs which map to the same dimension, then we sum
        # the motivation here is to allow the network to pretrain on the normal MT, task,
        # then keep some params static, and continue training with the context-enhanced task
        # the state transformer
        self.initial_transformer = MLP(activations=[Tanh()],
                                       dims=[attended_dim, self.dim],
                                       name='state_initializer')

        # the context transformer
        self.context_transformer = MLP(activations=[Tanh(),Tanh(),Tanh()],
                                       dims=[context_dim, 2000, 1000, self.dim],
                                       name='context_initializer')

        self.children.extend([self.initial_transformer, self.context_transformer])

    # THINKING: how to best combine the image info with the source info?
    @application
    def initial_states(self, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        context = kwargs['initial_state_context']
        attended_reverse_final_state = attended[0, :, -self.attended_dim:]
        initial_state_representation = self.initial_transformer.apply(attended_reverse_final_state)
        initial_context_representation = self.context_transformer.apply(context)
        initial_state = initial_state_representation + initial_context_representation
        return initial_state

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                                  name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                                                  name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)

class GRUInitialStateWithInitialStateConcatContext(GatedRecurrent):
    """Gated Recurrent with special initial state.

    Initial state of Gated Recurrent is set by an MLP that conditions on the
    last hidden state of the bidirectional encoder, applies an affine
    transformation followed by a tanh non-linearity to set initial state.

    """
    def __init__(self, attended_dim, context_dim, **kwargs):
        super(GRUInitialStateWithInitialStateConcatContext, self).__init__(**kwargs)
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


# TODO: Note that AttentionRecurrent is currently _hacked_ in blocks to remove 'initial_state_context' from the
# TODO: kwargs in the `compute_states` function
# TODO: this hack will need to be redone every time blocks is updated/re-installed
class InitialContextSequenceGenerator(BaseSequenceGenerator):

    def __init__(self, readout, transition, attention,
                 add_contexts=True, **kwargs):
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        transition = AttentionRecurrent(
            transition, attention,
            add_contexts=add_contexts, name="att_trans")
        super(InitialContextSequenceGenerator, self).__init__(
            readout, transition, **kwargs)

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


class MinRiskInitialContextSequenceGenerator(InitialContextSequenceGenerator):

    def __init__(self, *args, **kwargs):
        self.softmax = NDimensionalSoftmax()
        super(MinRiskInitialContextSequenceGenerator, self).__init__(*args, **kwargs)
        self.children.append(self.softmax)

    @application
    def probs(self, readouts):
        return self.softmax.apply(readouts, extra_ndim=readouts.ndim - 2)

    # TODO: check where 'target_samples_mask' is used -- do we need a mask for context features (probably not)
    # Note: the @application decorator inspects the arguments, and transparently adds args  ('application_call')
    @application(inputs=['representation', 'source_sentence_mask',
                         'target_samples_mask', 'target_samples', 'scores'],
                 outputs=['cost'])
    def expected_cost(self, application_call, representation, source_sentence_mask,
                      target_samples, target_samples_mask, scores, smoothing_constant=0.005,
                      **kwargs):
        """
        emulate the process in sequence_generator.cost_matrix, but compute log probabilities instead of costs
        for each sample, we need its probability according to the model (these could actually be passed from the
        sampling model, which could be more efficient)
        """

        # Transpose everything (note we can use transpose here only if it's 2d, otherwise we need dimshuffle)
        source_sentence_mask = source_sentence_mask.T

        # make samples (time, batch)
        samples = target_samples.T
        samples_mask = target_samples_mask.T

        # we need this to set the 'attended' kwarg
        keywords = {
            'mask': target_samples_mask,
            'outputs': target_samples,
            'attended': representation,
            'attended_mask': source_sentence_mask
        }

        batch_size = samples.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(keywords, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        # contexts = dict_subset(keywords, self._context_names, must_have=False)

        # add the initial state context features
        contexts = dict_subset(keywords, self._context_names, must_have=False)
        contexts['initial_state_context'] = kwargs['initial_state_context']

        feedback = self.readout.feedback(samples)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=samples_mask, return_initial_states=True, as_dict=True,
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

        word_probs = self.probs(readouts)
        word_probs = tensor.log(word_probs)

        # Note: converting the samples to one-hot wastes space, but it gets the job done
        # TODO: this may be the op that sometimes causes out-of-memory
        one_hot_samples = tensor.eye(word_probs.shape[-1])[samples]
        one_hot_samples.astype('float32')
        actual_probs = word_probs * one_hot_samples

        # reshape to (batch, time, prob), then sum over the batch dimension
        # to get sequence-level probability
        actual_probs = actual_probs.dimshuffle(1,0,2)
        # we are first summing over vocabulary (only one non-zero cell per row)
        sequence_probs = actual_probs.sum(axis=2)
        sequence_probs = sequence_probs * target_samples_mask
        # now sum over time dimension
        sequence_probs = sequence_probs.sum(axis=1)

        # reshape and do exp() to get the true probs back
        # sequence_probs = tensor.exp(sequence_probs.reshape(scores.shape))
        sequence_probs = sequence_probs.reshape(scores.shape)

        # Note that the smoothing constant can be set by user
        sequence_distributions = (tensor.exp(sequence_probs*smoothing_constant) /
                                  tensor.exp(sequence_probs*smoothing_constant)
                                  .sum(axis=1, keepdims=True))

        # the following lines are done explicitly for code clarity
        # -- first get sequence expectation, then sum up the expectations for every
        # seq in the minibatch
        expected_scores = (sequence_distributions * scores).sum(axis=1)
        expected_scores = expected_scores.sum(axis=0)

        return expected_scores


# TODO: a lot of code was duplicated from neural_mt during speedy prototyping -- CLEAN UP
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
                 representation_dim, context_dim, target_transition,
                 theano_seed=None, loss_function='cross_entropy', **kwargs):
        super(InitialContextDecoder, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.theano_seed = theano_seed

        # Initialize gru with special initial state
        self.transition = target_transition(
            attended_dim=state_dim, context_dim=context_dim, dim=state_dim,
            activation=Tanh(), name='decoder')

        # self.transition = GRUInitialStateWithInitialStateConcatContext(
        #     attended_dim=state_dim, context_dim=context_dim, dim=state_dim,
        #     activation=Tanh(), name='decoder')

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
        elif loss_function == 'min_risk':
            self.sequence_generator = MinRiskInitialContextSequenceGenerator(
                readout=readout,
                transition=self.transition,
                attention=self.attention,
                fork=Fork([name for name in self.transition.apply.sequences
                           if name != 'mask'], prototype=Linear())
            )
        # the name is important, because it lets us match the brick hierarchy names for the vanilla SequenceGenerator
        # to load pretrained models
            # TODO: quick hack to fix bug
            self.sequence_generator.name = 'initialcontextsequencegenerator'


        else:
            raise ValueError('The decoder does not support the loss function: {}'.format(loss_function))

        # TODO: uncomment this!!
        # self.sequence_generator.name = 'sequencegenerator'

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
            'initial_state_context': initial_state_context}
        )

        return (cost * target_sentence_mask).sum() / target_sentence_mask.shape[1]

    # Note: this requires the decoder to be using sequence_generator which implements expected cost
    # Note: initial_state_context is passed through in the kwargs
    @application(inputs=['representation', 'source_sentence_mask',
                         'target_samples_mask', 'target_samples', 'scores'],
                 outputs=['cost'])
    def expected_cost(self, representation, source_sentence_mask, target_samples, target_samples_mask, scores,
                      **kwargs):
        return self.sequence_generator.expected_cost(representation,
                                                     source_sentence_mask,
                                                     target_samples, target_samples_mask, scores, **kwargs)


    @application
    def generate(self, source_sentence, representation, initial_state_context, **kwargs):
        return self.sequence_generator.generate(
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            initial_state_context=initial_state_context,
            **kwargs)


