import numpy

from six.moves import cPickle

from fuel.datasets import IterableDataset
from fuel.transformers import Merge
from fuel.streams import DataStream
from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)
import numpy
from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping, Transformer)

from six.moves import cPickle

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

        # dev_stream = DataStream(dev_dataset)
        dev_stream = Merge([dev_dataset.get_example_stream(),
                            valid_image_stream], ('source', 'initial_context'))
    #         dev_stream = dev_stream.get_example_stream()

    return dev_stream



# Module for functionality associated with streaming data
# TODO: modify this to pass the correct args for multimodal initial context
class MMMTSampleStreamTransformer:
    """
    Stateful transformer which takes a stream of (source, target) and adds the sources ('samples', 'scores')

    Samples are generated by calling the sample func with the source as argument

    Scores are generated by comparing each generated sample to the reference

    Parameters
    ----------
    sample_func: function(num_samples=1) which takes source seq and outputs <num_samples> samples
    score_func: function

    At call time, we expect a stream providing (sources, references) -- i.e. something like a TextFile object


    """

    def __init__(self, sample_func, score_func, num_samples=1, **kwargs):
        self.sample_func = sample_func
        self.score_func = score_func
        self.num_samples = num_samples
        # kwargs will get passed to self.score_func when it gets called
        self.kwargs = kwargs

    def __call__(self, data, **kwargs):
        source = data[0]
        reference = data[1]
        initial_context = data[2]

        # each sample may be of different length
        samples = self.sample_func(numpy.array(source), initial_context, self.num_samples)

        # import ipdb;ipdb.set_trace()

        # TODO: we currently have to pass the source because of the interface to mteval_v13
        scores = numpy.array(self._compute_scores(source, reference, samples, **self.kwargs)).astype('float32')
        # import ipdb;ipdb.set_trace()

        return (samples, scores)

    # Note that many sentence-level metrics like BLEU can be computed directly over the indexes (not the strings),
    # Note that some sentence-level metrics like METEOR require the string representation
    # if the scoring function needs to map from ints to strings, provide 'src_vocab' and 'trg_vocab' via the kwargs
    # So we don't need to map back to a string representation
    def _compute_scores(self, source, reference, samples, **kwargs):
        """Call the scoring function to compare each sample to the reference"""

        return self.score_func(source, reference, samples, **kwargs)



# TODO: duplicate the initial context as well
class CopySourceAndContextNTimes(Transformer):
    """Duplicate the source N times to match the number of samples

    We need this transformer because the attention model expects one source sequence for each
    target sequence, but in the sampling case there are effectively (instances*sample_size) target sequences

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to wrap
    n_samples : int -- the number of samples that were generated for each source sequence

    """
    def __init__(self, data_stream, n_samples=5, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')
        self.n_samples = n_samples

        super(CopySourceAndContextNTimes, self).__init__(
            data_stream, produces_examples=False, **kwargs)


    @property
    def sources(self):
        return self.data_stream.sources

    def transform_batch(self, batch):
        batch_with_expanded_source = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source == 'source' or source == 'initial_context':

                # copy each source sequence self.n_samples times, but keep the tensor 2d
                expanded_source = []
                for ins in source_batch:
                    expanded_source.extend([ins for _ in range(self.n_samples)])

                batch_with_expanded_source.append(expanded_source)
            else:
                batch_with_expanded_source.append(source_batch)

        return tuple(batch_with_expanded_source)


