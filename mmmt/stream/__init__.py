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


