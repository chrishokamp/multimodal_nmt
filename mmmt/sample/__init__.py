from __future__ import print_function

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
