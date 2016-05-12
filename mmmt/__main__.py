from __future__ import print_function
import argparse
import logging
import pprint
import codecs
import re
import os
import time
from subprocess import Popen, PIPE, check_output

from machine_translation import configurations

from mmmt import main, NMTPredictor
from mmmt.stream import get_tr_stream_with_context_features, get_dev_stream_with_context_features

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("exp_config",
                    help="Path to the yaml config file for your experiment")
parser.add_argument("-m", "--mode", default='train',
                    help="The mode we are in [train,predict,server] -- default=train")
parser.add_argument("--bokeh",  default=False, action="store_true",
                    help="Use bokeh server for plotting")

if __name__ == "__main__":
    # Get configurations for model
    args = parser.parse_args()
    arg_dict = vars(args)
    configuration_file = arg_dict['exp_config']
    mode = arg_dict['mode']
    logger.info('Running Neural Machine Translation in mode: {}'.format(mode))
    config_obj = configurations.get_config(configuration_file)
    # add the config file name into config_obj
    config_obj['config_file'] = configuration_file
    logger.info("Model Configuration:\n{}".format(pprint.pformat(config_obj)))

    # TODO: organize mmmt code so that we can implement train for mmmt
    # TODO: support specifying target transition via config
    # TODO: use eval() to get the target transition we want

    if mode == 'train':
        # Get data streams and call main
        train_stream, source_vocab, target_vocab = get_tr_stream_with_context_features(**config_obj)
        dev_stream = get_dev_stream_with_context_features(**config_obj)
        main(config_obj, train_stream, dev_stream, source_vocab, target_vocab, args.bokeh)

    elif mode == 'predict':
        predictor = NMTPredictor(config_obj)
        predictor.predict_files(config_obj['test_set'], config_obj['test_context_features'],
                                output_file=config_obj.get('translated_output_file', None),
                                output_costs=config_obj.get('output_cost', False))

    elif mode == 'evaluate':
        logger.info("Started Evaluation: ")
        val_start_time = time.time()

        # translate if necessary, write output file, call external evaluation tools and show output
        translated_output_file = config_obj.get('translated_output_file', None)
        if translated_output_file is not None and os.path.isfile(translated_output_file):
                logger.info('{} already exists, so I\'m evaluating the BLEU score of this file with respect to the ' +
                            'reference that you provided: {}'.format(translated_output_file,
                                                                     config_obj['test_gold_refs']))
        else:
            predictor = NMTPredictor(config_obj)
            logger.info('Translating: {}'.format(config_obj['test_set']))
            translated_output_file = predictor.predict_files(config_obj['test_set'],
                                                             config_obj['test_context_features'],
                                                             translated_output_file)
            logger.info('Translated: {}, output was written to: {}'.format(config_obj['test_set'],
                                                                           translated_output_file))

        # BLEU
        multibleu_cmd = ['perl', config_obj['bleu_script'],
                         config_obj['test_gold_refs'], '<']

        mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE)

        with codecs.open(translated_output_file, encoding='utf8') as hyps:
            for l in hyps.read().strip().split('\n'):
                # send the line to the BLEU script
                print(l.encode('utf8'), file=mb_subprocess.stdin)

        mb_subprocess.stdin.flush()

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
        logger.info('BLEU SCORE: {}'.format(bleu_score))
        mb_subprocess.terminate()

        # Meteor
        meteor_directory = config_obj.get('meteor_directory', None)
        if meteor_directory is not None:
            target_language = config_obj.get('target_lang', 'de')
            # java -Xmx2G -jar meteor-*.jar test reference - l en - norm
            # Note: not using the `-norm` parameter with METEOR since the references are already tokenized
            meteor_cmd = ['java', '-Xmx4G', '-jar', os.path.join(meteor_directory, 'meteor-1.5.jar'),
                          translated_output_file, config_obj['test_gold_refs'], '-l', target_language, '-norm']

            meteor_output = check_output(meteor_cmd)
            meteor_score = float(meteor_output.strip().split('\n')[-1].split()[-1])
            logger.info('METEOR SCORE: {}'.format(meteor_score))


    elif mode == 'server':

        import sys
        sys.path.append('.')
        from server import run_nmt_server

        # start restful server and log its port
        predictor = NMTPredictor(config_obj)
        run_nmt_server(predictor)


