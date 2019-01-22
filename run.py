import sys
import getopt
import traceback
import json
import pandas as pd
import kmodel.kmodel as kmod
import kmodel.candidates as kcand


def main(argv):
    tr_dir, va_dir, imgdir = None, None, None
    model_params = None
    usage_str = 'run.py -t <traindir> -v <valdir> -i <imgdir> [-p <params>]'
    try:
        opts, args = getopt.getopt(
            argv, "ht:v:i:p:", ["traindir=", "valdir=", "imgdir=", "params="])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            ts = 'folder with classed images'
            print(usage_str)
            print('\timgdir: %s to use for %s' % (
                ts, 'finding ambiguous candidates with trained model.'))
            print('\ttraindir: %s to use for %s' % (
                ts, 'training the model'))
            print('\tvaldir: %s to use for %s' % (
                ts, 'validating training'))
            print('\tparams: path to a json file with kmodel parameters',
                  '\t\tSee `default_params.json` or '
                  '\t\t`kmodel.kmodel.sample_params`')
            sys.exit()
        elif opt in ("-i", "--imgdir"):
            imgdir = arg
        elif opt in ("-t", "--traindir"):
            tr_dir = arg
        elif opt in ("-v", "--valdir"):
            va_dir = arg
        elif opt in ("-p", "--params"):
            with open(arg) as json_file:
                model_params = json.load(json_file)

    if imgdir is None:
        print(usage_str)
        print('image dir is mandatory')
        sys.exit(2)
    if tr_dir is None:
        print(usage_str)
        print('train dir is mandatory')
        sys.exit(2)
    if va_dir is None:
        print(usage_str)
        print('validation dir is mandatory (can be same as train dir)')
        sys.exit(2)
    if model_params is None:
        model_params = kmod.sample_params

    print('Standalone execution with opts', opts)
    try:
        execute_standalone(tr_dir, va_dir, imgdir, model_params)
    except Exception as e:
        print('Error during execution', e)
        tb = traceback.format_exc()
        print(tb)
        sys.exit(2)
    print('run.py finished')


_logsep = '\n#################\n'


def execute_standalone(tr_dir, va_dir, img_dir, model_params):
    print(_logsep + '1. Creating custom model')
    md = kmod.create_custom_model(model_params)

    print(_logsep + '1.5 Create batch generator(s)')
    train_generator = kmod.train_generator_for_dir(tr_dir, md)
    validation_generator = kmod.validation_generator_for_dir(va_dir, md)

    print(_logsep + '2 Train top layers (this may take a few minutes)')
    history_top = kmod.train_phase(
        md, phase='top', train_gen=train_generator,
        valid_gen=validation_generator)
    assert max(history_top.history['acc']) > 0.85
    # TODO: save image?

    print(_logsep + '3 Fine-tune training')
    history_fit = kmod.train_phase(
        md, phase='fine', train_gen=train_generator,
        valid_gen=validation_generator)
    assert max(history_fit.history['acc']) > 0.89

    print(_logsep + '4. Find candidates')
    all_generator = kmod.validation_generator_for_dir(img_dir, md)
    pt = pd.DataFrame(kcand.predict_table(md['model'], all_generator))
    pt['sanushi'] = kcand.abs_p_diff(pt, categA='sandwich', categB='sushi')
    print(_logsep + '5. Save top k candidates to csv file')
    pt.sort_values(by=['sanushi'],
                   ascending=True)[:20].to_csv('sanushi_candidates.csv')


if __name__ == "__main__":
    main(sys.argv[1:])
