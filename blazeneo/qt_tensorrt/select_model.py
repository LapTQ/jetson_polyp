from pathlib import Path
import os
import json
import logging

HERE = Path(__file__).parent

CONFIG_PATH = str(HERE / 'app_config.json')
WEIGHTS_DIR = str(HERE / '../weights')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s\t|%(levelname)s\t|%(funcName)s\t|%(lineno)d\t|%(message)s'
)


def get_input(config):

    model_type = input('[INPUT] Select model:\n\t(1) segmentation\n\t(2) detection\n>>> ')
    if model_type == '1':
        model_type = 'segmentation'
        threshold = config['threshold']
    else:
        model_type = 'detection'
        threshold = input(f"[INPUT] Select threshold (0 to 1) [<ENTER> to choose {config['threshold']}]:\n>>> ") or config['threshold']
        threshold = float(threshold)
    classify_tumors = True if input('[INPUT] Classfy tumors?\n\t(1) Yes\n\t(2) No\n>>> ') == 1 else False



    return {
        'model_type': model_type,
        'threshold': threshold,
        'classify_tumors': classify_tumors
    }




def update_config(config, input_):


    config['model_type'] = input_['model_type']
    config['threshold'] = input_['threshold']
    if config['model_type'] == 'segmentation':
        logging.info('Segmentation model is selected.')

        # TODO model path
        config['engine_file_path'] = os.path.join(WEIGHTS_DIR, 'segementation_onnx', 'repvgg-a1-fulldim.onnx')

        # set input size
        config['width'] = 384
        config['height'] = 384

        # set preprocess and postprocess mode
        config['preprocess_mode'] = 5
        config['postprocess_mode'] = 2
    
    elif config['model_type'] == 'detection':
        logging.info('Detection model is selected.')

        # TODO model path
        # config['engine_file_path'] =  os.path.join(WEIGHTS_DIR, 'detection_onnx', 'model.onnx')
        config['engine_file_path'] =  os.path.join(WEIGHTS_DIR, 'detection_tftrt_fp16', 'trt_FP16_2') 

        # set input size
        config['width'] = 512
        config['height'] = 512

        # set preprocess_mode
        config['preprocess_mode'] = 6
        config['postprocess_mode'] = 3

        # set class_labels
        config['class_labels'] = ['polyp', 'neo']            
    
    # Opt for classifying tumors or not
    if input_['classify_tumors'] is True:
        config['n_classes'] = 2
    elif input_['classify_tumors'] is False:
        config['n_classes'] = 1
    logging.info(f"Number of classes is set to {config['n_classes']}")


def main():

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    input_ = get_input(config)
    update_config(config, input_)
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f) 

    


if __name__ == '__main__':
    main()