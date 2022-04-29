import os
import sys
import json
import boto3
import pandas as pd
import optuna

from train_v11_small import main as train
from inference_v11_full_weights import main as inference_weights
from inference_v11_font_interpolation import main as inference_font_interpolation
from make_cascade_visualization import main as make_cascade_highlight
from make_cascade_visualization_no_highlight import main as make_cascade


s3 = boto3.resource('s3')


def get_trial_history():
    bucket = s3.Bucket('neural-font-rasterization')
    trials = {}
    for obj in bucket.objects.filter(Prefix='hp-search').all():
        if obj.key.count('/') > 2:
            continue

        _, exp_name, filename = obj.key.split('/')
        if exp_name not in trials:
            trials[exp_name] = {}

        if filename == 'bce_loss_hist.json':
            trials[exp_name]['bce'] = json.load(obj.get()['Body'])[-1]
        elif filename == 'config.json':
            trials[exp_name].update(json.load(obj.get()['Body']))

    return pd.DataFrame(trials).T


def get_hyperparameters(trials_df):
    study = optuna.create_study(direction='minimize')

    for i, row in trials_df.iterrows():
        study.enqueue_trial({
            k: row[k]
            for k in [
                'REG_LAMBDA',
                'FOCAL_LOSS_GAMMA',
                'FONT_EMBEDDING_DIM',
                'modification_model_layer_size',
                'modification_model_n_layers',
                'rasterizer_layer_size',
                'rasterizer_n_layers_in_block'
            ]
        })

        trial = study.ask()

        study.tell(trial, row['bce'])

    trial = study.ask()
    return (
        trial.suggest_loguniform('REG_LAMBDA', 1e-3, 1e-2),
        trial.suggest_float('FOCAL_LOSS_GAMMA', 5, 10),
        trial.suggest_int('FONT_EMBEDDING_DIM', 600, 1600),
        trial.suggest_int('modification_model_layer_size', 100, 400),
        trial.suggest_int('modification_model_n_layers', 3, 7),
        trial.suggest_int('rasterizer_layer_size', 300, 2000),
        trial.suggest_int('rasterizer_n_layers_in_block', 2, 5)
    )


def main():
    if len(sys.argv) != 2:
        exit(f'{sys.argv[0]} takes exactly one argument - experiment name')

    exp_name = sys.argv[1].replace(' ', '_').lower()

    for exp_num in range(1000):
        trials_df = get_trial_history()
        trials_df.reset_index().to_csv('/data/trials.csv', index=False)
        os.system('aws s3 cp /data/trials.csv s3://neural-font-rasterization/hp-search/trials.csv')

        REG_LAMBDA = 1e-3
        FOCAL_LOSS_GAMMA = 6
        FONT_EMBEDDING_DIM = 1420
        modification_model_layer_size = 300
        modification_model_n_layers = 4
        rasterizer_layer_size = 1082
        rasterizer_n_layers_in_block = 3

        if trials_df.shape[0] > 0:
            (
                REG_LAMBDA,
                FOCAL_LOSS_GAMMA,
                FONT_EMBEDDING_DIM,
                modification_model_layer_size,
                modification_model_n_layers,
                rasterizer_layer_size,
                rasterizer_n_layers_in_block
            ) = get_hyperparameters(trials_df)

        train(
            REG_LAMBDA,
            FOCAL_LOSS_GAMMA,
            FONT_EMBEDDING_DIM,
            modification_model_layer_size,
            modification_model_n_layers,
            rasterizer_layer_size,
            rasterizer_n_layers_in_block
        )

        inference_weights()
        inference_font_interpolation()

        make_cascade_highlight('/data/results/v11_real_small/cap_a_full_weights/comparison')
        make_cascade('/data/results/v11_real_small/cap_a_interpolated_fonts/comparison')

        experiment_base = f's3://neural-font-rasterization/hp-search/{exp_name}-{exp_num}'
        os.system(f'aws s3 cp /data/training/v11_real_small/cap_a/bce_loss_hist.json {experiment_base}/')
        os.system(f'aws s3 cp /data/training/v11_real_small/cap_a/generator_loss_hist.json {experiment_base}/')
        os.system(f'aws s3 cp /data/training/v11_real_small/cap_a/discriminator_loss_hist.json {experiment_base}/')
        os.system(f'aws s3 cp --recursive '
                  f'/data/training/v11_real_small/cap_a/model '
                  f'{experiment_base}/model')
        os.system(f'aws s3 cp --recursive '
                  f'/data/training/v11_real_small/cap_a/inner_model '
                  f'{experiment_base}/inner_model')
        os.system(f'aws s3 cp --recursive '
                  f'/data/training/v11_real_small/cap_a/font_embeddings '
                  f'{experiment_base}/font_embeddings')
        os.system(f'aws s3 cp '
                  f'/data/results/v11_real_small/cap_a_full_weights/cascade.png '
                  f'{experiment_base}/cascade_weights.png')
        os.system(f'aws s3 cp '
                  f'/data/results/v11_real_small/cap_a_interpolated_fonts/cascade.png '
                  f'{experiment_base}/cascade_fonts.png')

        with open('/data/training/v11_real_small/cap_a/config.json', 'w') as f:
            json.dump({
                'REG_LAMBDA': REG_LAMBDA,
                'FOCAL_LOSS_GAMMA': FOCAL_LOSS_GAMMA,
                'FONT_EMBEDDING_DIM': FONT_EMBEDDING_DIM,
                'modification_model_layer_size': modification_model_layer_size,
                'modification_model_n_layers': modification_model_n_layers,
                'rasterizer_layer_size': rasterizer_layer_size,
                'rasterizer_n_layers_in_block': rasterizer_n_layers_in_block
            }, f, indent=4)

        os.system(f'aws s3 cp '
                  f'/data/training/v11_real_small/cap_a/config.json '
                  f'{experiment_base}/')


if __name__ == '__main__':
    main()
