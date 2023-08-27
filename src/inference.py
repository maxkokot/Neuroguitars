import click
import yaml
import logging

from visualization.visualize import make_samples, save_im
from models.model import load_model


def inference(model_config_path):

    with open(model_config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    img_size = config['img_size']
    latent_dim = config['latent_dim']
    model_dir = config['model_dir']
    pic_dir = config['pic_dir']
    lr = config['lr']

    gan = load_model(model_dir, img_size=img_size,
                     latent_dim=latent_dim, lr=lr)

    grid = make_samples(gan)
    save_im(grid, pic_dir)


@click.command(name="inference")
@click.option('--model_config_path', default='../config/model_config.yaml')
def inference_command(model_config_path):
    logger = logging.getLogger(__name__)
    logger.info('Generating images')
    inference(model_config_path)
    logger.info('Images have been generated and saved')


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    inference_command()
