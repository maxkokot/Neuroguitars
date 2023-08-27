import click
import yaml
import logging
import pytorch_lightning as pl

from data.data import DataModule
from models.model import DCGAN, save_model


def train(model_config_path):

    with open(model_config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    data_dir = config['data_dir']
    img_size = config['img_size']
    batch_size = config['batch_size']
    latent_dim = config['latent_dim']
    num_epochs = config['num_epochs']
    model_dir = config['model_dir']
    lr = config['lr']

    pl.seed_everything(2023)
    datamodule = DataModule(data_dir, img_size, batch_size)
    gan = DCGAN(img_size, latent_dim, lr)
    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs)
    trainer.fit(gan, datamodule)

    save_model(trainer, model_dir)


@click.command(name="train")
@click.option('--model_config_path', default='../config/model_config.yaml')
def train_command(model_config_path):
    logger = logging.getLogger(__name__)
    logger.info('Training DCGAN')
    train(model_config_path)
    logger.info('Model has been fitted and saved')


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train_command()
