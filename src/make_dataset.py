import click
import logging
import yaml
from pathlib import Path
from bing_image_downloader import downloader


def download(queries, data_path):
    for query in queries:
        downloader.download(query, limit=100,
                            output_dir=data_path,
                            timeout=10, verbose=True)


@click.command(name="make_dataset")
@click.option('--load_config_path', default='../config/load_config.yaml')
def main(load_config_path):
    """ Loads images to raw data (../raw).
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading images raw data')

    with open(load_config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    query_list = config['queries']
    img_path = config['data_path']

    download(query_list, img_path)
    logger.info('All files are downloaded')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    main()
