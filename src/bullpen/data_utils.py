import html
import json
from functools import cached_property

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from ftfy import fix_text


class Scraper:
    """
    Strong assumption pulling from https://www.baseball-reference.com/.
    """

    def __init__(self, year):
        self.year = year

    @cached_property
    def url(self):
        return (
            f'https://www.baseball-reference.com/leagues/majors/{self.year}-pitches-pitching.shtml'
        )

    @staticmethod
    def get_response(url):
        print(f'scraping {url}...')
        response = requests.get(url)

        if not response.ok:
            raise Exception(f'Failed to fetch {url}. Status code: {response.status_code}')

        return response

    @staticmethod
    def parse_player_stats_table(response):
        """
        Helper function to extract individual pitcher data.

        Normally, a BeautifulSoup object can be created directly
        from the response.content object, but baseball ref made
        this hard to scrape. Team data is easy, player data is not.
        Hacky way to get around it for now:
            - split text on last table tag seen
            - only take up to closing table html tag
            - add both back manually
        """
        player_stats = response.text.split('<table')[-1]
        player_stats = player_stats[: player_stats.index('</table>')]
        player_stats = f'<table {player_stats} </table>'

        table = BeautifulSoup(player_stats, 'lxml')

        if not table:
            raise Exception("Could not find the 'Player Pitching Pitches' table on the page.")

        return table

    @staticmethod
    def parse_table_headers(table):
        headers = [header.text for header in table.find('thead').find_all('th')]

        # Exclude the first header if it's a blank column (e.g., rank column)
        if headers[0] == '':
            headers = headers[1:]
        return headers

    @staticmethod
    def convert_perc_to_float(series):
        return series.replace('', np.nan).str.rstrip('%').astype(float) / 100

    @staticmethod
    def convert_spanish_letters(text):
        return fix_text(html.unescape(text))

    def format_data(self, dataframe):
        perc_columns = [
            'Str%',
            'S/Str',
            'F/Str',
            'I/Str',
            'AS/Str',
            'I/Bll',
            'AS/Pit',
            'Con',
            '1st%',
            '30%',
            '02%',
            'L/SO%',
        ]

        # TODO: only one for now (if multiple need to refactor)
        spanish_column = 'Name'

        dataframe[perc_columns] = dataframe[perc_columns].apply(
            lambda col: self.convert_perc_to_float(col)
        )

        dataframe[spanish_column] = dataframe[spanish_column].apply(self.convert_spanish_letters)

        return dataframe

    def make_dataframe(self, table, headers):
        rows = table.find('tbody').find_all('tr', class_=lambda x: x != 'thead')

        data = []
        for row in rows:
            cols = row.find_all(['th', 'td'])
            cols_text = [col.text.strip().replace('\xa0', ' ').replace('*', '') for col in cols]
            data.append(cols_text)

        df = pd.DataFrame(data, columns=headers)
        df = df.assign(Season=self.year)
        return df

    def scrape(self):
        self.response = self.get_response(self.url)
        self.table = self.parse_player_stats_table(self.response)
        self.headers = self.parse_table_headers(self.table)
        data = self.make_dataframe(self.table, self.headers)
        data = self.format_data(data)
        return data


def batch_scrape(years):
    """
    Batch process/scrape multiple years of data.

    Parameters
    ----------
    years: listlike of int
        Years to pull from MLB Reference.

    Returns
    -------
    pandas.DataFrame of aggregated year data.
    """
    dfs = []
    for year in years:
        scraper = Scraper(year)
        data = scraper.scrape()
        dfs.append(data)
    return pd.concat(dfs)


def load_data(provided_path, supplemental_path):
    provided_data = pd.read_csv(provided_path)
    supplemental_data = pd.read_csv(supplemental_path)
    supplemental_data.Name = supplemental_data.Name.replace(
        {
            'Manny Banuelos': 'Manny Bañuelos',
            'Ralph Garza': 'Ralph Garza Jr.',
            'Luis Ortiz': 'Luis L. Ortiz',
            'Jose Hernandez': 'Jose E. Hernandez',
            'Hyeon-jong Yang': 'Hyeon-Jong Yang',
            'Adrián Martinez': 'Adrián Martínez',
        }
    )

    provided_data.Name = provided_data.Name.replace(
        {
            'Eduardo Rodriguez': 'Eduardo Rodríguez',
            'Jose Alvarez': 'José Álvarez',
            'Sandy Alcantara': 'Sandy Alcántara',
            'Carlos Martinez': 'Carlos Martínez',
            'Phillips Valdez': 'Phillips Valdéz',
            'Jovani Moran': 'Jovani Morán',
            'Jose Cuas': 'José Cuas',
            'Jorge Alcala': 'Jorge Alcalá',
            'Jhoan Duran': 'Jhoan Durán',
            'Jesus Tinoco': 'Jesús Tinoco',
            'Brent Honeywell': 'Brent Honeywell Jr.',
            'Adrian Morejon': 'Adrián Morejón',
        }
    )

    return provided_data, supplemental_data


class PlayerLookup:
    sources = {
        'mlb': 'MLBAMID',
        'fangraphs': 'PlayerId',
    }

    @cached_property
    def mapping(self):
        print('loading player ids...')
        with open('../data/player_ids.json', 'r') as fp:
            loaded = json.load(fp)
        return pd.DataFrame(loaded)

    def _check_source(self, source):
        source_col = self.sources.get(source)
        assert source_col, f'Unrecognized {source=!r}. Must be one of {list(self.sources)}.'
        return source_col

    def get_name(self, player_id, source='mlb'):
        """
        Retrieve player name by id.
        Source can be 'mlb' or 'fangraphs'
        """
        source_col = self._check_source(source)
        filter_ = self.mapping[self.mapping[source_col] == player_id]
        if len(filter_) == 1:
            return filter_['Name'].item()
        return filter_[[source_col, 'Name']].reset_index(drop=True)

    def get_id(self, player_name, source='mlb'):
        source_col = self._check_source(source)
        filter_ = self.mapping[self.mapping.Name == player_name]
        if len(filter_) == 1:
            return filter_[source_col].item()
        return filter_[[source_col, 'Name']].reset_index(drop=True)
