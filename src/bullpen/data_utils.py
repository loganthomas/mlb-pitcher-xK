from functools import cached_property

import pandas as pd
import requests
from bs4 import BeautifulSoup


class MLBReferenceScraper:
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

    def make_dataframe(self, table, headers):
        rows = table.find('tbody').find_all('tr', class_=lambda x: x != 'thead')

        data = []
        for row in rows:
            cols = row.find_all(['th', 'td'])
            cols_text = [col.text.strip() for col in cols]
            data.append(cols_text)

        df = pd.DataFrame(data, columns=headers)
        df = df.assign(Season=self.year)
        return df

    def scrape(self):
        self.response = self.get_response(self.url)
        self.table = self.parse_player_stats_table(self.response)
        self.headers = self.parse_table_headers(self.table)
        data = self.make_dataframe(self.table, self.headers)
        return data
