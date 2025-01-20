import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import responses

from bullpen.data_utils import PlayerLookup, Scraper, batch_scrape, load_data


class TestScraper:
    @pytest.fixture
    def scraper(self):
        return Scraper(2023)

    def test_repr(self, scraper):
        assert repr(scraper) == 'Scraper(year=2023)'

    def test_url(self, scraper):
        assert (
            scraper.url
            == 'https://www.baseball-reference.com/leagues/majors/2023-pitches-pitching.shtml'
        )

    @responses.activate
    def test_get_response_success(self, scraper):
        url = scraper.url
        responses.add(responses.GET, url, body='<html></html>', status=200)

        response = scraper.get_response(url)
        assert response.text == '<html></html>'

    @responses.activate
    def test_get_response_failure(self, scraper):
        url = scraper.url
        responses.add(responses.GET, url, status=404)

        with pytest.raises(Exception) as e:
            scraper.get_response(url)

        assert (
            str(e.value)
            == 'Failed to fetch https://www.baseball-reference.com/leagues/majors/2023-pitches-pitching.shtml. Status code: 404'
        )

    @responses.activate
    def test_parse_player_stats_table(self, scraper):
        html_content = """
        <html>
            <table>
                <tbody>
                    <tr class="thead"><th>Header</th></tr>
                    <tr><td>Data</td></tr>
                </tbody>
            </table>
        </html>
        """
        url = scraper.url
        responses.add(responses.GET, url, body=html_content, status=200, content_type='text/html')

        response = scraper.get_response(url)
        table = scraper.parse_player_stats_table(response)

        assert table is not None
        assert table.find('tbody') is not None
        assert table.find('th').text == 'Header'
        assert table.find('td').text == 'Data'

    @responses.activate
    def test_parse_player_stats_table_failure(self, scraper):
        html_content = '<html></html>'
        url = scraper.url
        responses.add(responses.GET, url, body=html_content, status=200, content_type='text/html')

        response = scraper.get_response(url)

        with pytest.raises(Exception) as e:
            scraper.parse_player_stats_table(response)

        assert (
            str(e.value)
            == "Could not find the 'Player Pitching Pitches' table on the page. Error: substring not found"
        )

    @responses.activate
    def test_parse_table_headers(self, scraper):
        html_content = """
        <html>
            <table>
                <thead>
                    <tr>
                        <th>Header 1</th>
                        <th>Header 2</th>
                        <th>Header 3</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Data 1</td>
                        <td>Data 2</td>
                        <td>Data 3</td>
                    </tr>
                </tbody>
            </table>
        </html>
        """
        url = scraper.url
        responses.add(responses.GET, url, body=html_content, status=200, content_type='text/html')
        response = scraper.get_response(url)
        table = scraper.parse_player_stats_table(response)

        headers = scraper.parse_table_headers(table)
        assert headers == ['Header 1', 'Header 2', 'Header 3']

    @responses.activate
    def test_parse_table_headers_with_first_blank(self, scraper):
        html_content = """
        <html>
            <table>
                <thead>
                    <tr>
                        <th></th>
                        <th>Header A</th>
                        <th>Header B</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Data A</td>
                        <td>Data B</td>
                        <td>Data C</td>
                    </tr>
                </tbody>
            </table>
        </html>
        """
        url = scraper.url
        responses.add(responses.GET, url, body=html_content, status=200, content_type='text/html')
        response = scraper.get_response(url)
        table = scraper.parse_player_stats_table(response)

        headers = scraper.parse_table_headers(table)
        assert headers == ['Header A', 'Header B']

    def test_convert_perc_to_float(self, scraper):
        s = pd.Series(['', '50.5%', '33.5%', '0%', '100%'])
        assert scraper.convert_perc_to_float(s).equals(pd.Series([np.nan, 0.505, 0.335, 0.0, 1.0]))

    def test_convert_spanish_letters(self, scraper):
        s = 'Edwin DÃ\xadaz'
        assert scraper.convert_spanish_letters(s) == 'Edwin Díaz'

    def test_format_data(self, scraper):
        raw_data = pd.DataFrame(
            {
                'Name': ['Edwin DÃ\xadaz'],
                'Str%': ['85.5%'],
                'L/Str': ['53.5%'],
                'S/Str': ['90.0%'],
                'F/Str': ['75.2%'],
                'I/Str': ['60.0%'],
                'AS/Str': ['50.0%'],
                'I/Bll': ['45.0%'],
                'AS/Pit': ['80.0%'],
                'Con': ['70.0%'],
                '1st%': ['40.0%'],
                '30%': ['90.2%'],
                '02%': ['20.0%'],
                'L/SO%': ['10.0%'],
            }
        )

        formatted_data = scraper.format_data(raw_data)
        assert formatted_data['Name'].iloc[0] == 'Edwin Díaz'
        assert formatted_data['Str%'].iloc[0] == 0.855
        assert formatted_data['L/Str'].iloc[0] == 0.535
        assert formatted_data['S/Str'].iloc[0] == 0.90
        assert formatted_data['F/Str'].iloc[0] == 0.752
        assert formatted_data['I/Str'].iloc[0] == 0.60
        assert formatted_data['AS/Str'].iloc[0] == 0.50
        assert formatted_data['I/Bll'].iloc[0] == 0.45
        assert formatted_data['AS/Pit'].iloc[0] == 0.80
        assert formatted_data['Con'].iloc[0] == 0.70
        assert formatted_data['1st%'].iloc[0] == 0.40
        assert formatted_data['30%'].iloc[0] == 0.902
        assert formatted_data['02%'].iloc[0] == 0.20
        assert formatted_data['L/SO%'].iloc[0] == 0.10

    @responses.activate
    def test_make_dataframe(self, scraper):
        html_content = """
        <html>
            <table>
                <thead>
                    <tr>
                        <th>Header 1</th>
                        <th>Header 2</th>
                        <th>Header 3</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Data 1A</td>
                        <td>Data 1B</td>
                        <td>Data 1C</td>
                    </tr>
                    <tr>
                        <td>Data 2A</td>
                        <td>Data 2B</td>
                        <td>Data 2C</td>
                    </tr>
                </tbody>
            </table>
        </html>
        """
        url = scraper.url
        responses.add(responses.GET, url, body=html_content, status=200, content_type='text/html')
        response = scraper.get_response(url)
        table = scraper.parse_player_stats_table(response)
        headers = scraper.parse_table_headers(table)

        expected_data = {
            'Header 1': ['Data 1A', 'Data 2A'],
            'Header 2': ['Data 1B', 'Data 2B'],
            'Header 3': ['Data 1C', 'Data 2C'],
            'Season': [2023, 2023],
        }
        expected_df = pd.DataFrame(expected_data)

        result = scraper.make_dataframe(table, headers)
        assert result.equals(expected_df)

    @responses.activate
    def test_scrape(self, scraper):
        html_content = """
        <html>
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Str%</th>
                        <th>L/Str</th>
                        <th>S/Str</th>
                        <th>F/Str</th>
                        <th>I/Str</th>
                        <th>AS/Str</th>
                        <th>I/Bll</th>
                        <th>AS/Pit</th>
                        <th>Con</th>
                        <th>1st%</th>
                        <th>30%</th>
                        <th>02%</th>
                        <th>L/SO%</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Jackmerius</td>
                        <td>85.5%</td>
                        <td>53.5%</td>
                        <td>90.0%</td>
                        <td>75.2%</td>
                        <td>60.0%</td>
                        <td>50.0%</td>
                        <td>45.0%</td>
                        <td>80.0%</td>
                        <td>70.0%</td>
                        <td>40.0%</td>
                        <td>90.2%</td>
                        <td>20.0%</td>
                        <td>10.0%</td>
                    </tr>
                    <tr>
                        <td>Tacktheratrix</td>
                        <td>90.0%</td>
                        <td>57.3%</td>
                        <td>80.0%</td>
                        <td>70.5%</td>
                        <td>50.0%</td>
                        <td>40.0%</td>
                        <td>30.0%</td>
                        <td>60.0%</td>
                        <td>65.0%</td>
                        <td>35.0%</td>
                        <td>85.1%</td>
                        <td>25.0%</td>
                        <td>15.0%</td>
                    </tr>
                </tbody>
            </table>
        </html>
        """
        url = scraper.url
        responses.add(responses.GET, url, body=html_content, status=200, content_type='text/html')
        expected_data = {
            'Name': ['Jackmerius', 'Tacktheratrix'],
            'Str%': [0.855, 0.90],
            'L/Str': [0.535, 0.573],
            'S/Str': [0.90, 0.80],
            'F/Str': [0.752, 0.705],
            'I/Str': [0.60, 0.50],
            'AS/Str': [0.50, 0.40],
            'I/Bll': [0.45, 0.30],
            'AS/Pit': [0.80, 0.60],
            'Con': [0.70, 0.65],
            '1st%': [0.40, 0.35],
            '30%': [0.902, 0.851],
            '02%': [0.20, 0.25],
            'L/SO%': [0.10, 0.15],
            'Season': [2023, 2023],
        }
        expected_df = pd.DataFrame(expected_data)

        result = scraper.scrape()
        assert result.equals(expected_df)


@responses.activate
def test_batch_scrape():
    mock_data = pd.DataFrame({'Year': [2023]})

    with patch.object(Scraper, 'scrape', return_value=mock_data):
        url_2023 = 'https://www.baseball-reference.com/leagues/majors/2023-pitches-pitching.shtml'
        url_2024 = 'https://www.baseball-reference.com/leagues/majors/2024-pitches-pitching.shtml'

        responses.add(responses.GET, url_2023, body='<html></html>', status=200)
        responses.add(responses.GET, url_2024, body='<html></html>', status=200)

        data = batch_scrape([2023, 2024])
        assert len(data) == 2
        assert data['Year'].tolist() == [2023, 2023]


def test_load_data_exception(tmp_path):
    mock_provided = pd.DataFrame(
        {
            'Name': ['Player A', 'Player B'],
            'Season': [2023, 2023],
            'Age': [30, 25],
            'Team': ['Team X', 'Team Y'],
        }
    )

    mock_suppl = pd.DataFrame(
        {
            'Name': ['Player A', 'Player A'],
            'Season': [2023, 2023],
            'Age': [30, 30],
            'Tm': ['Team X', 'Team X'],  # Duplicate row to cause a mismatch
        }
    )
    tmp_data_dir = tmp_path.joinpath('tmp-data')
    tmp_data_dir.mkdir()

    tmp_provided_path = tmp_data_dir.joinpath('k.csv')
    tmp_suppl_path = tmp_data_dir.joinpath('supplemental-stats.csv')

    mock_provided.to_csv(tmp_provided_path, index=False)
    mock_suppl.to_csv(tmp_suppl_path, index=False)
    with pytest.raises(Exception) as e:
        _ = load_data(
            provided_path=tmp_provided_path,
            supplemental_path=tmp_suppl_path,
            return_intermediaries=False,
        )
    assert str(e.value) == 'len(provided_data)=2 and len(merged)=3 do not match post merge!'


def test_load_data_no_intermediaries(tmp_path):
    mock_provided = pd.DataFrame(
        {
            'MLBAMID': {0: 670990, 1: 670990, 2: 670990},
            'PlayerId': {0: 19444, 1: 19444, 2: 19444},
            'Name': {0: 'Yohan Ramírez', 1: 'Yohan Ramírez', 2: 'Yohan Ramírez'},
            'Team': {0: '- - -', 1: '- - -', 2: '- - -'},
            'Age': {0: 29, 1: 28, 2: 27},
            'Season': {0: 2024, 1: 2023, 2: 2022},
            'TBF': {0: 208, 1: 176, 2: 167},
            'K%': {0: 0.21634615, 1: 0.19886364, 2: 0.19161677},
        }
    )

    mock_suppl = pd.DataFrame(
        {
            'Rk': {
                0: 836,
                1: 784,
                2: 785,
                3: 786,
                4: 787,
                5: 788,
                6: 770,
                7: 771,
                8: 772,
                9: 794,
                10: 795,
                11: 796,
                12: 797,
                13: 798,
                14: 799,
                15: 800,
            },
            'Name': {
                0: 'Yohan Ramírez',
                1: 'Yohan Ramírez',
                2: 'Yohan Ramírez',
                3: 'Yohan Ramírez',
                4: 'Yohan Ramírez',
                5: 'Yohan Ramírez',
                6: 'Yohan Ramírez',
                7: 'Yohan Ramírez',
                8: 'Yohan Ramírez',
                9: 'Yohan Ramírez',
                10: 'Yohan Ramírez',
                11: 'Yohan Ramírez',
                12: 'Yohan Ramírez',
                13: 'Yohan Ramírez',
                14: 'Yohan Ramírez',
                15: 'Yohan Ramírez',
            },
            'Age': {
                0: 26,
                1: 27,
                2: 27,
                3: 27,
                4: 27,
                5: 27,
                6: 28,
                7: 28,
                8: 28,
                9: 29,
                10: 29,
                11: 29,
                12: 29,
                13: 29,
                14: 29,
                15: 29,
            },
            'Tm': {
                0: 'SEA',
                1: 'TOT',
                2: 'TOT',
                3: 'SEA',
                4: 'CLE',
                5: 'PIT',
                6: 'TOT',
                7: 'PIT',
                8: 'CHW',
                9: 'TOT',
                10: 'TOT',
                11: 'TOT',
                12: 'NYM',
                13: 'BAL',
                14: 'LAD',
                15: 'BOS',
            },
            'IP': {
                0: 27.2,
                1: 37.1,
                2: 10.1,
                3: 8.1,
                4: 2.0,
                5: 27.0,
                6: 38.1,
                7: 34.1,
                8: 4.0,
                9: 44.9,
                10: 7.1,
                11: 37.2,
                12: 8.1,
                13: 6.0,
                14: 29.1,
                15: 1.1,
            },
            'PA': {
                0: 114,
                1: 167,
                2: 51,
                3: 40,
                4: 11,
                5: 116,
                6: 177,
                7: 156,
                8: 21,
                9: 208,
                10: 33,
                11: 175,
                12: 41,
                13: 24,
                14: 134,
                15: 9,
            },
            'Pit': {
                0: 436,
                1: 620,
                2: 198,
                3: 158,
                4: 40,
                5: 422,
                6: 708,
                7: 602,
                8: 106,
                9: 761,
                10: 133,
                11: 628,
                12: 147,
                13: 104,
                14: 481,
                15: 29,
            },
            'Pit/PA': {
                0: 3.82,
                1: 3.71,
                2: 3.88,
                3: 3.95,
                4: 3.64,
                5: 3.64,
                6: 4.0,
                7: 3.86,
                8: 5.05,
                9: 3.66,
                10: 4.03,
                11: 3.59,
                12: 3.59,
                13: 4.33,
                14: 3.59,
                15: 3.22,
            },
            'Str': {
                0: 275,
                1: 383,
                2: 117,
                3: 92,
                4: 25,
                5: 266,
                6: 428,
                7: 373,
                8: 55,
                9: 467,
                10: 77,
                11: 390,
                12: 93,
                13: 61,
                14: 297,
                15: 16,
            },
            'Str%': {
                0: 0.631,
                1: 0.618,
                2: 0.591,
                3: 0.5820000000000001,
                4: 0.625,
                5: 0.63,
                6: 0.605,
                7: 0.62,
                8: 0.519,
                9: 0.614,
                10: 0.579,
                11: 0.621,
                12: 0.633,
                13: 0.5870000000000001,
                14: 0.617,
                15: 0.552,
            },
            'L/Str': {
                0: 0.28,
                1: 0.308,
                2: 0.2739999999999999,
                3: 0.217,
                4: 0.48,
                5: 0.3229999999999999,
                6: 0.332,
                7: 0.303,
                8: 0.527,
                9: 0.315,
                10: 0.377,
                11: 0.303,
                12: 0.3229999999999999,
                13: 0.41,
                14: 0.296,
                15: 0.25,
            },
            'S/Str': {
                0: 0.258,
                1: 0.1639999999999999,
                2: 0.256,
                3: 0.326,
                4: 0.0,
                5: 0.124,
                6: 0.138,
                7: 0.145,
                8: 0.091,
                9: 0.171,
                10: 0.182,
                11: 0.1689999999999999,
                12: 0.247,
                13: 0.1639999999999999,
                14: 0.145,
                15: 0.25,
            },
            'F/Str': {
                0: 0.2289999999999999,
                1: 0.243,
                2: 0.214,
                3: 0.217,
                4: 0.2,
                5: 0.256,
                6: 0.266,
                7: 0.284,
                8: 0.145,
                9: 0.2269999999999999,
                10: 0.195,
                11: 0.233,
                12: 0.151,
                13: 0.213,
                14: 0.259,
                15: 0.125,
            },
            'I/Str': {
                0: 0.233,
                1: 0.285,
                2: 0.256,
                3: 0.239,
                4: 0.32,
                5: 0.297,
                6: 0.264,
                7: 0.268,
                8: 0.236,
                9: 0.287,
                10: 0.247,
                11: 0.295,
                12: 0.28,
                13: 0.213,
                14: 0.3,
                15: 0.375,
            },
            'AS/Str': {
                0: 0.72,
                1: 0.6920000000000001,
                2: 0.726,
                3: 0.7829999999999999,
                4: 0.52,
                5: 0.677,
                6: 0.6679999999999999,
                7: 0.6970000000000001,
                8: 0.473,
                9: 0.685,
                10: 0.623,
                11: 0.6970000000000001,
                12: 0.677,
                13: 0.59,
                14: 0.7040000000000001,
                15: 0.75,
            },
            'I/Bll': {
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.0,
                4: 0.0,
                5: 0.0,
                6: 0.0,
                7: 0.0,
                8: 0.0,
                9: 0.0,
                10: 0.0,
                11: 0.0,
                12: 0.0,
                13: 0.0,
                14: 0.0,
                15: 0.0,
            },
            'AS/Pit': {
                0: 0.4539999999999999,
                1: 0.427,
                2: 0.429,
                3: 0.456,
                4: 0.325,
                5: 0.427,
                6: 0.4039999999999999,
                7: 0.432,
                8: 0.245,
                9: 0.42,
                10: 0.361,
                11: 0.433,
                12: 0.429,
                13: 0.346,
                14: 0.435,
                15: 0.414,
            },
            'Con': {
                0: 0.6409999999999999,
                1: 0.762,
                2: 0.647,
                3: 0.583,
                4: 1.0,
                5: 0.8170000000000001,
                6: 0.794,
                7: 0.792,
                8: 0.8079999999999999,
                9: 0.75,
                10: 0.708,
                11: 0.757,
                12: 0.635,
                13: 0.722,
                14: 0.794,
                15: 0.667,
            },
            '1st%': {
                0: 0.544,
                1: 0.569,
                2: 0.569,
                3: 0.55,
                4: 0.636,
                5: 0.569,
                6: 0.5539999999999999,
                7: 0.59,
                8: 0.286,
                9: 0.51,
                10: 0.303,
                11: 0.5489999999999999,
                12: 0.537,
                13: 0.375,
                14: 0.552,
                15: 0.111,
            },
            '30%': {
                0: 0.053,
                1: 0.042,
                2: 0.02,
                3: 0.025,
                4: 0.0,
                5: 0.052,
                6: 0.045,
                7: 0.038,
                8: 0.095,
                9: 0.077,
                10: 0.061,
                11: 0.08,
                12: 0.122,
                13: 0.083,
                14: 0.067,
                15: 0.0,
            },
            '30c': {
                0: 6,
                1: 7,
                2: 1,
                3: 1,
                4: 0,
                5: 6,
                6: 8,
                7: 6,
                8: 2,
                9: 16,
                10: 2,
                11: 14,
                12: 5,
                13: 2,
                14: 9,
                15: 0,
            },
            '30s': {
                0: 5,
                1: 5,
                2: 1,
                3: 1,
                4: 0,
                5: 4,
                6: 2,
                7: 2,
                8: 0,
                9: 7,
                10: 1,
                11: 6,
                12: 2,
                13: 1,
                14: 4,
                15: 0,
            },
            '02%': {
                0: 0.281,
                1: 0.281,
                2: 0.1369999999999999,
                3: 0.175,
                4: 0.0,
                5: 0.345,
                6: 0.237,
                7: 0.263,
                8: 0.048,
                9: 0.207,
                10: 0.03,
                11: 0.24,
                12: 0.195,
                13: 0.042,
                14: 0.254,
                15: 0.0,
            },
            '02c': {
                0: 32,
                1: 47,
                2: 7,
                3: 7,
                4: 0,
                5: 40,
                6: 42,
                7: 41,
                8: 1,
                9: 43,
                10: 1,
                11: 42,
                12: 8,
                13: 1,
                14: 34,
                15: 0,
            },
            '02s': {
                0: 19,
                1: 26,
                2: 5,
                3: 5,
                4: 0,
                5: 21,
                6: 19,
                7: 19,
                8: 0,
                9: 30,
                10: 1,
                11: 29,
                12: 4,
                13: 1,
                14: 25,
                15: 0,
            },
            '02h': {
                0: 2,
                1: 3,
                2: 1,
                3: 1,
                4: 0,
                5: 2,
                6: 5,
                7: 5,
                8: 0,
                9: 1,
                10: 0,
                11: 1,
                12: 0,
                13: 0,
                14: 1,
                15: 0,
            },
            'L/SO': {
                0: 4,
                1: 12,
                2: 3,
                3: 2,
                4: 1,
                5: 9,
                6: 12,
                7: 9,
                8: 3,
                9: 9,
                10: 0,
                11: 9,
                12: 3,
                13: 0,
                14: 6,
                15: 0,
            },
            'S/SO': {
                0: 31,
                1: 20,
                2: 8,
                3: 8,
                4: 0,
                5: 12,
                6: 23,
                7: 22,
                8: 1,
                9: 36,
                10: 7,
                11: 29,
                12: 8,
                13: 6,
                14: 21,
                15: 1,
            },
            'L/SO%': {
                0: 0.114,
                1: 0.375,
                2: 0.273,
                3: 0.2,
                4: 1.0,
                5: 0.429,
                6: 0.3429999999999999,
                7: 0.29,
                8: 0.75,
                9: 0.2,
                10: 0.0,
                11: 0.237,
                12: 0.273,
                13: 0.0,
                14: 0.222,
                15: 0.0,
            },
            '3pK': {
                0: 6,
                1: 7,
                2: 2,
                3: 2,
                4: 0,
                5: 5,
                6: 4,
                7: 4,
                8: 0,
                9: 10,
                10: 0,
                11: 10,
                12: 2,
                13: 0,
                14: 8,
                15: 0,
            },
            '4pW': {
                0: 1,
                1: 1,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 5,
                7: 3,
                8: 2,
                9: 5,
                10: 1,
                11: 4,
                12: 2,
                13: 1,
                14: 2,
                15: 0,
            },
            'PAu': {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,
                10: 0,
                11: 0,
                12: 0,
                13: 0,
                14: 0,
                15: 0,
            },
            'Pitu': {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,
                10: 0,
                11: 0,
                12: 0,
                13: 0,
                14: 0,
                15: 0,
            },
            'Stru': {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,
                10: 0,
                11: 0,
                12: 0,
                13: 0,
                14: 0,
                15: 0,
            },
            'Season': {
                0: 2021,
                1: 2022,
                2: 2022,
                3: 2022,
                4: 2022,
                5: 2022,
                6: 2023,
                7: 2023,
                8: 2023,
                9: 2024,
                10: 2024,
                11: 2024,
                12: 2024,
                13: 2024,
                14: 2024,
                15: 2024,
            },
        }
    )

    expected_merged = pd.DataFrame(
        {
            'PlayerId': {0: 19444, 1: 19444, 2: 19444},
            'Team': {0: '- - -', 1: '- - -', 2: '- - -'},
            'Season': {0: 2022, 1: 2023, 2: 2024},
            'MLBAMID': {0: 670990, 1: 670990, 2: 670990},
            'Name': {0: 'Yohan Ramírez', 1: 'Yohan Ramírez', 2: 'Yohan Ramírez'},
            'Age': {0: 27, 1: 28, 2: 29},
            'TBF': {0: 167, 1: 176, 2: 208},
            'K%': {0: 0.19161677, 1: 0.19886364, 2: 0.21634615},
            'Rk': {0: 784, 1: 770, 2: 794},
            'IP': {0: 37.1, 1: 38.1, 2: 44.9},
            'PA': {0: 167, 1: 177, 2: 208},
            'Pit': {0: 620, 1: 708, 2: 761},
            'Pit/PA': {0: 3.71, 1: 4.0, 2: 3.66},
            'Str': {0: 383, 1: 428, 2: 467},
            'Str%': {0: 0.618, 1: 0.605, 2: 0.614},
            'L/Str': {0: 0.308, 1: 0.332, 2: 0.315},
            'S/Str': {0: 0.1639999999999999, 1: 0.138, 2: 0.171},
            'F/Str': {0: 0.243, 1: 0.266, 2: 0.2269999999999999},
            'I/Str': {0: 0.285, 1: 0.264, 2: 0.287},
            'AS/Str': {0: 0.6920000000000001, 1: 0.6679999999999999, 2: 0.685},
            'I/Bll': {0: 0.0, 1: 0.0, 2: 0.0},
            'AS/Pit': {0: 0.427, 1: 0.4039999999999999, 2: 0.42},
            'Con': {0: 0.762, 1: 0.794, 2: 0.75},
            '1st%': {0: 0.569, 1: 0.5539999999999999, 2: 0.51},
            '30%': {0: 0.042, 1: 0.045, 2: 0.077},
            '30c': {0: 7, 1: 8, 2: 16},
            '30s': {0: 5, 1: 2, 2: 7},
            '02%': {0: 0.281, 1: 0.237, 2: 0.207},
            '02c': {0: 47, 1: 42, 2: 43},
            '02s': {0: 26, 1: 19, 2: 30},
            '02h': {0: 3, 1: 5, 2: 1},
            'L/SO': {0: 12, 1: 12, 2: 9},
            'S/SO': {0: 20, 1: 23, 2: 36},
            'L/SO%': {0: 0.375, 1: 0.3429999999999999, 2: 0.2},
            '3pK': {0: 7, 1: 4, 2: 10},
            '4pW': {0: 1, 1: 5, 2: 5},
            'PAu': {0: 0, 1: 0, 2: 0},
            'Pitu': {0: 0, 1: 0, 2: 0},
            'Stru': {0: 0, 1: 0, 2: 0},
        }
    )
    tmp_data_dir = tmp_path.joinpath('tmp-data')
    tmp_data_dir.mkdir()

    tmp_provided_path = tmp_data_dir.joinpath('k.csv')
    tmp_suppl_path = tmp_data_dir.joinpath('supplemental-stats.csv')

    mock_provided.to_csv(tmp_provided_path, index=False)
    mock_suppl.to_csv(tmp_suppl_path, index=False)

    merged = load_data(
        provided_path=tmp_provided_path,
        supplemental_path=tmp_suppl_path,
        return_intermediaries=False,
    )
    assert merged.round(6).equals(expected_merged.round(6))


# def test_load_data():
#     mock_data = pd.DataFrame({'Name': ['Eduardo Rodriguez']})
#     with patch.object(pd, 'read_csv', return_value=mock_data) as mock_read_csv:
#         provided, supplemental = load_data()
#
#     assert 'Eduardo Rodríguez' in provided['Name'].values
#     mock_read_csv.assert_called()
#

# This is another way to do the above...keeeping as a nice to know
# The below is a little trickier to grasp...
#     the input, 'mock_read_csv' is a MagicMock used with the test function
# @patch('pandas.read_csv')
# def test_load_data(mock_read_csv):
#     mock_data = pd.DataFrame({'Name': ['Eduardo Rodriguez']})
#     mock_read_csv.return_value = mock_data
#
#     provided, supplemental = load_data()
#     assert 'Eduardo Rodríguez' in provided['Name'].values
#     mock_read_csv.assert_called()


class TestPlayerLookup:
    @pytest.fixture
    def mock_data(self):
        data = [
            {'Name': 'Edwin Díaz', 'MLBAMID': 12345, 'PlayerId': 67890},
            {'Name': 'John Doe', 'MLBAMID': 54321, 'PlayerId': 98765},
            {'Name': 'Jackmerius Tacktheratrix', 'MLBAMID': 671106, 'PlayerId': 27589},
            {'Name': 'Jackmerius Tacktheratrix', 'MLBAMID': 555555, 'PlayerId': 33333},
        ]
        return data

    @pytest.fixture
    def lookup(self, tmp_path, mock_data, monkeypatch):
        tmp_data_dir = tmp_path.joinpath('tmp-data')
        tmp_data_dir.mkdir()

        tmp_player_ids_path = tmp_data_dir.joinpath('tmp-player-ids.json')
        with open(tmp_player_ids_path, 'w') as fp:
            json.dump(mock_data, fp)

        lookup = PlayerLookup(datapath=str(tmp_player_ids_path.resolve()))
        monkeypatch.setattr(lookup, 'mapping', pd.DataFrame(mock_data))
        return lookup

    def test_repr(self, lookup):
        assert repr(lookup) == f'PlayerLookup(datapath={lookup.datapath!r})'

    def test_soruces(self, lookup):
        assert lookup.sources == {'fangraphs': 'PlayerId', 'mlb': 'MLBAMID'}

    def test_mapping(self, lookup, mock_data):
        assert lookup.mapping.equals(pd.DataFrame(mock_data))

    @pytest.mark.parametrize('source, expected', [('fangraphs', 'PlayerId'), ('mlb', 'MLBAMID')])
    def test__check_source(self, lookup, source, expected):
        result = lookup._check_source(source)
        assert result == expected

    def test__check_source_bad(self, lookup):
        with pytest.raises(ValueError) as e:
            lookup._check_source('bad')
        assert str(e.value) == "Unrecognized source='bad'. Must be one of ('mlb', 'fangraphs')."

    def test_get_name(self, lookup):
        assert lookup.get_name_from_id(12345) == 'Edwin Díaz'
        assert lookup.get_name_from_id(12345, source='mlb') == 'Edwin Díaz'
        assert lookup.get_name_from_id(67890, source='fangraphs') == 'Edwin Díaz'

        assert lookup.get_name_from_id(54321) == 'John Doe'
        assert lookup.get_name_from_id(54321, source='mlb') == 'John Doe'
        assert lookup.get_name_from_id(98765, source='fangraphs') == 'John Doe'

    def test_get_id(self, lookup):
        assert lookup.get_id_from_name('Edwin Díaz') == 12345
        assert lookup.get_id_from_name('Edwin Díaz', source='mlb') == 12345
        assert lookup.get_id_from_name('Edwin Díaz', source='fangraphs') == 67890

        assert lookup.get_id_from_name('John Doe') == 54321
        assert lookup.get_id_from_name('John Doe', source='mlb') == 54321
        assert lookup.get_id_from_name('John Doe', source='fangraphs') == 98765

    def test_get_name_repeats(self, lookup, mock_data):
        id = 55555
        expected = pd.DataFrame(mock_data)
        expected = expected.loc[expected.MLBAMID == id, ['MLBAMID', 'Name']].reset_index(drop=True)
        assert lookup.get_name_from_id(id).equals(expected)

    def test_get_id_repeats(self, lookup, mock_data):
        name = 'Jackmerius Tacktheratrix'
        expected = pd.DataFrame(mock_data)
        expected = expected.loc[expected.Name == name, ['Name', 'MLBAMID']].reset_index(drop=True)
        assert lookup.get_id_from_name(name).equals(expected)
