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


def test_load_data():
    mock_data = pd.DataFrame({'Name': ['Eduardo Rodriguez']})
    with patch.object(pd, 'read_csv', return_value=mock_data) as mock_read_csv:
        provided, supplemental = load_data()

    assert 'Eduardo Rodríguez' in provided['Name'].values
    mock_read_csv.assert_called()


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
        with pytest.raises(Exception) as e:
            lookup._check_source('bad')
        assert str(e.value) == "Unrecognized source='bad'. Must be one of ['mlb', 'fangraphs']."

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
