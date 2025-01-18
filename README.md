# mlb-pitcher-xK

## References
- [The Definitive Pitcher Expected K% Formula](https://fantasy.fangraphs.com/the-definitive-pitcher-expected-k-formula/)
- [TensorFlow Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Baseball Reference Pitcher Data](https://www.baseball-reference.com/leagues/majors/2014-pitches-pitching.shtml)

## Data
- 'Rk': arbitrary sorting rank based on selected column
- 'Name': player name
- 'Age': player age at midnight on June 30th of season year
- 'Tm': abbreviated team name
- 'IP': innings pitched
- 'PA': number of plate appearances for which pitch-by-pitch data exists
-       (note that inning-ending baserunning outs are counted as a PA, so these may be larger than batting PAs)
- 'Pit': number of pitches in the PA
- 'Pit/PA': pitches per plate appearance
- 'Str': strikes (includes both pitches in the zone and those swung at out of the zone)
- 'Str%': strike percentage (strikes / total pitches; intentional balls included)
- 'L/Str': looking strike percentage (strikes looking / total strikes)
- 'S/Str': swinging strike percentage (swinging strikes w/o contact / total strikes)
- 'F/Str': foul ball strike percentage (pitches fouled off / total strikes seen)
- 'I/Str': ball in play percentage (balls put into play including hr / total strikes)
- 'AS/Str': swung at strike percentage ((inplay + foul + swinging strikes) / total strikes)
- 'I/Bll': intentional ball percentage (intentional balls / all balls)
- 'AS/Pit': percentage of pitches swung at ((inplay + foul + swinging strikes) / (total pitches - intentional balls))
- 'Con': contact percentage ((foul + inplay strikes) / (inplay + foul + swinging strikes))
- '1st%': first pitch strike percentage (percent of play appearances being with 0-1 or with a ball inplay
- '30%': 3-0 count seen percentage (3-0 counts / PA)
- '30c': 3-0 counts seen
- '30s': 3-0 count strikes
- '02%': 0-2 count seen percentage (0-2 counts / PA)
- '02c': 0-2 counts seen
- '02s': 0-2 count strikes
- '02h': hits given up on an 0-2 count
- 'L/SO': strikeouts looking
- 'S/SO': strikeouts swinging
- 'L/SO%': strikeout looking percentage (stikeouts looking / all strikeouts)
- '3pK': 3 pitch strikeouts
- '4pW': 4 pitch walks
- 'PAu': Plate appearances for which data is unknown
- 'Pitu': Pitches for which ball-strike results are unknown
- 'Stru': Strikes for which detailed results are unknown
- 'Season': Year of stats
