# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Xcryptoid/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Xcryptoid/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for how many GTA VI players will hold each PlayStation 5 trophy one week after launch. It reads every Exophase trophy-page read in a case, collapses the repeated reads into the few distinct refresh states Exophase actually published, and separates the two forces moving each trophy's earn rate: new players joining the tracked population, which dilutes rates, and existing players earning more trophies, which lifts them. Both the tracked-players count and each trophy's implied earner count are fitted in log-log space against time since release and divided at the target time to give the median rate, while the spread around it combines the fit's residual scatter, a random-walk term in the horizon, and a refresh-jump allowance that grows with the predicted influx of new players and stretches further downward than upward, because dilution pushes rates down.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=sm39bzkk&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
