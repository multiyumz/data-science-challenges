# pylint: disable=missing-docstring

import csv

COUNTRIES_FILEPATH = "data/dictionary.csv"
MEDALS_FILEPATH = "data/winter.csv"


def most_decorated_athlete_ever():
    """Returns who won the most winter olympic games medals (gold/silver/bronze) ever"""
    athletes = {}
    with open(MEDALS_FILEPATH, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['Athlete'] not in athletes:
                athletes[row['Athlete']] = 1
            else:
                athletes[row['Athlete']] += 1

    best_athlete = None
    best_athlete_medal_count = 0

    for athlete, medals in athletes.items():
        if medals > best_athlete_medal_count:
            best_athlete_medal_count = medals
            best_athlete = athlete

    return best_athlete


def country_with_most_gold_medals(min_year, max_year):
    """Returns which country won the most gold medals between `min_year` and `max_year`"""
    countries = {}
    with open(MEDALS_FILEPATH, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            year = int(row['Year'])
            if min_year <= year <= max_year and row['Medal'] == "Gold":
                if row['Country'] not in countries:
                    countries[row['Country']] = 1
                else:
                    countries[row['Country']] += 1

    best_country = None
    best_country_gold_medals = 0

    for country, medals in countries.items():
        if medals > best_country_gold_medals:
            best_country_gold_medals = medals
            best_country = country

    with open(COUNTRIES_FILEPATH, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['Code'] == best_country:
                return row['Country']

    return best_country

def top_three_women_in_five_thousand_meters():
    """Returns the three women with the most 5000 meters medals(gold/silver/bronze)"""
    women = {}
    with open(MEDALS_FILEPATH, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['Gender'] == "Women" and row['Event'] == "5000M":
                if row['Athlete'] not in women:
                    women[row['Athlete']] = 1
                else:
                    women[row['Athlete']] += 1
        # women = sorted(women.items(), key=lambda k: k[1], reverse=True)
        # return list(map(lambda woman: woman[0], women[:3]))

        women = sorted(women.items(), key=lambda k: k[1], reverse=True)

        result = []
        for x in women:
            result.append(x[0])
        return result[:3]



print(top_three_women_in_five_thousand_meters())
