# pylint: disable=C0103, missing-docstring

import sqlite3

conn = sqlite3.connect('data/movies.sqlite')
db = conn.cursor()

def detailed_movies(db):
    '''return the list of movies with their genres and director name'''
    query = """
            SELECT
            movies.title,
            movies.genres,
            directors.name
            FROM movies
            JOIN directors on movies.director_id = directors.id
            """
    db.execute(query)
    movies = db.fetchall()
    return movies


def late_released_movies(db):
    '''return the list of all movies released after their director death'''
    query = """
            SELECT
            movies.title
            FROM movies
            JOIN directors on movies.director_id = directors.id
            WHERE (movies.start_year - directors.death_year) > 0
            """
    db.execute(query)
    latest = db.fetchall()
    return [late[0] for late in latest]


def stats_on(db, genre_name):
    '''return a dict of stats for a given genre'''
    query = """
            SELECT
            movies.genres,
            COUNT(*),
            ROUND(AVG(minutes),2)
            FROM movies
            WHERE movies.genres = ?
            """
    db.execute(query, (genre_name,))
    stats = db.fetchone()
    # return {
    #     'genre': stats[0],
    #     'number_of_movies': stats[1],
    #     'avg_length': stats[2]
    # }
    return dict({'genre': stats[0],
                 'number_of_movies': stats[1],
                 'avg_length': stats[2]})

def top_five_directors_for(db, genre_name):
    '''return the top 5 of the directors with the most movies for a given genre'''
    query = """
            SELECT
            directors.name,
            COUNT(*) number_of_movies
            FROM movies
            JOIN directors on movies.director_id = directors.id
            WHERE movies.genres = ?
            GROUP BY directors.name
            ORDER BY number_of_movies DESC, directors.name
            LIMIT 5
            """
    db.execute(query, (genre_name,))
    top_directors = db.fetchall()
    return top_directors

def movie_duration_buckets(db):
    '''return the movie counts grouped by bucket of 30 min duration'''
    query = """
        SELECT (minutes / 30+1)*30 time_range, COUNT(*)
        FROM movies
        WHERE minutes IS NOT NULL
        GROUP BY time_range
        """
    db.execute(query)
    buckets = db.fetchall()
    return buckets


def top_five_youngest_newly_directors(db):
    '''return the top 5 youngest directors when they direct their first movie'''
    query = """
            SELECT
            directors.name,
            (movies.start_year - directors.birth_year) age
            FROM movies
            JOIN directors ON movies.director_id = directors.id
            GROUP BY directors.name
            HAVING age IS NOT NULL
            ORDER BY age
            LIMIT 5
            """
    db.execute(query)
    top_5 = db.fetchall()
    return top_5

# print(stats_on(db, 'Action,Adventure,Comedy'))
print(detailed_movies(db))
