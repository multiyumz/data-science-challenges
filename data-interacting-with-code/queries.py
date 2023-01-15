# pylint: disable=missing-docstring, C0103
import sqlite3

def directors_count(db):
    # return the number of directors contained in the database
    query = """SELECT COUNT(*)
                from directors"""
    db.execute(query)
    count = db.fetchone()
    return count[0]


def directors_list(db):
    # return the list of all the directors sorted in alphabetical order
    query = """SELECT name from directors
                ORDER BY name"""
    db.execute(query)
    directors = db.fetchall()
    list = []
    for director in directors:
        list.append(director[0])
    return list
    # [director[0] for director in directors]



def love_movies(db):
    # return the list of all movies which contain the exact word "love"
    # in their title, sorted in alphabetical order
    query = """
            SELECT title
            FROM movies
            WHERE UPPER(title) LIKE '% LOVE %'
            OR UPPER(title) LIKE 'LOVE %'
            OR UPPER(title) LIKE '% LOVE'
            OR UPPER(title) LIKE 'LOVE'
            OR UPPER(title) LIKE '% LOVE''%'
            OR UPPER(title) LIKE '% LOVE.'
            OR UPPER(title) LIKE 'LOVE,%'
            ORDER BY title"""

    db.execute(query)
    movies = db.fetchall()
    return [movie[0] for movie in movies]


def directors_named_like_count(db, name):
    # return the number of directors which contain a given word in their name
    query = """
            SELECT count(*)
            FROM directors
            WHERE name LIKE ?
            """
    db.execute(query, (f"%{name}%",))
    directors = db.fetchone()
    return directors[0]


def movies_longer_than(db, min_length):
    # return this list of all movies which are longer than a given duration,
    # sorted in the alphabetical order
    query = """
            SELECT title
            FROM movies
            WHERE minutes > ?
            ORDER BY title
            """

    db.execute(query, (min_length,))
    movies = db.fetchall()
    return [movie[0] for movie in movies]

conn = sqlite3.connect('data/movies.sqlite')
db = conn.cursor()


print(movies_longer_than(db, 400))
