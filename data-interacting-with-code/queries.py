# pylint: disable=missing-docstring, C0103
import sqlite3
conn = sqlite3.connect('data/movies.sqlite')
db = conn.cursor()


def directors_count(db):
    # return the number of directors contained in the database
    query = """SELECT COUNT(*)
                FROM directors"""
    db.execute(query)
    count = db.fetchone()
    return count[0]

def directors_list(db):
    # return the list of all the directors sorted in alphabetical order
    query = """SELECT name
            FROM directors
            ORDER BY name"""
    db.execute(query)
    directors = db.fetchall()
    # lst = []
    # for director in directors:
    #     lst.append(director[0])
    # return lst
    [director[0] for director in directors]

def love_movies(db):
    # return the list of all movies which contain the exact word "love"
    # in their title, sorted in alphabetical order
    query = """SELECT title FROM movies
            WHERE UPPER(title) LIKE '% LOVE %'
            OR UPPER(title) LIKE 'LOVE %'
            OR UPPER(title) LIKE '% LOVE'
            OR UPPER(title) LIKE 'LOVE'
            OR UPPER(title) LIKE '% LOVE''%'
            OR UPPER(title) LIKE '% LOVE.'
            OR UPPER(title) LIKE 'LOVE,%'
            ORDER BY title
            """
    db.execute(query)
    love_movies = db.fetchall()
    return [movie[0] for movie in love_movies]


def directors_named_like_count(db, name):
    # return the number of directors which contain a given word in their name
    query = """
            SELECT COUNT(*)
            FROM directors
            WHERE NAME LIKE ?
            """
    db.execute(query, (f"%{name}%",))
    names = db.fetchone()
    return names[0]


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
    longer_movies = db.fetchall()
    return [movie[0] for movie in longer_movies]


print(directors_named_like_count(db, 'Jones'))
# print("test")
