# Worked Recommendation Examples

Three sample users, each with 15-30 ratings in the training set (a meaningful but not overly dense taste profile), selected via `numpy.random.default_rng(42)`. UBCF: k=20, similarity=pearson, min_support=10. IBCF: k=30, min_support=1.

## User 60

### Taste profile: top 5 highest-rated movies in training

| movieId | Title | Genres | Rating |
|---|---|---|---|
| 527 | Schindler's List (1993) | Drama|War | 5.00 |
| 858 | Godfather, The (1972) | Crime|Drama | 5.00 |
| 58559 | Dark Knight, The (2008) | Action|Crime|Drama|IMAX | 5.00 |
| 318 | Shawshank Redemption, The (1994) | Crime|Drama | 4.00 |
| 362 | Jungle Book, The (1994) | Adventure|Children|Romance | 4.00 |

### UBCF top 10 recommendations

| movieId | Title | Genres | Predicted score |
|---|---|---|---|
| 2858 | American Beauty (1999) | Drama|Romance | 5.00 |
| 293 | Léon: The Professional (a.k.a. The Professional) (Léon) (1994) | Action|Crime|Drama|Thriller | 5.00 |
| 296 | Pulp Fiction (1994) | Comedy|Crime|Drama|Thriller | 5.00 |
| 356 | Forrest Gump (1994) | Comedy|Drama|Romance|War | 5.00 |
| 1196 | Star Wars: Episode V - The Empire Strikes Back (1980) | Action|Adventure|Sci-Fi | 5.00 |
| 1197 | Princess Bride, The (1987) | Action|Adventure|Comedy|Fantasy|Romance | 5.00 |
| 4226 | Memento (2000) | Mystery|Thriller | 5.00 |
| 4262 | Scarface (1983) | Action|Crime|Drama | 5.00 |
| 7361 | Eternal Sunshine of the Spotless Mind (2004) | Drama|Romance|Sci-Fi | 5.00 |
| 68157 | Inglourious Basterds (2009) | Action|Drama|War | 5.00 |

### IBCF top 10 recommendations

| movieId | Title | Genres | Predicted score |
|---|---|---|---|
| 168252 | Logan (2017) | Action|Sci-Fi | 4.35 |
| 926 | All About Eve (1950) | Drama | 4.30 |
| 1203 | 12 Angry Men (1957) | Drama | 4.29 |
| 1272 | Patton (1970) | Drama|War | 4.28 |
| 898 | Philadelphia Story, The (1940) | Comedy|Drama|Romance | 4.25 |
| 56782 | There Will Be Blood (2007) | Drama|Western | 4.25 |
| 215 | Before Sunrise (1995) | Drama|Romance | 4.25 |
| 2501 | October Sky (1999) | Drama | 4.25 |
| 922 | Sunset Blvd. (a.k.a. Sunset Boulevard) (1950) | Drama|Film-Noir|Romance | 4.22 |
| 246 | Hoop Dreams (1994) | Documentary | 4.20 |

### Overlap

No overlap -- UBCF and IBCF recommended entirely disjoint movie sets.

## User 394

### Taste profile: top 5 highest-rated movies in training

| movieId | Title | Genres | Rating |
|---|---|---|---|
| 50 | Usual Suspects, The (1995) | Crime|Mystery|Thriller | 5.00 |
| 110 | Braveheart (1995) | Action|Drama|War | 5.00 |
| 225 | Disclosure (1994) | Drama|Thriller | 4.00 |
| 356 | Forrest Gump (1994) | Comedy|Drama|Romance|War | 4.00 |
| 457 | Fugitive, The (1993) | Thriller | 4.00 |

### UBCF top 10 recommendations

| movieId | Title | Genres | Predicted score |
|---|---|---|---|
| 4973 | Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001) | Comedy|Romance | 4.40 |
| 1262 | Great Escape, The (1963) | Action|Adventure|Drama|War | 4.32 |
| 122882 | Mad Max: Fury Road (2015) | Action|Adventure|Sci-Fi|Thriller | 4.17 |
| 2019 | Seven Samurai (Shichinin no samurai) (1954) | Action|Adventure|Drama | 4.16 |
| 2288 | Thing, The (1982) | Action|Horror|Sci-Fi|Thriller | 4.14 |
| 858 | Godfather, The (1972) | Crime|Drama | 4.09 |
| 1215 | Army of Darkness (1993) | Action|Adventure|Comedy|Fantasy|Horror | 4.06 |
| 497 | Much Ado About Nothing (1993) | Comedy|Romance | 4.05 |
| 1250 | Bridge on the River Kwai, The (1957) | Adventure|Drama|War | 4.05 |
| 1204 | Lawrence of Arabia (1962) | Adventure|Drama|War | 4.01 |

### IBCF top 10 recommendations

| movieId | Title | Genres | Predicted score |
|---|---|---|---|
| 3681 | For a Few Dollars More (Per qualche dollaro in più) (1965) | Action|Drama|Thriller|Western | 4.98 |
| 1228 | Raging Bull (1980) | Drama | 4.94 |
| 2019 | Seven Samurai (Shichinin no samurai) (1954) | Action|Adventure|Drama | 4.93 |
| 115713 | Ex Machina (2015) | Drama|Sci-Fi|Thriller | 4.88 |
| 1252 | Chinatown (1974) | Crime|Film-Noir|Mystery|Thriller | 4.85 |
| 2858 | American Beauty (1999) | Drama|Romance | 4.83 |
| 48774 | Children of Men (2006) | Action|Adventure|Drama|Sci-Fi|Thriller | 4.82 |
| 318 | Shawshank Redemption, The (1994) | Crime|Drama | 4.80 |
| 750 | Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) | Comedy|War | 4.80 |
| 1136 | Monty Python and the Holy Grail (1975) | Adventure|Comedy|Fantasy | 4.79 |

### Overlap

1 movie(s) appear in both UBCF's and IBCF's top-10: Seven Samurai (Shichinin no samurai) (1954)

## User 463

### Taste profile: top 5 highest-rated movies in training

| movieId | Title | Genres | Rating |
|---|---|---|---|
| 7153 | Lord of the Rings: The Return of the King, The (2003) | Action|Adventure|Drama|Fantasy | 5.00 |
| 110 | Braveheart (1995) | Action|Drama|War | 4.50 |
| 1221 | Godfather: Part II, The (1974) | Crime|Drama | 4.50 |
| 1552 | Con Air (1997) | Action|Adventure|Thriller | 4.50 |
| 36529 | Lord of War (2005) | Action|Crime|Drama|Thriller|War | 4.50 |

### UBCF top 10 recommendations

| movieId | Title | Genres | Predicted score |
|---|---|---|---|
| 858 | Godfather, The (1972) | Crime|Drama | 4.96 |
| 1262 | Great Escape, The (1963) | Action|Adventure|Drama|War | 4.86 |
| 1223 | Grand Day Out with Wallace and Gromit, A (1989) | Adventure|Animation|Children|Comedy|Sci-Fi | 4.83 |
| 5618 | Spirited Away (Sen to Chihiro no kamikakushi) (2001) | Adventure|Animation|Fantasy | 4.78 |
| 1196 | Star Wars: Episode V - The Empire Strikes Back (1980) | Action|Adventure|Sci-Fi | 4.78 |
| 79132 | Inception (2010) | Action|Crime|Drama|Mystery|Sci-Fi|Thriller|IMAX | 4.74 |
| 1287 | Ben-Hur (1959) | Action|Adventure|Drama | 4.73 |
| 1250 | Bridge on the River Kwai, The (1957) | Adventure|Drama|War | 4.73 |
| 4235 | Amores Perros (Love's a Bitch) (2000) | Drama|Thriller | 4.72 |
| 2571 | Matrix, The (1999) | Action|Sci-Fi|Thriller | 4.70 |

### IBCF top 10 recommendations

| movieId | Title | Genres | Predicted score |
|---|---|---|---|
| 1261 | Evil Dead II (Dead by Dawn) (1987) | Action|Comedy|Fantasy|Horror | 4.38 |
| 60684 | Watchmen (2009) | Action|Drama|Mystery|Sci-Fi|Thriller|IMAX | 4.34 |
| 246 | Hoop Dreams (1994) | Documentary | 4.33 |
| 2580 | Go (1999) | Comedy|Crime | 4.27 |
| 318 | Shawshank Redemption, The (1994) | Crime|Drama | 4.23 |
| 54997 | 3:10 to Yuma (2007) | Action|Crime|Drama|Western | 4.21 |
| 1228 | Raging Bull (1980) | Drama | 4.20 |
| 168252 | Logan (2017) | Action|Sci-Fi | 4.19 |
| 58559 | Dark Knight, The (2008) | Action|Crime|Drama|IMAX | 4.18 |
| 38061 | Kiss Kiss Bang Bang (2005) | Comedy|Crime|Mystery|Thriller | 4.18 |

### Overlap

No overlap -- UBCF and IBCF recommended entirely disjoint movie sets.
