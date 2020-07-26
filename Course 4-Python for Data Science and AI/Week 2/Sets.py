#Convert the list ['rap','house','electronic music', 'rap'] to a set:

set(['rap','house','electronic music', 'rap'])

#Consider the list A = [1, 2, 2, 1] and set B = set([1, 2, 2, 1]), does sum(A) = sum(B)

A = [1, 2, 2, 1]
B = set([1, 2, 2, 1])

sum(A) == sum(B)

#Create a new set album_set3 that is the union of album_set1 and album_set2:

album_set1 = set(["Thriller", 'AC/DC', 'Back in Black'])
album_set2 = set([ "AC/DC", "Back in Black", "The Dark Side of the Moon"])

album_set3 = album_set1.union(album_set2)

#Find out if album_set1 is a subset of album_set3

album_set1.issubset(album_set3)

