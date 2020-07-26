# sample tuple

genres_tuple = ("pop", "rock", "soul", "hard rock", "soft rock", \
                "R&B", "progressive rock", "disco") 
genres_tuple

#Find the length of the tuple, genres_tuple:
len(genres_tuple)

#Access the element, with respect to index 3:

genres_tuple[3]

#Use slicing to obtain indexes 3, 4 and 5:
    
genres_tuple[3:6]

#Find the first two elements of the tuple genres_tuple:

genres_tuple[0]    
genres_tuple[1]

#Find the first index of "disco":

genres_tuple.index("disco")

#Generate a sorted List from the Tuple C_tuple=(-5, 1, -3):

C_tuple=(-5, 1, -3)
sorted_C_tuple=sorted(C_tuple)