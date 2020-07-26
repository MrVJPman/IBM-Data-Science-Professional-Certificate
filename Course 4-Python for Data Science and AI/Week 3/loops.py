#Write a for loop the prints out all the element between -5 and 5 using the range function.

for number in range(-5, 6):
    print(number)
    
#Print the elements of the following list: Genres=[ 'rock', 'R&B', 'Soundtrack', 'R&B', 'soul', 'pop']
#Make sure you follow Python conventions.

Genres=[ 'rock', 'R&B', 'Soundtrack', 'R&B', 'soul', 'pop']

for element in Genres:
    print(element)
    
#Write a for loop that prints out the following list: squares=['red', 'yellow', 'green', 'purple', 'blue']
squares=['red', 'yellow', 'green', 'purple', 'blue']
for colour in squares:
    print(colour)
    
#Write a while loop to display the values of the Rating of an album playlist stored in the list PlayListRatings. 
#If the score is less than 6, exit the loop. 
#The list PlayListRatings is given by: PlayListRatings = [10, 9.5, 10, 8, 7.5, 5, 10, 10]

PlayListRatings = [10, 9.5, 10, 8, 7.5, 5, 10, 10]
index = 0
while index < len(PlayListRatings):
    if PlayListRatings[index] < 6:
        break
    print (PlayListRatings[index])
    index += 1
    
#Write a while loop to copy the strings 'orange' of the list squares to the list new_squares.
#Stop and exit the loop if the value on the list is not 'orange':

# Write your code below and press Shift+Enter to execute

squares = ['orange', 'orange', 'purple', 'blue ', 'orange']
new_squares = []

index = 0 
while squares[index] == "orange":
    new_squares.append(squares[index])
    index += 1