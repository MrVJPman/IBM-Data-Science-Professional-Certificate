#Write an if statement to determine if an album had a rating greater than 8. 
#Test it using the rating for the album “Back in Black” that had a rating of 8.5. 
#If the statement is true print "This album is Amazing!"

rating = 8
if rating > 8:
    print("This album is Amazing!")
    
    
#Write an if-else statement that performs the following.
#If the rating is larger then eight print “this album is amazing”. 
#If the rating is less than or equal to 8 print “this album is ok”.

if rating > 8:
    print("This album is Amazing!")
else:
    print("this album is ok")

#Write an if statement to determine if an album came out before 1980 or in the years: 1991 or 1993. 
#If the condition is true print out the year the album came out.

if year < 1980:
    print(year)
elif year == 1991 or year == 1993:
    print(year)
