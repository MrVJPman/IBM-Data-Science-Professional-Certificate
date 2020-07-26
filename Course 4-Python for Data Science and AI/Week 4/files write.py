# Write line to file

with open('/resources/data/Example2.txt', 'w') as writefile:
    writefile.write("This is line A")

# Read file

with open('/resources/data/Example2.txt', 'r') as testwritefile:
    print(testwritefile.read())
    
# Write lines to file

with open('/resources/data/Example2.txt', 'w') as writefile:
    writefile.write("This is line A\n")
    writefile.write("This is line B\n")

# Check whether write to file

with open('/resources/data/Example2.txt', 'r') as testwritefile:
    print(testwritefile.read())
    
# Write a new line to text file

with open('/resources/data/Example2.txt', 'a') as testwritefile:
    testwritefile.write("This is line C\n")
    
# Verify if the new line is in the text file

with open('/resources/data/Example2.txt', 'r') as testwritefile:
    print(testwritefile.read())

# Sample list of text

Lines = ["This is line A\n", "This is line B\n", "This is line C\n"]
Lines

# Write the strings in the list to text file

with open('Example2.txt', 'w') as writefile:
    for line in Lines:
        print(line)
        writefile.write(line)

# Verify if writing to file is successfully executed

with open('Example2.txt', 'r') as testwritefile:
    print(testwritefile.read())
    
# Append the line to the file

with open('Example2.txt', 'a') as testwritefile:
    testwritefile.write("This is line D\n")
    
# Verify if the appending is successfully executed

with open('Example2.txt', 'r') as testwritefile:
    print(testwritefile.read())
    


# Copy file to another

with open('Example2.txt','r') as readfile:
    with open('Example3.txt','w') as writefile:
        for line in readfile:
            writefile.write(line)
            
# Verify if the copy is successfully executed

with open('Example3.txt','r') as testwritefile:
    print(testwritefile.read())