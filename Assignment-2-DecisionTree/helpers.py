def peek(data):
    x=0
    for line in data:
        if x<5:
            print(line)
            x+=1
