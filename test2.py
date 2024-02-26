a = [True, True, True]
b = [False, False, False]
c = [True, False, True]

def testBoolan(x):
    if not any(x) != True and all(x) != True:
        print("has true and false")
    elif all(x) == True and not any(x) == False:
        print("all true")
    elif all(x) == False and not any(x) == True:
        print("all false")

testBoolan(c)