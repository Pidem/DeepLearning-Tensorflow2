def PartieEntiere(x):
    if x>=0:
        return int(x)
    else:
        return -int(-x)-1

def PartieDecimale(x):
    return x-PartieEntiere(x)


def PacDecVers(x):
    liste=[]
    x=PartieDecimale(x)
    while x!=0:
        x = x * 2
        r = PartieEntiere(x)

        x = PartieDecimale(x)


        print(x)
        lis = [r]
        liste.extend(lis)
    return liste

print(PartieEntiere(.5))