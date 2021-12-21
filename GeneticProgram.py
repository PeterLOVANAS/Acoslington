import numpy as np
import SubSet as SS
import random
import string
# print(SS.Subset({"A", "a" , "B" , "b"}))

'''
def LowerUpper(A1 = str , A2 = str):
    try:
        if A1 == A2.lower() or A1 == A2.upper():
            return True
        else:
            return False
    except (ValueError  , TypeError):
        return "There has some issue"
'''
'''
def CheckEle(S1 = set):
    Slst=  list(S1)
    # lower = map(lambda ele : ele.lower() , Slst)
    for i in Slst:
        Test = Slst.count(i)
        if Test  == 1 :
            lowerlst = list(map(lambda ele : ele.lower() , Slst))
            print(lowerlst)
            for i1 in lowerlst:
                Test2 = lowerlst.count(i1)
                if Test2 ==  1:
                    pass
                elif Test2 > 1  :
                    return False
        elif Test > 1:
            return False


    Anslst = []
    # Check whether the element is the lowercase or copy of another element in the set or list or not
    # for i in Slst:

        # if i
        # if i in lower or i in Anslst:



# print(LowerUpper("z" , "S"))
# print(set("ABC"))
# print(len({5, 4,3}))
def Permutation_Expo(S = str):
    Strlen = len(S)
    Meiolen = Strlen / 2
    StrSet = set(S)
    Sub = SS.Subset(StrSet)  #It will return a list
    print(Sub)
    emplst = []
    for i in Sub:
        if len(i) == Meiolen:
            if CheckEle(i) == False:
                pass
            elif CheckEle(i) == True:
                emplst.append(i)
        elif len(i) != Meiolen:
            pass


print(Permutation_Expo("AaBb"))
'''

'''
def ExpoPermu(Gene = str):
    try:
        lstp = ["AA" , "Bb", "cc"] 
        lst0 = lstp.copy()
        # Split String into pair
        lstStr = list(lst0[0])
        lst1 = []
        lst2 = []
        # Fstr = lst0[0]
        for i1 in range(1):
            for i2 in lstp.pop(0):
                for i3 in i2:
                    # lst1.append(lstStr[i1])
                    # lst1.append(lstStr[i1])
'''
'''
lstp = ["AA" , "Bb", "Cc", "DD"]
lspc=  lstp.copy()
lstStr = list(lstp[0])
lst1 = [lstStr[0]]
lst2 = lst1.copy()
emplst=  []
lspc.pop(0)
for i in lspc :
    for i1 in i:
        x = list(map(lambda n : n + i1 , lst1))
        emplst.append(x)
    Ans1 = emplst[0] + emplst[-1]
    lst1 = Ans1.copy()
    Ans1.clear()
    emplst.clear()



print(lst1)
'''
'''
lstp = ["Aa" , "Bb","Cc"]
lspc=  lstp.copy()
lstStr = list(lstp[0])
Anslst = []


for i3 in [0,1]:
    lst1 = [lstStr[i3]]
    lst2 = lst1.copy()
    emplst=  []
    lspc.pop(0)
    for i in lspc :
        for i1 in i:
            x = list(map(lambda n : n + i1 , lst1))
            emplst.append(x)
        Ans1 = emplst[0] + emplst[-1]
        lst1 = Ans1.copy()
        Ans1.clear()
        emplst.clear()

    Anslst.append(lst1.copy())
    lst1.clear()
    lst2.clear()
    lspc = lstp.copy()
Answer = Anslst[0] + Anslst[-1]
print(Answer)
print(len(Answer))
'''

def Split(Str = str , num  = int):
    Anslst = []
    for i in range(0 , len(Str), num):
        Anslst.append(Str[i : i + num])
    return Anslst

def Permutation(Gene = str):
    lstp = Split(Gene, 2)
    lspc=  lstp.copy()
    lstStr = list(lstp[0])
    Anslst = []


    for i3 in [0,1]:
        lst1 = [lstStr[i3]]
        lst2 = lst1.copy()
        emplst=  []
        lspc.pop(0)
        for i in lspc :
            for i1 in i:
                x = list(map(lambda n : n + i1 , lst1))
                emplst.append(x)
            Ans1 = emplst[0] + emplst[-1]
            lst1 = Ans1.copy()
            Ans1.clear()
            emplst.clear()

        Anslst.append(lst1.copy())
        lst1.clear()
        lst2.clear()
        lspc = lstp.copy()
    Answer = Anslst[0] + Anslst[-1]
    return Answer


print(Permutation("AaBbCc"))
print(len(Permutation("AaBbCc")))

def ReWrite(Text = str):
    # LstText = list(Text)
    check = int(len(Text) /2)
    Anslst = []

    Newlst = Split(Text , check)
    lst1 = list(Newlst[0])
    lst2 = list(Newlst[1])
    for  i in range(len(lst1)):
        if lst1[i] == lst2[i].lower()  and lst1[i] != lst2[i]:
            Ans = lst2[i] + lst1[i]
            Anslst.append(Ans)
        else:
            Ans = lst1[i] + lst2[i]
            Anslst.append(Ans)
    RealAns = "".join(map(str , Anslst))
    return RealAns

def RandomAlpha(FM = str):
    Anslst = []

    for i in range(len(FM)):
        alpha=  random.choice(string.ascii_letters)
        Anslst.append(alpha)
    RealAns = "".join(map(str , Anslst))
    return RealAns


def Combine(F = str , M = str):
    try:
        Mainlst = []
        if len(F) > 2  or len(M) > 2 :
            F1 = Permutation(F)
            M1 = Permutation(M)
            Mainlst.append(F1)
            Mainlst.append(M1)

        elif len(F) or len(M) == 2 :
            F1 = list(F)
            M1 = list(M)
            Mainlst.append(F1)
            Mainlst.append(M1)

        # Initialize Matrix
        Inifig = RandomAlpha(F)
        Arr1 = np.full((len(Mainlst[0] ), len(Mainlst[1])) ,Inifig)

        # Put inside with combine
        for i1 in range(len(Mainlst[1])): #Column fix
            for i in range(len(Mainlst[0])): #Run rows
                Ft = Mainlst[0]
                Mt = Mainlst[1]
                Ans = Ft[i] + Mt[i1]
                RealAns = ReWrite(Ans)
                Arr1[i1 , i] = RealAns

        return Arr1

    except(ValueError , IndexError , TypeError , RuntimeError):
        return "Has some problem during the computation"

print(len("Aa"))
print(Combine("Aa" , "aa"))

print(Combine("BbSs" , "BbSs"))
# print(ReWrite("ABCdabCd"))

class itpt: #Interpretetion of an array
    def __init__(self, condi , Arr , Phenodict= dict ):
        self.condi = condi
        self.Arr = Arr
        self.Phenodict = Phenodict
    def Genotype(self):
        try:
            Arr = self.Arr
            condi = self.condi
            count = 0
            Shape = list(Arr.shape)
            for i in range(Shape[1]):
                    for i1 in range(Shape[0]):
                        if Arr[i1 , i] == condi:
                            count += 1
                        elif Arr[i1 , i] != condi:
                            pass
            return count
        except(RuntimeError , ValueError):
            return "Problem during computation"

# print(Genotype(Combine("Aa" , "aa")  , "Aa"))
# print(Combine("Aa" , "aa"))

    def Geno_to_Pheno(self): #Transform into Phenotype
        try:
            Phenodict = self.Phenodict
            condi = self.condi # "Aa" , "AABbCcdd"
            Phenolst = []
            if len(Phenodict.keys()) == len(condi):
                pass
            elif len(Phenodict.keys()) != len(condi):
                return f"Please change your dictionary keys to {len(condi)} keys"
            if len(condi) == 2:
                if condi[0] == condi[1] and condi[0].isupper():
                    Ans = Phenodict.get(condi[0])
                    Phenolst.append(Ans)
                elif condi[0] == condi[1] and condi[0].islower():
                    Ans = Phenodict.get(condi[0])
                    Phenolst.append(Ans)
                elif condi[0] != condi[1] and condi[1] == condi[0].lower():
                    Ans = Phenodict.get(condi[0])
                    Phenolst.append(Ans)

            elif len(condi) > 2 :
                Txtlst = Split(condi , int(len(condi) / 2))
                for i in Txtlst:
                    if i[0] == i[1] and i[0].isupper():
                        Ans = Phenodict.get(i[0])
                        Phenolst.append(Ans)
                    elif i[0] == i[1] and i[0].islower():
                        Ans = Phenodict.get(i[0])
                        Phenolst.append(Ans)
                    elif i[0] != i[1] and i[1] == i[0].lower():
                        Ans = Phenodict.get(i[0])
                        Phenolst.append(Ans)

            # RealAns = Phenolst.append(self.Genotype())
            return Phenolst
        except(RuntimeError  , KeyError, ValueError):
            return "Problem occurs"

    def Phenotype(self , condi = list ):
        try:
            Phenodict = self.Phenodict
            #Condi : ABCD [tall ,  yellow body , male ,red eyes]
            if len(Phenodict.keys()) == len(condi):
                pass
            elif len(Phenodict.keys()) != len(condi)*2:
                return f"Please change your dictionary keys to {len(condi) *2} keys"

            vallist = list(Phenodict.values())
            countch = 0
            notmatch = []
            for i4 in condi:
                if i4 in vallist:
                    countch += 1
                elif i4 not in vallist:
                    notmatch.append(i4)
            if countch == len(condi):
                pass
            elif countch != len(condi):
                return f"Please change some elements in the condition list , this is the elements you must to change: {notmatch}"

            Comlist = [] #CompareList
            count= 0
            # Read Array
            Matrix = self.Arr
            Shape = list(Matrix.shape)
            for i in range(Shape[1]):
                for i1 in range(Shape[0]):
                    MaVal = Matrix[i1 , i]
                    if len(MaVal) > 2:
                        Spi = Split(MaVal , int(len(MaVal) / 2))
                        for i2 in Spi:
                            if i2[0] == i2[1] and i2[0].isupper():
                                Val = Phenodict.get(i2[0])
                                Comlist.append(Val)
                            elif i2[0] == i2[1] and i2[0].islower():
                                Val = Phenodict.get(i2[0])
                                Comlist.append(Val)
                            elif i2[0] != i2[1] and i2[1] == i2[0].lower():
                                Val = Phenodict.get(i2[0])
                                Comlist.append(Val)

                    elif len(MaVal) == 2 :
                        if MaVal[0] == MaVal[1] and MaVal[0].isupper():
                                Val = Phenodict.get(MaVal[0])
                                Comlist.append(Val)
                        elif MaVal[0] == MaVal[1] and MaVal[0].islower():
                                Val = Phenodict.get(MaVal[0])
                                Comlist.append(Val)
                        elif MaVal[0] != MaVal[1] and MaVal[1] == MaVal[0].lower():
                                Val = Phenodict.get(MaVal[0])
                                Comlist.append(Val)

                    if Comlist == condi:
                        count +=1
                    elif Comlist != condi:
                        pass
                    Comlist.clear()
            return count
        except(RuntimeError ,ValueError ,TypeError , KeyError):
            return "There has some problem occurs during runtime"



    def Pheno_to_Geno(self , condi = list):
        try:
            Phenodict = self.Phenodict
            #Condi : ABCD [tall ,  yellow body , male ,red eyes]
            if len(Phenodict.keys()) == len(condi):
                pass
            elif len(Phenodict.keys()) != len(condi)*2:
                return f"Please change your dictionary keys to {len(condi) *2} keys"

            vallist = list(Phenodict.values())
            countch = 0
            notmatch = []
            for i4 in condi:
                if i4 in vallist:
                    countch += 1
                elif i4 not in vallist:
                    notmatch.append(i4)
            if countch == len(condi):
                pass
            elif countch != len(condi):
                return f"Please change some elements in the condition list , this is the elements you must to change: {notmatch}"

            Comlist = [] #CompareList
            Anslst = []
            # count= 0
            # Read Array
            Matrix = self.Arr
            Shape = list(Matrix.shape)
            for i in range(Shape[1]):
                for i1 in range(Shape[0]):
                    MaVal = Matrix[i1 , i]
                    if len(MaVal) > 2 :
                        Spi = Split(MaVal , int(len(MaVal) / 2))
                        for i2 in Spi:
                            if i2[0] == i2[1] and i2[0].isupper():
                                Val = Phenodict.get(i2[0])
                                if Val in condi:
                                    Comlist.append(i2)
                                elif Val not in condi:
                                    pass
                            elif i2[0] == i2[1] and i2[0].islower():
                                Val = Phenodict.get(i2[0])
                                if Val in condi:
                                    Comlist.append(i2)
                                elif Val not in condi:
                                    pass
                            elif i2[0] != i2[1] and i2[1] == i2[0].lower():
                                Val = Phenodict.get(i2[0])
                                if Val in condi:
                                    Comlist.append(i2)
                                elif Val not in condi:
                                    pass
                        if len(Comlist) == len(Split(MaVal , int(len(MaVal) / 2))):
                            Ans = "".join(map(str , Comlist))
                            Anslst.append(Ans)
                            Comlist.clear()
                        elif len(Comlist) != len(Split(MaVal , int(len(MaVal) / 2))):
                            Comlist.clear()


                    elif len(MaVal) == 2:
                        if MaVal[0] == MaVal[1] and MaVal[0].isupper():
                            Val = Phenodict.get(MaVal[0])
                            if Val in condi:
                                Comlist.append(MaVal)
                            elif Val not in condi:
                                pass
                        elif MaVal[0] == MaVal[1] and MaVal[0].islower():
                            Val = Phenodict.get(MaVal[0])
                            if Val in condi:
                                Comlist.append(MaVal)
                            elif Val not in condi:
                                pass
                        elif MaVal[0] != MaVal[1] and MaVal[1] == MaVal[0].lower():
                            Val = Phenodict.get(MaVal[0])
                            if Val in condi:
                                Comlist.append(MaVal)
                            elif Val not in condi:
                                pass
                        Anslst = Comlist
            return Anslst
        except(RuntimeError ,ValueError ,TypeError , KeyError):
            return "There has some problem occurs during runtime"




obj1 = itpt("BbSs" , Combine("BbSs", "BbSs") , {"B" : "Black" , "b": "Brown" , "S" : "Short" , "s" : "Long"} )
print(obj1.Genotype())
print(obj1.Geno_to_Pheno())
print(obj1.Phenotype(condi = ["Brown" , "Short"] ))
print(obj1.Pheno_to_Geno(condi = ["Black" , "Short"]  ))
# The count is specific for the BbSs  not the other genotype that have same Phenotype
# Note : Change the Phenodict to object attribute

obj2 = itpt("Aa" , Combine("aa" , "aa") , {"A" : "Tall" , "a" : "Short"})
print(obj2.Genotype()) #Output : 0 (No this Genotype, "Aa", inside the combination )
print(obj2.Geno_to_Pheno()) #Output : ['Tall']
print(obj2.Phenotype(condi = ["Short"])) #No releted to condi in the attribute
print(obj2.Pheno_to_Geno(condi = ["Short"]))

obj3 = itpt("AAbbccDd" , Combine("AABbCcDd" , "AaBbCcDd" ) , {"A" : "Tall" , "a" : "Short" , "B" : ""})
print(obj3.Genotype())
# print(Combine("AABbCcDd" , "AaBbCcDd"))
print("")
print(Combine("Aabb" , "AABb"))
obj4 = itpt("TtRR" , Combine("TtRr" , "TtRr") , {"T" : "Tall" , "t" : "Short" , "R" : "Yellow" , "r" : "red"})
print(obj4.Phenotype(condi = ["Tall" , "Yellow"]))
print(obj4.Phenotype(condi = ["Short" , "Yellow"]))
obj5 = itpt("AaBBccDd" , Combine("AaBbCcDD" , "AaBbccDd") , {"A" : "Tall"})
print(obj5.Genotype())
obj6 = itpt("AaBB" , Combine("AaBb" , "AaBb") , {"A" : "White" , "a" : "Black" , "B" :"Long" , "b" : "Short"})
print(obj6.Phenotype(condi = ["Black" , "Long"]))
# print(Combine("AaBbCcDD" , "AaBbccDd").shape)
# print(Combine("AaBbCcDD" , "AaBbccDd"))
# print(Combine("AABbCCDd" , "AABBccDD"))
# print(Permutation("AABbCCDd"))
'''
Anslst = []
for i in range(16):
    for i1 in range(16):
        Arr = Combine("AABbCcDd" , "AaBbCcDd" )
        if Arr[i1 , i] == "AAbbccDd":
            Anslst.append((i1 , i , Arr[i1 , i]))
        else:
            pass
print(Anslst)
Arr = Combine("AABbCcDd" , "AaBbCcDd" )
print(Arr)

'''
